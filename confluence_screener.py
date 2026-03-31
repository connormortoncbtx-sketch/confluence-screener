# confluence_screener.py
# ---------------------------------------------
# Screener with Finnhub bars (free), paced at ~50 req/min.
# Features:
# - RSI/MACD/EMA/Bollinger/Volume + 50-SMA trend bias
# - Confluence scoring -> BUY/SELL alerts
# - Discord notifications with reason codes
# - BUY→SELL P&L since last BUY (persisted state)
# - Targets ensemble: Fib + ATR + Pivots + Donchian (T1/T2/Stop)
# - Robust ticker CSV loader (filters ETFs/SPACs by name when present)
# - Provider switch: DATA_PROVIDER = "finnhub" (default) or "alpaca" (IEX)
# - Rate-limit pacing for Finnhub (50/min) via thread-safe RateLimiter
# - Parallel Finnhub fetches via ThreadPoolExecutor
# - Cooldown tracking persisted in state.json (survives across runs)
# - Retry with exponential backoff on Finnhub transient errors
# - Discord chunking on signal boundaries (never splits mid-line)

# NOTE ON PROVIDER CHOICE:
# Switching DATA_PROVIDER=alpaca in screener.yml is the single biggest speed
# win available. Alpaca batches up to ~120 symbols per request, so 600 tickers
# becomes ~5 requests instead of 600. Finnhub free tier is 60 req/min with
# 1 symbol per request — parallelism helps but the rate cap is the hard ceiling.

import os, time, json, csv, ssl, smtplib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading

import pandas as pd
import numpy as np
import requests

# --------- ENV / Paths ----------
DATA_PROVIDER         = os.getenv("DATA_PROVIDER", "finnhub").lower()
FINNHUB_API_KEY       = os.getenv("FINNHUB_API_KEY")
ALPACA_API_KEY        = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY     = os.getenv("ALPACA_SECRET_KEY")
DISCORD_WEBHOOK_URL   = os.getenv("DISCORD_WEBHOOK_URL")

CSV_PATH   = Path("nasdaqlisted.csv")
STATE_DIR  = Path("state"); STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = STATE_DIR / "state.json"
ROTATE_PATH = STATE_DIR / "scan_offset.json"

# --------- Scan Config ----------
TIMEFRAME_MINUTES = 5
LOOKBACK          = 400
BUY_THRESHOLD     = 60
SELL_THRESHOLD    = 60

# Finnhub free tier: 60 req/min. Use 50 for safety.
FINNHUB_CALLS_PER_MIN = 50

# Symbols per run (rotation window).
SCAN_LIMIT = int(os.getenv("SCAN_LIMIT", "600"))

# Cooldown between repeat signals for the same symbol (seconds).
# FIX: was previously reset to 0 every run — now persisted in state.json.
COOLDOWN_SEC = 30 * 60

# Confluence weights
W = dict(RSI=25, MACD=25, EMA=25, BB=15, VOL=10)

# --------- Thread-safe Rate Limiter ----------
class RateLimiter:
    """
    Serialises dispatch to max_per_min calls/minute across any number of threads.
    Each thread that calls .wait() blocks until its turn, spaced at 60/max_per_min
    seconds apart. This keeps total dispatch rate at or below the cap while allowing
    multiple HTTP responses to be in-flight simultaneously.
    """
    def __init__(self, max_per_min: int):
        self._delay = 60.0 / max_per_min
        self._lock  = threading.Lock()
        self._last  = 0.0

    def wait(self):
        with self._lock:
            now  = time.monotonic()
            gap  = self._last + self._delay - now
            if gap > 0:
                time.sleep(gap)
            self._last = time.monotonic()

_finnhub_limiter = RateLimiter(FINNHUB_CALLS_PER_MIN)

# ---------- Discord ----------
def discord(msg: str):
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": msg}, timeout=15)
    except Exception as e:
        print(f"[discord] post failed: {e}")

# ---------- State ----------
def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[state] read error: {e}")
    return {}

def save_state(state: dict):
    try:
        STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[state] write error: {e}")

def load_offset() -> int:
    if ROTATE_PATH.exists():
        try:
            return json.loads(ROTATE_PATH.read_text()).get("offset", 0)
        except:
            return 0
    return 0

def save_offset(offset: int):
    try:
        ROTATE_PATH.write_text(json.dumps({"offset": offset}))
    except:
        pass

# ---------- CSV tickers loader ----------
def read_tickers_from_csv(path: Path):
    if not path.exists():
        raise SystemExit(f"Ticker file not found: {path}")
    with open(path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        try:
            delim = csv.Sniffer().sniff(sample, delimiters="|,\t;").delimiter
        except Exception:
            delim = ","
    df = pd.read_csv(path, delimiter=delim)

    candidates = [c for c in df.columns if c.lower() in ["symbol","ticker","symbols","tickers"]]
    col = candidates[0] if candidates else df.columns[0]

    name_cols = [c for c in df.columns if c.lower() in ["security name","name","description","company name","issuer name"]]
    if name_cols:
        nm = df[name_cols[0]].astype(str).str.lower()
        mask = ~nm.str.contains(r"\b(?:etf|trust|fund|warrant|unit|spac)\b", regex=True, na=False)
        df = df[mask]

    ser = (df[col].astype(str).str.strip()
           .str.replace(r"[^\w\.-]", "", regex=True)
           .replace("", pd.NA).dropna().drop_duplicates())
    tickers = ser.tolist()
    print(f"[tickers] loaded {len(tickers)} symbols from {path.name}")
    return tickers

# ---------- Index normalize ----------
def normalize_to_multi(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure MultiIndex ('symbol','timestamp') with symbol first."""
    if df is None or df.empty:
        return df
    if getattr(df.index, "nlevels", 1) == 2:
        lvl0 = df.index.get_level_values(0)
        lvl1 = df.index.get_level_values(1)
        lvl0_is_dt = pd.api.types.is_datetime64_any_dtype(lvl0)
        lvl1_is_dt = pd.api.types.is_datetime64_any_dtype(lvl1)
        if lvl0_is_dt and not lvl1_is_dt:
            df = df.swaplevel(0, 1)
        df.index = df.index.set_names(["symbol", "timestamp"])
        return df.sort_index()

    if not df.index.name:
        df.index.name = "timestamp"
    if "symbol" in df.columns:
        return df.reset_index().set_index(["symbol","timestamp"]).sort_index()

    # Fallback: raise rather than silently masking a data shape bug.
    raise ValueError(
        "normalize_to_multi: DataFrame has no 'symbol' column and a single index. "
        "Check the data source response shape."
    )

# ---------- Indicators ----------
def compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """
    Wilder's RSI using EWM smoothing (alpha=1/length).
    FIX: original used rolling().mean() which diverges from charting platform values.
    """
    delta   = series.diff()
    up      = delta.clip(lower=0)
    down    = (-delta).clip(lower=0)
    alpha   = 1.0 / length
    ma_up   = up.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    ma_down = down.ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    g = df.groupby(level=0)

    # RSI (Wilder's)
    df["rsi"] = g["close"].transform(compute_rsi)

    # MACD
    def _macd(s):
        return s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    def _macd_signal(s):
        return _macd(s).ewm(span=9, adjust=False).mean()

    df["macd"]        = g["close"].transform(_macd)
    df["macd_signal"] = g["close"].transform(_macd_signal)

    # EMAs
    df["ema9"]  = g["close"].transform(lambda x: x.ewm(span=9,  adjust=False).mean())
    df["ema21"] = g["close"].transform(lambda x: x.ewm(span=21, adjust=False).mean())

    # Volume MA
    df["vol_ma20"] = g["volume"].transform(lambda x: x.rolling(20, min_periods=20).mean())

    # Bollinger Bands
    df["bb_mid"]   = g["close"].transform(lambda x: x.rolling(20, min_periods=20).mean())
    df["bb_std"]   = g["close"].transform(lambda x: x.rolling(20, min_periods=20).std())
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]

    # SMA50 trend bias
    df["sma50"] = g["close"].transform(lambda x: x.rolling(50, min_periods=50).mean())

    # ATR (vectorized per symbol — avoids per-symbol recompute in build_targets)
    def _atr(sub):
        high, low, close = sub["high"], sub["low"], sub["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.rolling(14, min_periods=14).mean()

    df["atr14"] = g.apply(_atr).droplevel(0)

    return df

# ---------- Targets ensemble ----------
def prior_day_hlc(df: pd.DataFrame):
    """
    Returns (high, low, close) for the most recent completed trading day
    relative to the last bar in df. Returns None if insufficient history.
    Note: will return None on Mondays / post-holiday if the lookback window
    doesn't reach the prior session — callers handle None gracefully.
    """
    ts    = df.index.get_level_values("timestamp")
    dates = pd.to_datetime(ts).normalize()
    last_day  = dates[-1]
    prev_days = dates[dates < last_day].unique()
    if len(prev_days) == 0:
        return None
    prev_date = prev_days[-1]
    day_df    = df[dates == prev_date]
    if day_df.empty:
        return None
    return float(day_df["high"].max()), float(day_df["low"].min()), float(day_df["close"].iloc[-1])

def classic_pivots(h, l, c):
    P  = (h + l + c) / 3.0
    R1 = 2*P - l;  S1 = 2*P - h
    R2 = P + (h-l);S2 = P - (h-l)
    return {"P":P, "R1":R1, "R2":R2, "S1":S1, "S2":S2}

def recent_swing(prices: pd.Series, lookback: int = 80):
    tail = prices.tail(lookback)
    return float(tail.min()), float(tail.max())

def fib_levels_from_swing(low: float, high: float):
    diff = high - low
    return {
        "23.6%":  high - 0.236 * diff,
        "38.2%":  high - 0.382 * diff,
        "50%":    high - 0.500 * diff,
        "61.8%":  high - 0.618 * diff,
        "78.6%":  high - 0.786 * diff,
        "127.2%": high + 0.272 * diff,
        "161.8%": high + 0.618 * diff,
    }

def donchian_levels(df: pd.DataFrame, length: int = 20):
    return (
        float(df["high"].rolling(length, min_periods=length).max().iloc[-1]),
        float(df["low"].rolling(length, min_periods=length).min().iloc[-1])
    )

def median_ignore_nans(vals):
    vals = [v for v in vals if v is not None and pd.notna(v)]
    if not vals:
        return None
    return float(pd.Series(vals).median())

def build_targets(df_symbol: pd.DataFrame, side: str, last_close: float):
    """
    Ensemble targets using Fib + ATR + Pivots + Donchian.
    ATR is now read from the pre-computed 'atr14' column instead of recomputed here.
    """
    if df_symbol is None or df_symbol.shape[0] < 30 or pd.isna(last_close):
        return {}

    # Read pre-computed ATR from indicators (avoids redundant rolling)
    a_val = df_symbol["atr14"].iloc[-1] if "atr14" in df_symbol.columns else np.nan
    a = float(a_val) if pd.notna(a_val) else None

    d_hi, d_lo = donchian_levels(df_symbol, 20)
    pr  = prior_day_hlc(df_symbol)
    piv = classic_pivots(*pr) if pr else None
    sw_lo, sw_hi = recent_swing(df_symbol["close"], 80)
    fibs = fib_levels_from_swing(sw_lo, sw_hi) if sw_hi > sw_lo else None

    def clean(vals):
        return [v for v in vals if v is not None and not pd.isna(v)]

    if side == "BUY":
        stop = min(clean([
            fibs.get("23.6%")       if fibs else None,
            d_lo,
            piv.get("S1")           if piv  else None,
            last_close - a          if a    else None,
        ]), default=None)
        t1 = median_ignore_nans([
            fibs.get("38.2%")       if fibs else None,
            d_hi,
            piv.get("R1")           if piv  else None,
            last_close + a          if a    else None,
        ])
        t2 = median_ignore_nans([
            fibs.get("61.8%")       if fibs else None,
            piv.get("R2")           if piv  else None,
            last_close + 1.5*a      if a    else None,
            fibs.get("127.2%")      if fibs else None,
        ])
    else:  # SELL
        stop = max(clean([
            fibs.get("23.6%")       if fibs else None,
            d_hi,
            piv.get("R1")           if piv  else None,
            last_close + a          if a    else None,
        ]), default=None)
        t1 = median_ignore_nans([
            fibs.get("61.8%")       if fibs else None,
            d_lo,
            piv.get("S1")           if piv  else None,
            last_close - a          if a    else None,
        ])
        t2 = median_ignore_nans([
            fibs.get("78.6%")       if fibs else None,
            piv.get("S2")           if piv  else None,
            last_close - 1.5*a      if a    else None,
            fibs.get("161.8%")      if fibs else None,
        ])

    return {"stop": stop, "t1": t1, "t2": t2}

# ---------- Scoring ----------
def score_row(row, prev):
    buy_score = sell_score = 0
    buy_reasons, sell_reasons = [], []

    # RSI crossings
    if pd.notna(prev.get("rsi")) and pd.notna(row.get("rsi")):
        if prev["rsi"] < 30 and row["rsi"] >= 30:
            buy_score  += W["RSI"]; buy_reasons.append("RSI↑30")
        if prev["rsi"] > 70 and row["rsi"] <= 70:
            sell_score += W["RSI"]; sell_reasons.append("RSI↓70")

    # MACD cross
    if all(pd.notna(prev.get(k)) for k in ["macd","macd_signal"]) and \
       all(pd.notna(row.get(k))  for k in ["macd","macd_signal"]):
        if prev["macd"] <= prev["macd_signal"] and row["macd"] > row["macd_signal"]:
            buy_score  += W["MACD"]; buy_reasons.append("MACD×")
        if prev["macd"] >= prev["macd_signal"] and row["macd"] < row["macd_signal"]:
            sell_score += W["MACD"]; sell_reasons.append("MACD×")

    # EMA 9/21 cross
    if all(pd.notna(row.get(k)) for k in ["ema9","ema21"]) and \
       all(pd.notna(prev.get(k)) for k in ["ema9","ema21"]):
        if prev["ema9"] <= prev["ema21"] and row["ema9"] > row["ema21"]:
            buy_score  += W["EMA"]; buy_reasons.append("EMA9>21")
        if prev["ema9"] >= prev["ema21"] and row["ema9"] < row["ema21"]:
            sell_score += W["EMA"]; sell_reasons.append("EMA9<21")

    # Bollinger bounces
    if all(pd.notna(prev.get(k)) for k in ["bb_lower","bb_upper","close"]) and \
       all(pd.notna(row.get(k))  for k in ["bb_lower","bb_upper","close"]):
        if prev["close"] < prev["bb_lower"] and row["close"] > row["bb_lower"]:
            buy_score  += W["BB"]; buy_reasons.append("BB▲")
        if prev["close"] > prev["bb_upper"] and row["close"] < row["bb_upper"]:
            sell_score += W["BB"]; sell_reasons.append("BB▼")

    # Volume expansion (confirms direction — applied to both)
    if pd.notna(row.get("vol_ma20")) and pd.notna(row.get("volume")) and row["vol_ma20"] > 0:
        if row["volume"] > row["vol_ma20"]:
            buy_score  += W["VOL"]; buy_reasons.append("VOL↑")
            sell_score += W["VOL"]; sell_reasons.append("VOL↑")

    # SMA50 trend bias
    # Note: hard-zeroes score for off-trend signals. A stock 0.01% below SMA50
    # is treated the same as one 20% below. Adjust to a penalty if you prefer
    # softer filtering (e.g. buy_score = int(buy_score * 0.5) when below SMA50).
    if pd.notna(row.get("sma50")) and pd.notna(row.get("close")):
        if not (row["close"] > row["sma50"]):
            buy_score  = 0
        if not (row["close"] < row["sma50"]):
            sell_score = 0

    return (buy_score, buy_reasons), (sell_score, sell_reasons)

# ---------- Provider fetchers ----------
def get_bars_finnhub(
    symbol: str,
    lookback_bars: int = LOOKBACK,
    resolution: str = "5",
    retries: int = 3,
) -> pd.DataFrame | None:
    """
    Single-symbol OHLCV from Finnhub.
    Dispatch rate is managed externally by _finnhub_limiter.
    Retries with exponential backoff on transient errors (network/5xx/429).
    """
    if not FINNHUB_API_KEY:
        raise SystemExit("Missing FINNHUB_API_KEY")

    end   = int(datetime.now(timezone.utc).timestamp())
    start = end - lookback_bars * TIMEFRAME_MINUTES * 60

    url    = "https://finnhub.io/api/v1/stock/candle"
    params = {
        "symbol":     symbol,
        "resolution": resolution,
        "from":       start,
        "to":         end,
        "token":      FINNHUB_API_KEY,
    }

    for attempt in range(retries):
        try:
            r    = requests.get(url, params=params, timeout=15)
            data = r.json()
        except Exception as e:
            wait = 2 ** attempt
            print(f"[finnhub] {symbol} attempt {attempt+1} error: {e} — retrying in {wait}s")
            time.sleep(wait)
            continue

        # 429 or 5xx — back off and retry (consumes a rate slot; acceptable)
        if r.status_code == 429 or r.status_code >= 500:
            wait = 2 ** attempt
            print(f"[finnhub] {symbol} HTTP {r.status_code} — retrying in {wait}s")
            time.sleep(wait)
            continue

        if not isinstance(data, dict) or data.get("s") != "ok":
            return None  # "no_data" or unknown symbol — not retryable

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(data["t"], unit="s", utc=True),
            "open":      data["o"],
            "high":      data["h"],
            "low":       data["l"],
            "close":     data["c"],
            "volume":    data["v"],
        }).set_index("timestamp")

        df["symbol"] = symbol
        df = df.set_index("symbol", append=True).swaplevel(0, 1)
        df.index = df.index.set_names(["symbol", "timestamp"])
        return df

    print(f"[finnhub] {symbol} failed after {retries} attempts")
    return None

def get_bars_alpaca(chunk: list[str]) -> pd.DataFrame | None:
    """
    Multi-symbol OHLCV via Alpaca (IEX feed).
    Primary provider if DATA_PROVIDER=alpaca — ~120 symbols per request.
    """
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests  import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except Exception as e:
        print(f"[alpaca] missing alpaca-py package: {e}")
        return None
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("[alpaca] missing API keys")
        return None

    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    end    = datetime.now(timezone.utc)
    start  = end - timedelta(minutes=LOOKBACK * TIMEFRAME_MINUTES)

    req = StockBarsRequest(
        symbol_or_symbols=chunk,
        timeframe=TimeFrame(TIMEFRAME_MINUTES, TimeFrameUnit.Minute),
        start=start, end=end, limit=LOOKBACK,
        feed="iex",
    )
    try:
        df = client.get_stock_bars(req).df
        return normalize_to_multi(df)
    except Exception as e:
        print(f"[alpaca] fetch error: {e}")
        return None

# ---------- Scan ----------
def scan_once(tickers: list[str]):
    """
    Returns (buys, sells).
    Each element: (symbol, timestamp, price, score, reasons, targets_dict)
    """
    debug = os.getenv("DEBUG", "0") == "1"

    all_buys, all_sells = [], []

    N = len(tickers)
    if N == 0:
        print("[scan] no tickers supplied")
        return all_buys, all_sells

    # Rotation slice
    offset   = load_offset() % N
    end_ix   = offset + SCAN_LIMIT
    universe = (tickers[offset:end_ix] if end_ix <= N
                else tickers[offset:] + tickers[:(end_ix % N)])
    save_offset(end_ix % N)

    print(
        f"[scan] provider={DATA_PROVIDER}  slice={len(universe)}  "
        f"offset={offset}→{end_ix % N}  timeframe={TIMEFRAME_MINUTES}Min  lookback={LOOKBACK}"
    )

    fetched_ok = fetched_empty = fetch_errors = 0
    sample_logged = False

    # ---- fetch bars ----
    if DATA_PROVIDER == "finnhub":
        frames = []

        def fetch_one(sym: str):
            """Rate-limited fetch dispatched from the thread pool."""
            _finnhub_limiter.wait()
            return sym, get_bars_finnhub(sym)

        # 10 workers: HTTP responses overlap in-flight while rate limiter
        # serialises dispatches to ≤50/min.
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_one, s): s for s in universe}
            for future in as_completed(futures):
                sym, df = future.result()
                if df is None:
                    fetch_errors += 1
                elif df.empty:
                    fetched_empty += 1
                else:
                    frames.append(df)
                    fetched_ok += 1
                    if debug and not sample_logged:
                        print("[debug] sample symbol:", sym)
                        try:
                            print("[debug] first 3 rows:\n", df.head(3).to_string())
                            print("[debug] columns:", list(df.columns))
                            ts = df.index.get_level_values("timestamp")
                            print("[debug] date range:", ts.min(), "→", ts.max())
                        except Exception as e:
                            print("[debug] sample log error:", e)
                        sample_logged = True

        if not frames:
            print(f"[scan] no frames collected (ok={fetched_ok}, empty={fetched_empty}, errors={fetch_errors})")
            return all_buys, all_sells

        bars = pd.concat(frames).sort_index()

    else:
        # Alpaca: batch 120 symbols per request
        bars_list = []
        for i in range(0, len(universe), 120):
            chunk = universe[i:i+120]
            print(f"[scan] alpaca chunk {i//120+1}: {chunk[0]}..{chunk[-1]} ({len(chunk)})")
            try:
                df = get_bars_alpaca(chunk)
            except Exception as e:
                print("[scan] alpaca fetch exception:", e)
                df = None
            if df is None:
                fetch_errors += 1
            elif df.empty:
                fetched_empty += 1
            else:
                bars_list.append(df)
                fetched_ok += 1
                if debug and not sample_logged:
                    print("[debug] first 3 rows:\n", df.head(3).to_string())
                    print("[debug] columns:", list(df.columns))
                    ts = df.index.get_level_values("timestamp")
                    print("[debug] date range:", ts.min(), "→", ts.max())
                    sample_logged = True
            time.sleep(0.25)

        if not bars_list:
            print(f"[scan] no frames collected (ok={fetched_ok}, empty={fetched_empty}, errors={fetch_errors})")
            return all_buys, all_sells
        bars = pd.concat(bars_list).sort_index()

    # Normalize index
    bars = normalize_to_multi(bars)
    if bars is None or bars.empty:
        print("[scan] normalized bars empty")
        return all_buys, all_sells

    # Compute indicators (ATR now included here)
    data = compute_indicators(bars)
    if data is None or data.empty:
        print("[scan] indicators produced empty dataframe")
        return all_buys, all_sells

    if debug:
        try:
            sym0 = data.index.get_level_values(0).unique()[0]
            df0  = data.xs(sym0, level=0).tail(2)
            print(f"[debug] post-indicators sample for {sym0}")
            print(df0.to_string())
        except Exception as e:
            print("[debug] post-indicators sample error:", e)

    # Per-symbol scan
    skipped_short = 0
    for sym in data.index.get_level_values(0).unique():
        df_sym = data.xs(sym, level=0)
        if not isinstance(df_sym, pd.DataFrame) or df_sym.shape[0] < 30:
            skipped_short += 1
            continue

        last = df_sym.iloc[-1]
        prev = df_sym.iloc[-2]
        if not isinstance(last, pd.Series) or not isinstance(prev, pd.Series):
            continue

        (b_score, b_reasons), (s_score, s_reasons) = score_row(last, prev)
        last_px = float(last.get("close", np.nan))
        if pd.isna(last_px):
            continue

        if b_score >= BUY_THRESHOLD:
            tg = build_targets(df_sym, side="BUY", last_close=last_px)
            all_buys.append((sym, last.name, last_px, int(b_score), b_reasons, tg))

        if s_score >= SELL_THRESHOLD:
            tg = build_targets(df_sym, side="SELL", last_close=last_px)
            all_sells.append((sym, last.name, last_px, int(s_score), s_reasons, tg))

    print(
        f"[scan] fetch ok={fetched_ok}, empty={fetched_empty}, errors={fetch_errors} | "
        f"scanned={len(universe)}, skipped_short={skipped_short}, "
        f"buys={len(all_buys)}, sells={len(all_sells)}"
    )

    return all_buys, all_sells

# ---------- Notify ----------
def run_and_notify():
    tickers = read_tickers_from_csv(CSV_PATH)
    state   = load_state()
    buys, sells = scan_once(tickers)

    now = time.time()

    # ---- Cooldown filter ----
    # FIX: cooldowns are now persisted in state["_cooldowns"] so they survive
    # across runs. The original dict was re-initialised every run, making the
    # 30-minute cooldown completely ineffective.
    cooldowns = state.get("_cooldowns", {})

    def filter_cooldown(events, tag):
        out = []
        for sym, ts, px, score, reasons, tg in events:
            key = f"{tag}:{sym}"
            last_fired = cooldowns.get(key, 0)
            if now - last_fired >= COOLDOWN_SEC:
                cooldowns[key] = now
                out.append((sym, ts, px, score, reasons, tg))
            else:
                remaining = int((COOLDOWN_SEC - (now - last_fired)) / 60)
                print(f"[cooldown] suppressed {key} ({remaining}min remaining)")
        return out

    buys  = filter_cooldown(buys,  "BUY")
    sells = filter_cooldown(sells, "SELL")

    # Prune stale cooldown entries (older than 2× window) to keep state lean
    state["_cooldowns"] = {k: v for k, v in cooldowns.items() if now - v < COOLDOWN_SEC * 2}

    # Record BUYs
    for sym, ts, px, score, reasons, tg in buys:
        try:
            ts_iso = pd.Timestamp(ts).isoformat()
        except Exception:
            ts_iso = str(ts)
        state[sym] = {"buy_px": float(px), "buy_ts": ts_iso}

    # ---- Build signal lines ----
    buy_lines  = []
    sell_lines = []

    for sym, ts, px, score, reasons, tg in buys:
        reason_str = ", ".join(reasons) if reasons else "-"
        tgt = _format_targets(tg)
        buy_lines.append(f"- {sym} @ {ts} — ${px:.2f} (score {score}) [{reason_str}]{tgt}")

    for sym, ts, px, score, reasons, tg in sells:
        reason_str = ", ".join(reasons) if reasons else "-"
        tgt = _format_targets(tg)
        growth_note = ""
        if sym in state and "buy_px" in state[sym]:
            buy_px = state[sym]["buy_px"]
            if buy_px and buy_px > 0:
                pct      = (px / buy_px - 1.0) * 100.0
                buy_ts   = state[sym].get("buy_ts", "")
                sign     = "+" if pct >= 0 else ""
                growth_note = f" [{sign}{pct:.2f}% since BUY @ ${buy_px:.2f} on {buy_ts}]"
            state.pop(sym, None)
        sell_lines.append(f"- {sym} @ {ts} — ${px:.2f} (score {score}) [{reason_str}]{growth_note}{tgt}")

    save_state(state)

    if not buy_lines and not sell_lines:
        print("[notify] No signals this run.")
        return

    # ---- Discord dispatch ----
    # FIX: chunk on signal boundaries, never split a line across messages.
    sections = []
    if buy_lines:
        sections.append(("**BUY**", buy_lines))
    if sell_lines:
        sections.append(("**SELL**", sell_lines))

    messages = _build_discord_messages(sections)
    for msg in messages:
        discord(msg)

    print(f"[notify] Discord — {len(messages)} message(s) sent. buys={len(buy_lines)}, sells={len(sell_lines)}")

def _format_targets(tg: dict) -> str:
    if not tg:
        return ""
    parts = []
    t1 = tg.get("t1"); t2 = tg.get("t2"); s = tg.get("stop")
    if t1 is not None: parts.append(f"T1 ${t1:.2f}")
    if t2 is not None: parts.append(f"T2 ${t2:.2f}")
    if s  is not None: parts.append(f"Stop ${s:.2f}")
    return ("  " + ", ".join(parts)) if parts else ""

def _build_discord_messages(sections: list[tuple[str, list[str]]], limit: int = 1800) -> list[str]:
    """
    Build Discord messages that never split a signal line across chunks.
    Each section header stays with at least its first signal line.
    """
    messages: list[str] = []
    current   = "**Confluence Signals**\n"

    for header, lines in sections:
        block = header + "\n\n" + "\n".join(lines)
        # If adding this whole section fits, append it
        if len(current) + len(block) + 1 <= limit:
            current += "\n" + block
        else:
            # Try to fit line by line
            header_added = False
            for line in lines:
                prefix = ("\n" + header + "\n\n") if not header_added else "\n"
                if len(current) + len(prefix) + len(line) + 1 <= limit:
                    current += prefix + line
                    header_added = True
                else:
                    # Flush current and start a new message
                    messages.append(current.strip())
                    current = header + "\n\n" + line
                    header_added = True

    if current.strip():
        messages.append(current.strip())

    return messages

# ---------- Main ----------
if __name__ == "__main__":
    run_and_notify()
