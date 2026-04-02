# confluence_screener.py
# ---------------------------------------------
# Screener with Alpaca batched bars (preferred) or Finnhub (free, slower).
# Features:
# - RSI/MACD/EMA/Bollinger/Volume + daily trend gate
# - Confluence scoring -> BUY/SELL alerts (2-bar confirmation required)
# - Discord notifications with reason codes
# - Adaptive learning: trade ledger, per-indicator win rate, weight adaptation
# - Position timeout: force-close open buys after POSITION_TIMEOUT_DAYS
# - BUY->SELL P&L, hold time, annualized return on every closed trade
# - Targets ensemble: Fib + ATR + Pivots + Donchian (T1/T2/Stop)
# - Robust ticker CSV loader (filters ETFs/SPACs by name when present)
# - Provider switch: DATA_PROVIDER = "alpaca" (default) or "finnhub"
# - Rate-limit pacing via thread-safe RateLimiter (both providers)
# - Parallel fetches via ThreadPoolExecutor
# - Cooldown tracking persisted in state.json (survives across runs)
# - Retry with exponential backoff on Finnhub transient errors
# - Discord chunking on signal boundaries (never splits mid-line)
# - Price + liquidity filters (MIN_PRICE, MIN_VOL_MA)
# - Directional volume scoring (up-bar adds to buy, down-bar adds to sell)
# - Daily trend gate derived from resampled intraday data (no extra API call)
# - Finnhub news sentiment enrichment for signaling symbols only

import os, time, json, csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
import threading

import pandas as pd
import numpy as np
import requests

# Load .env for local runs — no-op in GitHub Actions where secrets are
# injected directly into the environment.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --------- ENV / Paths ----------
DATA_PROVIDER       = os.getenv("DATA_PROVIDER", "alpaca").lower()
FINNHUB_API_KEY     = os.getenv("FINNHUB_API_KEY")
ALPACA_API_KEY      = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY   = os.getenv("ALPACA_SECRET_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# Startup key check — helps diagnose auth failures locally
print(f"[keys] ALPACA_API_KEY={'SET' if ALPACA_API_KEY else 'MISSING'} "
      f"ALPACA_SECRET_KEY={'SET' if ALPACA_SECRET_KEY else 'MISSING'} "
      f"FINNHUB_API_KEY={'SET' if FINNHUB_API_KEY else 'MISSING'}")

# CRYPTO_MODE=1 switches the entire pipeline to crypto:
#   - reads cryptolisted.csv instead of nasdaqlisted.csv
#   - uses CryptoHistoricalDataClient + CryptoBarsRequest
#   - uses longer timeframe and higher thresholds (crypto is more volatile)
#   - namespaces all state keys with "_cx_" so equity/crypto ledgers never mix
#   - skips MIN_PRICE and MIN_VOL_MA filters (irrelevant for crypto)
CRYPTO_MODE = os.getenv("CRYPTO_MODE", "0") == "1"

CSV_PATH    = Path("cryptolisted.csv") if CRYPTO_MODE else Path("nasdaqlisted.csv")
STATE_DIR   = Path("state"); STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH  = STATE_DIR / "state.json"
ROTATE_PATH = STATE_DIR / ("crypto_offset.json" if CRYPTO_MODE else "scan_offset.json")

# State key prefix — keeps equity and crypto positions/trades/weights separate
# in the same state.json file without collision.
SK = "_cx_" if CRYPTO_MODE else "_"

# --------- Scan Config ----------
# Crypto uses 15-min bars (5-min is too noisy for volatile assets) and
# higher score thresholds to reduce false positives.
TIMEFRAME_MINUTES = int(os.getenv("TIMEFRAME_MINUTES", "15" if CRYPTO_MODE else "5"))
LOOKBACK          = 400
BUY_THRESHOLD     = int(os.getenv("BUY_THRESHOLD",  "70" if CRYPTO_MODE else "60"))
SELL_THRESHOLD    = int(os.getenv("SELL_THRESHOLD", "70" if CRYPTO_MODE else "60"))

FINNHUB_CALLS_PER_MIN = 50
ALPACA_CALLS_PER_MIN  = int(os.getenv("ALPACA_CALLS_PER_MIN", "180"))
SCAN_LIMIT            = int(os.getenv("SCAN_LIMIT", "9999"))
COOLDOWN_SEC          = 30 * 60

# Equity filters — not applied in crypto mode (different volume units, SHIB < $0.01)
MIN_PRICE  = float(os.getenv("MIN_PRICE",  "5.0"))
MIN_VOL_MA = float(os.getenv("MIN_VOL_MA", "5000"))

# --------- Adaptive Learning Config ----------
# ADAPT_EVERY:           closed trades required before weight adjustment
# POSITION_TIMEOUT_DAYS: force-close open buys older than this
# WEIGHT_MIN/MAX:        per-indicator weight clamps
# WEIGHT_STEP:           max points moved per indicator per adaptation cycle
# MAX_TRADES_STORED:     cap ledger size to avoid unbounded state growth
ADAPT_EVERY           = int(os.getenv("ADAPT_EVERY",           "20"))
POSITION_TIMEOUT_DAYS = int(os.getenv("POSITION_TIMEOUT_DAYS", "5"))
WEIGHT_MIN        = 5
WEIGHT_MAX        = 45
WEIGHT_STEP       = 3
MAX_TRADES_STORED = 500

# Default weights — overridden by state[SK+"weights"] after adaptation.
# MACD raised (strongest independent intraday signal),
# RSI lowered (most collinear with MACD/EMA on 5-min bars). Total = 100.
W_DEFAULT = dict(RSI=20, MACD=30, EMA=25, BB=15, VOL=10)
W = dict(W_DEFAULT)  # mutable; updated from state at runtime

# --------- Thread-safe Rate Limiter ----------
class RateLimiter:
    def __init__(self, max_per_min: int):
        self._delay = 60.0 / max_per_min
        self._lock  = threading.Lock()
        self._last  = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            gap = self._last + self._delay - now
            if gap > 0:
                time.sleep(gap)
            self._last = time.monotonic()

_finnhub_limiter = RateLimiter(FINNHUB_CALLS_PER_MIN)
_alpaca_limiter  = RateLimiter(ALPACA_CALLS_PER_MIN)

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
        df = df[~nm.str.contains(r"\b(?:etf|trust|fund|warrant|unit|spac)\b", regex=True, na=False)]
    ser = (df[col].astype(str).str.strip()
           .str.replace(r"[^\w\.-]", "", regex=True)
           .replace("", pd.NA).dropna().drop_duplicates())
    tickers = ser.tolist()
    print(f"[tickers] loaded {len(tickers)} symbols from {path.name}")
    return tickers

# ---------- Index normalize ----------
def normalize_to_multi(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if getattr(df.index, "nlevels", 1) == 2:
        lvl0 = df.index.get_level_values(0)
        lvl1 = df.index.get_level_values(1)
        if pd.api.types.is_datetime64_any_dtype(lvl0) and not pd.api.types.is_datetime64_any_dtype(lvl1):
            df = df.swaplevel(0, 1)
        df.index = df.index.set_names(["symbol", "timestamp"])
        return df.sort_index()
    if not df.index.name:
        df.index.name = "timestamp"
    if "symbol" in df.columns:
        return df.reset_index().set_index(["symbol","timestamp"]).sort_index()
    raise ValueError("normalize_to_multi: no symbol column and single index — check data source.")

# ---------- Indicators ----------
def compute_rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Wilder's RSI (EWM alpha=1/length) — matches charting platform values."""
    delta   = series.diff()
    alpha   = 1.0 / length
    ma_up   = delta.clip(lower=0).ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    ma_down = (-delta).clip(lower=0).ewm(alpha=alpha, adjust=False, min_periods=length).mean()
    rs = ma_up / ma_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    g = df.groupby(level=0)

    df["rsi"] = g["close"].transform(compute_rsi)

    def _macd(s):        return s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
    def _macd_sig(s):    return _macd(s).ewm(span=9, adjust=False).mean()

    df["macd"]        = g["close"].transform(_macd)
    df["macd_signal"] = g["close"].transform(_macd_sig)
    df["ema9"]        = g["close"].transform(lambda x: x.ewm(span=9,  adjust=False).mean())
    df["ema21"]       = g["close"].transform(lambda x: x.ewm(span=21, adjust=False).mean())
    df["vol_ma20"]    = g["volume"].transform(lambda x: x.rolling(20, min_periods=20).mean())
    df["bb_mid"]      = g["close"].transform(lambda x: x.rolling(20, min_periods=20).mean())
    df["bb_std"]      = g["close"].transform(lambda x: x.rolling(20, min_periods=20).std())
    df["bb_upper"]    = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"]    = df["bb_mid"] - 2 * df["bb_std"]
    df["sma50"]       = g["close"].transform(lambda x: x.rolling(50, min_periods=50).mean())

    def _atr(sub):
        pc = sub["close"].shift(1)
        tr = pd.concat([sub["high"]-sub["low"],
                        (sub["high"]-pc).abs(), (sub["low"]-pc).abs()], axis=1).max(axis=1)
        return tr.rolling(14, min_periods=14).mean()

    df["atr14"] = g.apply(_atr).droplevel(0)
    return df

# ---------- Daily trend gate ----------
def get_daily_trend(df_sym: pd.DataFrame) -> str:
    """
    Derive trend from resampled intraday data — no extra API call.
    Compares last daily close vs prior daily close.
    Returns 'up', 'down', or 'neutral'.
    """
    try:
        daily = df_sym["close"].resample("1D").last().dropna()
        if len(daily) < 2:
            return "neutral"
        if daily.iloc[-1] > daily.iloc[-2]:
            return "up"
        if daily.iloc[-1] < daily.iloc[-2]:
            return "down"
    except Exception:
        pass
    return "neutral"

# ---------- Targets ensemble ----------
def prior_day_hlc(df: pd.DataFrame):
    idx   = df.index.get_level_values("timestamp") if df.index.nlevels > 1 else df.index
    dates = pd.to_datetime(idx).normalize()
    prev_days = dates[dates < dates[-1]].unique()
    if len(prev_days) == 0:
        return None
    day_df = df[dates == prev_days[-1]]
    return (float(day_df["high"].max()), float(day_df["low"].min()),
            float(day_df["close"].iloc[-1])) if not day_df.empty else None

def classic_pivots(h, l, c):
    P = (h+l+c)/3.0
    return {"P":P,"R1":2*P-l,"R2":P+(h-l),"S1":2*P-h,"S2":P-(h-l)}

def recent_swing(prices: pd.Series, lookback: int = 80):
    t = prices.tail(lookback)
    return float(t.min()), float(t.max())

def fib_levels_from_swing(low, high):
    d = high-low
    return {"23.6%":high-0.236*d,"38.2%":high-0.382*d,"50%":high-0.5*d,
            "61.8%":high-0.618*d,"78.6%":high-0.786*d,
            "127.2%":high+0.272*d,"161.8%":high+0.618*d}

def donchian_levels(df, length=20):
    return (float(df["high"].rolling(length, min_periods=length).max().iloc[-1]),
            float(df["low"].rolling(length,  min_periods=length).min().iloc[-1]))

def median_ignore_nans(vals):
    vals = [v for v in vals if v is not None and pd.notna(v)]
    return float(pd.Series(vals).median()) if vals else None

def build_targets(df_sym, side, last_close):
    if df_sym is None or df_sym.shape[0] < 30 or pd.isna(last_close):
        return {}
    a_raw = df_sym["atr14"].iloc[-1] if "atr14" in df_sym.columns else np.nan
    a     = float(a_raw) if pd.notna(a_raw) else None
    d_hi, d_lo = donchian_levels(df_sym)
    pr    = prior_day_hlc(df_sym)
    piv   = classic_pivots(*pr) if pr else None
    sw_lo, sw_hi = recent_swing(df_sym["close"])
    fibs  = fib_levels_from_swing(sw_lo, sw_hi) if sw_hi > sw_lo else None

    def clean(vals): return [v for v in vals if v is not None and not pd.isna(v)]

    if side == "BUY":
        stop = min(clean([fibs.get("23.6%") if fibs else None, d_lo,
                          piv.get("S1") if piv else None, last_close-a if a else None]), default=None)
        t1   = median_ignore_nans([fibs.get("38.2%") if fibs else None, d_hi,
                                   piv.get("R1") if piv else None, last_close+a if a else None])
        t2   = median_ignore_nans([fibs.get("61.8%") if fibs else None,
                                   piv.get("R2") if piv else None, last_close+1.5*a if a else None,
                                   fibs.get("127.2%") if fibs else None])
    else:
        stop = max(clean([fibs.get("23.6%") if fibs else None, d_hi,
                          piv.get("R1") if piv else None, last_close+a if a else None]), default=None)
        t1   = median_ignore_nans([fibs.get("61.8%") if fibs else None, d_lo,
                                   piv.get("S1") if piv else None, last_close-a if a else None])
        t2   = median_ignore_nans([fibs.get("78.6%") if fibs else None,
                                   piv.get("S2") if piv else None, last_close-1.5*a if a else None,
                                   fibs.get("161.8%") if fibs else None])
    return {"stop": stop, "t1": t1, "t2": t2}

# ---------- Scoring ----------
def score_row(row, prev, prev2, daily_trend: str = "neutral"):
    """
    2-bar confirmed confluence score.
    Every signal requires: prev2 was wrong side, prev crossed, row still holds.
    Volume is directional: up-bar -> buy only, down-bar -> sell only.
    Daily trend gates: 'up' suppresses sells, 'down' suppresses buys.
    W is adaptive (updated from state[SK+"weights"] at runtime).
    """
    buy_score = sell_score = 0
    buy_reasons, sell_reasons = [], []

    if all(pd.notna(x.get("rsi")) for x in [row, prev, prev2]):
        if prev2["rsi"] < 30 and prev["rsi"] >= 30 and row["rsi"] >= 30:
            buy_score  += W["RSI"]; buy_reasons.append("RSI↑30")
        if prev2["rsi"] > 70 and prev["rsi"] <= 70 and row["rsi"] <= 70:
            sell_score += W["RSI"]; sell_reasons.append("RSI↓70")

    if all(pd.notna(x.get(k)) for x in [row,prev,prev2] for k in ["macd","macd_signal"]):
        if (prev2["macd"] <= prev2["macd_signal"]
                and prev["macd"] > prev["macd_signal"]
                and row["macd"]  > row["macd_signal"]):
            buy_score  += W["MACD"]; buy_reasons.append("MACD×↑")
        if (prev2["macd"] >= prev2["macd_signal"]
                and prev["macd"] < prev["macd_signal"]
                and row["macd"]  < row["macd_signal"]):
            sell_score += W["MACD"]; sell_reasons.append("MACD×↓")

    if all(pd.notna(x.get(k)) for x in [row,prev,prev2] for k in ["ema9","ema21"]):
        if (prev2["ema9"] <= prev2["ema21"]
                and prev["ema9"] > prev["ema21"]
                and row["ema9"]  > row["ema21"]):
            buy_score  += W["EMA"]; buy_reasons.append("EMA9>21")
        if (prev2["ema9"] >= prev2["ema21"]
                and prev["ema9"] < prev["ema21"]
                and row["ema9"]  < row["ema21"]):
            sell_score += W["EMA"]; sell_reasons.append("EMA9<21")

    if all(pd.notna(x.get(k)) for x in [row,prev,prev2] for k in ["bb_lower","bb_upper","close"]):
        if (prev2["close"] < prev2["bb_lower"]
                and prev["close"] > prev["bb_lower"]
                and row["close"]  > row["bb_lower"]):
            buy_score  += W["BB"]; buy_reasons.append("BB▲")
        if (prev2["close"] > prev2["bb_upper"]
                and prev["close"] < prev["bb_upper"]
                and row["close"]  < row["bb_upper"]):
            sell_score += W["BB"]; sell_reasons.append("BB▼")

    if pd.notna(row.get("vol_ma20")) and pd.notna(row.get("volume")) and row["vol_ma20"] > 0:
        if row["volume"] > row["vol_ma20"] and pd.notna(row.get("open")):
            if row["close"] > row["open"]: buy_score  += W["VOL"]; buy_reasons.append("VOL↑")
            if row["close"] < row["open"]: sell_score += W["VOL"]; sell_reasons.append("VOL↑")

    if daily_trend == "down":  buy_score  = 0
    elif daily_trend == "up":  sell_score = 0

    return (buy_score, buy_reasons), (sell_score, sell_reasons)

# ==================== Adaptive Learning ====================

def close_trade(state: dict, sym: str, exit_px: float, exit_ts, exit_reason: str = "SELL") -> dict | None:
    """
    Record a completed BUY->SELL (or TIMEOUT) trade in state[SK+"trades"].

    Metrics recorded per trade:
      pnl_pct      — raw return %
      hold_minutes — time in position
      annualized   — (pnl_pct / hold_days) * 365
                     Makes a 5% gain in 1 day rank higher than 5% in 30 days.
      win          — True if pnl_pct > 0

    buy_reasons is stored at buy time so we can attribute performance back
    to whichever indicators fired — this is what drives weight adaptation.
    """
    if sym not in state or "buy_px" not in state[sym]:
        return None

    rec       = state[sym]
    buy_px    = float(rec["buy_px"])
    buy_ts_dt = pd.Timestamp(rec["buy_ts"])
    exit_ts_dt = pd.Timestamp(exit_ts)

    hold_min  = max((exit_ts_dt - buy_ts_dt.tz_convert("UTC")).total_seconds() / 60, 1.0)
    pnl_pct   = (exit_px / buy_px - 1.0) * 100.0
    annualized = (pnl_pct / (hold_min / 1440)) * 365

    trade = {
        "sym":         sym,
        "buy_ts":      rec.get("buy_ts"),
        "buy_px":      buy_px,
        "buy_score":   rec.get("buy_score", 0),
        "buy_reasons": rec.get("buy_reasons", []),
        "exit_ts":     str(exit_ts),
        "exit_px":     round(exit_px, 4),
        "exit_reason": exit_reason,
        "pnl_pct":     round(pnl_pct, 4),
        "hold_min":    round(hold_min, 1),
        "annualized":  round(annualized, 2),
        "win":         pnl_pct > 0,
    }

    if SK+"trades" not in state:
        state[SK+"trades"] = []
    state[SK+"trades"].append(trade)
    if len(state[SK+"trades"]) > MAX_TRADES_STORED:
        state[SK+"trades"] = state[SK+"trades"][-MAX_TRADES_STORED:]

    state.pop(sym, None)
    return trade


def adapt_weights(trades: list, current_weights: dict) -> tuple[dict, list[str]]:
    """
    Nudge indicator weights toward what's demonstrably working.

    For each indicator:
      1. Isolate trades where it fired on the buy.
      2. Compare mean annualized return to the overall mean.
      3. Outperformers get +WEIGHT_STEP; underperformers get -WEIGHT_STEP.
      4. Clamp to [WEIGHT_MIN, WEIGHT_MAX].
      5. Rescale all weights to preserve the original total.

    Minimum 5 trades per indicator required before moving its weight.
    Returns (new_weights, change_log).
    """
    if not trades:
        return current_weights, []

    overall_avg  = float(np.mean([t["annualized"] for t in trades]))
    total_weight = sum(current_weights.values())
    new_weights  = dict(current_weights)
    change_log   = []

    for ind in current_weights:
        ind_trades = [t for t in trades if any(ind in r for r in (t.get("buy_reasons") or []))]
        n = len(ind_trades)
        if n < 5:
            change_log.append(f"{ind}: only {n} trades — unchanged")
            continue

        ind_avg  = float(np.mean([t["annualized"] for t in ind_trades]))
        ind_wins = sum(1 for t in ind_trades if t["win"]) / n
        old      = new_weights[ind]
        delta    = WEIGHT_STEP if ind_avg > overall_avg else -WEIGHT_STEP
        new_weights[ind] = max(WEIGHT_MIN, min(WEIGHT_MAX, old + delta))

        sign = "+" if delta > 0 else ""
        change_log.append(
            f"{ind}: {old}→{new_weights[ind]} ({sign}{delta}) | "
            f"ann={ind_avg:+.0f}% vs base {overall_avg:+.0f}%, "
            f"win={ind_wins:.0%} n={n}"
        )

    # Rescale to preserve total
    current_total = sum(new_weights.values())
    if current_total != total_weight and current_total > 0:
        scale = total_weight / current_total
        new_weights = {k: max(WEIGHT_MIN, min(WEIGHT_MAX, round(v * scale)))
                       for k, v in new_weights.items()}

    return new_weights, change_log


def build_performance_report(trades: list, weights: dict) -> str:
    """
    Full performance summary for Discord.
    Includes: win rate, avg P&L, avg annualized, avg hold, best/worst,
    per-indicator breakdown, and current weights.
    """
    if not trades:
        return "**📊 Performance Report**\nNo closed trades yet."

    n        = len(trades)
    win_rate = sum(1 for t in trades if t["win"]) / n
    avg_pnl  = float(np.mean([t["pnl_pct"]    for t in trades]))
    avg_ann  = float(np.mean([t["annualized"]  for t in trades]))
    avg_hold = float(np.mean([t["hold_min"]    for t in trades]))
    best     = max(trades, key=lambda t: t["pnl_pct"])
    worst    = min(trades, key=lambda t: t["pnl_pct"])

    lines = [
        "**📊 Performance Report**",
        f"Closed trades: {n}  |  Win rate: {win_rate:.0%}  |  Avg P&L: {avg_pnl:+.2f}%",
        f"Avg annualized: {avg_ann:+.0f}%  |  Avg hold: {_fmt_dur(avg_hold)}",
        f"Best:  {best['sym']} {best['pnl_pct']:+.2f}% in {_fmt_dur(best['hold_min'])}",
        f"Worst: {worst['sym']} {worst['pnl_pct']:+.2f}% in {_fmt_dur(worst['hold_min'])}",
        "",
        "**Indicator breakdown:**",
    ]
    for ind in weights:
        it = [t for t in trades if any(ind in r for r in (t.get("buy_reasons") or []))]
        if not it:
            lines.append(f"  {ind} (wt={weights[ind]}): no data")
            continue
        lines.append(
            f"  {ind} (wt={weights[ind]}): "
            f"{sum(1 for t in it if t['win'])/len(it):.0%} win, "
            f"{float(np.mean([t['annualized'] for t in it])):+.0f}% ann, "
            f"n={len(it)}"
        )
    lines += ["", "**Weights:** " + "  ".join(f"{k}={v}" for k, v in weights.items())]
    return "\n".join(lines)


def _fmt_dur(minutes: float) -> str:
    if minutes < 60:    return f"{minutes:.0f}m"
    if minutes < 1440:  return f"{minutes/60:.1f}h"
    return f"{minutes/1440:.1f}d"

# ==================== Sentiment ====================

def get_sentiment_finnhub(symbol: str) -> dict | None:
    if not FINNHUB_API_KEY:
        return None
    try:
        r = requests.get("https://finnhub.io/api/v1/news-sentiment",
                         params={"symbol": symbol, "token": FINNHUB_API_KEY}, timeout=10)
        d = r.json()
        if not isinstance(d, dict) or "sentiment" not in d:
            return None
        return {
            "bull_pct":   float(d["sentiment"].get("bullishPercent", 0)),
            "bear_pct":   float(d["sentiment"].get("bearishPercent", 0)),
            "buzz":       float(d.get("buzz", {}).get("buzz", 1.0)),
            "news_score": float(d.get("companyNewsScore", 0.5)),
        }
    except Exception as e:
        print(f"[sentiment] {symbol}: {e}"); return None

def fetch_sentiments(symbols: list[str]) -> dict[str, dict]:
    if not symbols or not FINNHUB_API_KEY:
        return {}
    def _f(sym):
        _finnhub_limiter.wait()
        return sym, get_sentiment_finnhub(sym)
    results = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        for fut in as_completed({ex.submit(_f, s): s for s in symbols}):
            sym, sent = fut.result()
            if sent: results[sym] = sent
    print(f"[sentiment] {len(results)}/{len(symbols)} symbols")
    return results

# ==================== Providers ====================

def get_bars_finnhub(symbol: str, lookback_bars: int = LOOKBACK,
                     resolution: str = "5", retries: int = 3) -> pd.DataFrame | None:
    if not FINNHUB_API_KEY:
        raise SystemExit("Missing FINNHUB_API_KEY")
    end   = int(datetime.now(timezone.utc).timestamp())
    start = end - lookback_bars * TIMEFRAME_MINUTES * 60
    params = {"symbol": symbol, "resolution": resolution,
               "from": start, "to": end, "token": FINNHUB_API_KEY}
    for attempt in range(retries):
        try:
            r = requests.get("https://finnhub.io/api/v1/stock/candle", params=params, timeout=15)
            data = r.json()
        except Exception as e:
            w = 2**attempt; print(f"[finnhub] {symbol} err: {e} retry {w}s"); time.sleep(w); continue
        if r.status_code == 429 or r.status_code >= 500:
            w = 2**attempt; print(f"[finnhub] {symbol} {r.status_code} retry {w}s"); time.sleep(w); continue
        if not isinstance(data, dict) or data.get("s") != "ok":
            return None
        df = pd.DataFrame({"timestamp": pd.to_datetime(data["t"], unit="s", utc=True),
                            "open": data["o"], "high": data["h"], "low": data["l"],
                            "close": data["c"], "volume": data["v"]}).set_index("timestamp")
        df["symbol"] = symbol
        df = df.set_index("symbol", append=True).swaplevel(0,1)
        df.index = df.index.set_names(["symbol","timestamp"])
        return df
    return None

def get_bars_alpaca_crypto(chunk: list[str]) -> pd.DataFrame | None:
    """
    Multi-symbol OHLCV via Alpaca crypto feed.
    Symbols must be in 'BTC/USD' format (as stored in cryptolisted.csv).
    Uses CryptoHistoricalDataClient — separate from the equity client.
    Feed is 'us' (Alpaca's consolidated crypto feed, free tier).
    """
    try:
        from alpaca.data.historical import CryptoHistoricalDataClient
        from alpaca.data.requests  import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except Exception as e:
        print(f"[alpaca-crypto] missing alpaca-py: {e}"); return None

    # Crypto client does not require API keys for market data on free tier,
    # but passing them is fine and may unlock higher rate limits.
    client = CryptoHistoricalDataClient(
        api_key=ALPACA_API_KEY or None,
        secret_key=ALPACA_SECRET_KEY or None,
    )
    end   = datetime.now(timezone.utc)
    start = end - timedelta(minutes=LOOKBACK * TIMEFRAME_MINUTES)

    req = CryptoBarsRequest(
        symbol_or_symbols=chunk,
        timeframe=TimeFrame(TIMEFRAME_MINUTES, TimeFrameUnit.Minute),
        start=start, end=end, limit=LOOKBACK,
    )
    try:
        df = client.get_crypto_bars(req).df
        return normalize_to_multi(df)
    except Exception as e:
        print(f"[alpaca-crypto] fetch error: {e}"); return None

def get_bars_alpaca(chunk: list[str]) -> pd.DataFrame | None:
    """Multi-symbol OHLCV via Alpaca IEX — ~120 symbols per request."""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests  import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except Exception as e:
        print(f"[alpaca] {e}"); return None
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("[alpaca] missing keys"); return None
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    end    = datetime.now(timezone.utc)
    start  = end - timedelta(minutes=LOOKBACK * TIMEFRAME_MINUTES)
    req = StockBarsRequest(symbol_or_symbols=chunk,
                           timeframe=TimeFrame(TIMEFRAME_MINUTES, TimeFrameUnit.Minute),
                           start=start, end=end, limit=LOOKBACK, feed="iex")
    try:
        return normalize_to_multi(client.get_stock_bars(req).df)
    except Exception as e:
        print(f"[alpaca] {e}"); return None

# ==================== Scan ====================

def scan_once(tickers: list[str]) -> tuple[list, list, dict]:
    """
    Returns (buys, sells, last_prices).
    last_prices: {symbol: float} — last bar close for every scanned symbol.
    Used by run_and_notify to close timed-out positions at a real price.
    """
    debug = os.getenv("DEBUG", "0") == "1"
    all_buys, all_sells = [], []
    last_prices: dict[str, float] = {}

    N = len(tickers)
    if N == 0:
        return all_buys, all_sells, last_prices

    offset   = load_offset() % N
    end_ix   = offset + SCAN_LIMIT
    universe = tickers[offset:end_ix] if end_ix <= N else tickers[offset:] + tickers[:(end_ix % N)]
    save_offset(end_ix % N)

    print(f"[scan] provider={DATA_PROVIDER} mode={'crypto' if CRYPTO_MODE else 'equity'} "
          f"slice={len(universe)} offset={offset}->{end_ix%N} "
          f"TF={TIMEFRAME_MINUTES}Min W={W}")

    fetched_ok = fetched_empty = fetch_errors = 0
    sample_logged = False

    if DATA_PROVIDER == "finnhub":
        frames = []
        def fetch_one(sym):
            _finnhub_limiter.wait()
            return sym, get_bars_finnhub(sym)
        with ThreadPoolExecutor(max_workers=10) as ex:
            for fut in as_completed({ex.submit(fetch_one, s): s for s in universe}):
                sym, df = fut.result()
                if df is None:    fetch_errors  += 1
                elif df.empty:    fetched_empty += 1
                else:
                    frames.append(df); fetched_ok += 1
                    if debug and not sample_logged:
                        print(f"[debug] sample {sym}"); sample_logged = True
        if not frames:
            print(f"[scan] no frames ok={fetched_ok} empty={fetched_empty} err={fetch_errors}")
            return all_buys, all_sells, last_prices
        bars = pd.concat(frames).sort_index()
    else:
        # Choose fetcher based on mode
        fetcher = get_bars_alpaca_crypto if CRYPTO_MODE else get_bars_alpaca
        # Crypto: all 20 symbols fit in one chunk; equity: 120 per chunk
        chunk_size = len(universe) if CRYPTO_MODE else 120
        chunks = [universe[i:i+chunk_size] for i in range(0, len(universe), chunk_size)]
        mode_label = "crypto" if CRYPTO_MODE else "equity"
        print(f"[scan] alpaca-{mode_label} {len(chunks)} chunk(s) parallel")
        bars_list = []
        def fetch_chunk(idx_chunk):
            idx, chunk = idx_chunk
            _alpaca_limiter.wait()
            try:
                return fetcher(chunk)
            except Exception as e:
                print(f"[scan] chunk {idx+1} err: {e}"); return None
        with ThreadPoolExecutor(max_workers=20) as ex:
            for fut in as_completed({ex.submit(fetch_chunk, (i,c)): i for i,c in enumerate(chunks)}):
                df = fut.result()
                if df is None:    fetch_errors  += 1
                elif df.empty:    fetched_empty += 1
                else:
                    bars_list.append(df); fetched_ok += 1
                    if debug and not sample_logged:
                        print("[debug] first chunk received"); sample_logged = True
        if not bars_list:
            print(f"[scan] no frames ok={fetched_ok} empty={fetched_empty} err={fetch_errors}")
            return all_buys, all_sells, last_prices
        bars = pd.concat(bars_list).sort_index()

    bars = normalize_to_multi(bars)
    if bars is None or bars.empty:
        return all_buys, all_sells, last_prices

    data = compute_indicators(bars)
    if data is None or data.empty:
        return all_buys, all_sells, last_prices

    skipped_short = skipped_filter = 0
    for sym in data.index.get_level_values(0).unique():
        df_sym = data.xs(sym, level=0)
        if not isinstance(df_sym, pd.DataFrame) or df_sym.shape[0] < 30:
            skipped_short += 1; continue

        last  = df_sym.iloc[-1]
        prev  = df_sym.iloc[-2]
        prev2 = df_sym.iloc[-3]
        if not isinstance(last, pd.Series): continue

        last_px = float(last.get("close", np.nan))
        if pd.isna(last_px): continue

        last_prices[sym] = last_px  # record for timeout closes regardless of filters

        # Equity-only filters — not applied in crypto mode.
        # Crypto has different volume units and SHIB trades below $0.01.
        if not CRYPTO_MODE:
            if last_px < MIN_PRICE:
                skipped_filter += 1; continue
            vol_ma = last.get("vol_ma20", np.nan)
            if pd.notna(vol_ma) and float(vol_ma) < MIN_VOL_MA:
                skipped_filter += 1; continue

        daily_trend = get_daily_trend(df_sym)
        (b_score, b_reasons), (s_score, s_reasons) = score_row(last, prev, prev2, daily_trend)

        if b_score >= BUY_THRESHOLD:
            all_buys.append((sym, last.name, last_px, int(b_score), b_reasons,
                             build_targets(df_sym, "BUY", last_px)))
        if s_score >= SELL_THRESHOLD:
            all_sells.append((sym, last.name, last_px, int(s_score), s_reasons,
                              build_targets(df_sym, "SELL", last_px)))

    print(f"[scan] ok={fetched_ok} empty={fetched_empty} err={fetch_errors} | "
          f"scanned={len(universe)} short={skipped_short} filtered={skipped_filter} "
          f"buys={len(all_buys)} sells={len(all_sells)}")
    return all_buys, all_sells, last_prices

# ==================== Notify + Learn ====================

def run_and_notify():
    tickers = read_tickers_from_csv(CSV_PATH)
    state   = load_state()

    # Load adaptive weights (fall back to W_DEFAULT if not yet adapted)
    saved_w = state.get(SK+"weights")
    if saved_w:
        W.update(saved_w)
        print(f"[weights] from state: {W}")
    else:
        print(f"[weights] defaults: {W}")

    buys, sells, last_prices = scan_once(tickers)
    now_ts = datetime.now(timezone.utc)
    now    = time.time()

    # ---- Position timeouts ----
    # Force-close open buys older than POSITION_TIMEOUT_DAYS.
    # Uses last_prices from the current scan so the exit price is real.
    timeout_secs = POSITION_TIMEOUT_DAYS * 86400
    newly_closed = []
    for sym, rec in list(state.items()):
        if sym.startswith("_") or "buy_ts" not in rec:
            continue
        try:
            age = (now_ts - pd.Timestamp(rec["buy_ts"]).tz_convert("UTC")).total_seconds()
        except Exception:
            continue
        if age > timeout_secs:
            exit_px = last_prices.get(sym)
            if exit_px is None:
                continue  # not in this scan — retry next run
            trade = close_trade(state, sym, exit_px, now_ts, "TIMEOUT")
            if trade:
                newly_closed.append(trade)
                sign = "+" if trade["pnl_pct"] >= 0 else ""
                print(f"[timeout] {sym} {sign}{trade['pnl_pct']:.2f}% "
                      f"in {_fmt_dur(trade['hold_min'])}")

    # ---- Cooldown filter ----
    cooldowns = state.get(SK+"cooldowns", {})
    def filter_cooldown(events, tag):
        out = []
        for item in events:
            sym = item[0]; key = f"{tag}:{sym}"
            if now - cooldowns.get(key, 0) >= COOLDOWN_SEC:
                cooldowns[key] = now; out.append(item)
            else:
                rem = int((COOLDOWN_SEC - (now - cooldowns[key])) / 60)
                print(f"[cooldown] {key} suppressed ({rem}min)")
        return out
    buys  = filter_cooldown(buys,  "BUY")
    sells = filter_cooldown(sells, "SELL")
    state[SK+"cooldowns"] = {k: v for k, v in cooldowns.items() if now - v < COOLDOWN_SEC * 2}

    # ---- Close SELL signals ----
    for sym, ts, px, score, reasons, tg in sells:
        trade = close_trade(state, sym, px, ts, "SELL")
        if trade:
            newly_closed.append(trade)
            sign = "+" if trade["pnl_pct"] >= 0 else ""
            print(f"[trade] {sym} {sign}{trade['pnl_pct']:.2f}% "
                  f"in {_fmt_dur(trade['hold_min'])} ann={trade['annualized']:+.0f}%")

    # ---- Record new BUYs ----
    # Store buy_reasons and buy_score so adapt_weights can attribute outcomes.
    # Guard: if a position is already open for this symbol, don't overwrite it.
    # A duplicate BUY would silently replace the original entry price, causing
    # the original trade to vanish from the ledger entirely.
    for sym, ts, px, score, reasons, tg in buys:
        if sym in state and "buy_px" in state[sym]:
            print(f"[buy] {sym} already open at ${state[sym]['buy_px']:.2f} "
                  f"(since {state[sym].get('buy_ts','?')}) — skipping duplicate at ${px:.2f}")
            continue
        try:   ts_iso = pd.Timestamp(ts).isoformat()
        except: ts_iso = str(ts)
        state[sym] = {"buy_px": float(px), "buy_ts": ts_iso,
                      "buy_score": score, "buy_reasons": reasons}

    # ---- Weight adaptation ----
    all_trades       = state.get(SK+"trades", [])
    prev_count       = len(all_trades) - len(newly_closed)
    adapted          = False
    adapt_report     = []

    if (newly_closed
            and len(all_trades) >= ADAPT_EVERY
            and len(all_trades) // ADAPT_EVERY > max(prev_count, 0) // ADAPT_EVERY):
        new_w, adapt_report = adapt_weights(all_trades, dict(W))
        if new_w != dict(W):
            print(f"[adapt] {adapt_report}")
            W.update(new_w)
            state[SK+"weights"] = dict(W)
            adapted = True

    save_state(state)

    # ---- Sentiment enrichment ----
    signal_syms = list({s for s,*_ in buys} | {s for s,*_ in sells})
    sentiments  = fetch_sentiments(signal_syms)

    # ---- Build Discord lines ----
    buy_lines, sell_lines = [], []

    for sym, ts, px, score, reasons, tg in buys:
        buy_lines.append(
            f"- {sym} @ {ts} — ${px:.2f} (score {score}) "
            f"[{', '.join(reasons) or '-'}]"
            f"{_format_sentiment(sentiments.get(sym), 'BUY')}"
            f"{_format_targets(tg)}"
        )

    for sym, ts, px, score, reasons, tg in sells:
        closed = next((t for t in newly_closed if t["sym"]==sym and t["exit_reason"]=="SELL"), None)
        pnl_note = ""
        if closed:
            sign = "+" if closed["pnl_pct"] >= 0 else ""
            pnl_note = (f" [{sign}{closed['pnl_pct']:.2f}% "
                        f"in {_fmt_dur(closed['hold_min'])}, "
                        f"ann {closed['annualized']:+.0f}%]")
        sell_lines.append(
            f"- {sym} @ {ts} — ${px:.2f} (score {score}) "
            f"[{', '.join(reasons) or '-'}]{pnl_note}"
            f"{_format_sentiment(sentiments.get(sym), 'SELL')}"
            f"{_format_targets(tg)}"
        )

    if buy_lines or sell_lines:
        sections = []
        if buy_lines:  sections.append(("**BUY**",  buy_lines))
        if sell_lines: sections.append(("**SELL**", sell_lines))
        for msg in _build_discord_messages(sections):
            discord(msg)
        print(f"[notify] sent buys={len(buy_lines)} sells={len(sell_lines)}")
    else:
        print("[notify] No signals this run.")

    # Adaptation report to Discord when weights change
    if adapted and adapt_report:
        lines  = ["**⚙️ Weight Adaptation**"] + adapt_report
        lines += ["", "**New weights:** " + "  ".join(f"{k}={v}" for k,v in W.items())]
        discord("\n".join(lines))
        discord(build_performance_report(all_trades, dict(W)))

# ==================== Formatting helpers ====================

def _format_targets(tg: dict) -> str:
    if not tg: return ""
    parts = []
    if tg.get("t1") is not None:   parts.append(f"T1 ${tg['t1']:.2f}")
    if tg.get("t2") is not None:   parts.append(f"T2 ${tg['t2']:.2f}")
    if tg.get("stop") is not None: parts.append(f"Stop ${tg['stop']:.2f}")
    return ("  " + ", ".join(parts)) if parts else ""

def _format_sentiment(sent: dict | None, side: str) -> str:
    if not sent: return ""
    bull = sent["bull_pct"]; bear = sent["bear_pct"]; buzz = sent["buzz"]
    direction = f"📈{round(bull*100)}% bull" if bull >= bear else f"📉{round(bear*100)}% bear"
    buzz_str  = f", BUZZ {buzz:.1f}x" if buzz > 1.2 else ""
    warning   = (" ⚠ bearish sent" if side=="BUY"  and bear > 0.65 else
                 " ⚠ bullish sent" if side=="SELL" and bull > 0.65 else "")
    return f" | sent: {direction}{buzz_str}{warning}"

def _build_discord_messages(sections: list[tuple[str, list[str]]], limit: int = 1800) -> list[str]:
    """Chunk Discord messages on line boundaries — never splits a signal."""
    messages: list[str] = []
    header_label = "**🪙 Crypto Signals**\n" if CRYPTO_MODE else "**Confluence Signals**\n"
    current = header_label
    for header, lines in sections:
        block = (header + "\n\n" if header else "") + "\n".join(lines)
        if len(current) + len(block) + 1 <= limit:
            current += "\n" + block
        else:
            added = False
            for line in lines:
                prefix = ("\n" + header + "\n\n") if (header and not added) else "\n"
                if len(current) + len(prefix) + len(line) + 1 <= limit:
                    current += prefix + line; added = True
                else:
                    messages.append(current.strip())
                    current = (header + "\n\n" + line) if header else line
                    added = True
    if current.strip():
        messages.append(current.strip())
    return messages

# ==================== Main ====================

if __name__ == "__main__":
    # Set REPORT=1 in workflow_dispatch inputs for an on-demand performance report
    # without running the full scan.
    if os.getenv("REPORT") == "1":
        state   = load_state()
        trades  = state.get(SK+"trades", [])
        weights = state.get(SK+"weights", W_DEFAULT)
        report  = build_performance_report(trades, weights)
        discord(report)
        print(report)
    else:
        run_and_notify()
