import os, time, json, math, sys
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# ----------------- Config -----------------
ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

TIMEFRAME = TimeFrame(5, TimeFrameUnit.Minute)
LOOKBACK = 400
CHUNK = 120
BUY_THRESHOLD = 60
SELL_THRESHOLD = 60

# ----------------- Alpaca -----------------
client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# ----------------- Helpers -----------------
def sget(row, col, default=np.nan):
    try:
        return row[col]
    except Exception:
        return default

def normalize_bars_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Alpaca bars DataFrame to a MultiIndex ('symbol','timestamp') with symbol first.
    Handles MultiIndex in any order, single DatetimeIndex + symbol col, or single index.
    """
    if df is None or df.empty:
        return df
    if getattr(df.index, "nlevels", 1) == 2:
        lvl0 = df.index.get_level_values(0)
        lvl1 = df.index.get_level_values(1)
        lvl0_is_dt = pd.api.types.is_datetime64_any_dtype(lvl0)
        lvl1_is_dt = pd.api.types.is_datetime64_any_dtype(lvl1)
        if lvl0_is_dt and not lvl1_is_dt:
            df = df.swaplevel(0,1)
        df.index = df.index.set_names(["symbol","timestamp"])
        return df.sort_index()
    if not df.index.name:
        df.index.name = "timestamp"
    if "symbol" in df.columns:
        df2 = df.reset_index().set_index(["symbol","timestamp"]).sort_index()
        return df2
    df2 = df.copy()
    df2["symbol"] = "UNK"
    df2 = df2.reset_index().set_index(["symbol","timestamp"]).sort_index()
    return df2

# ----------------- Indicators -----------------
def compute_rsi(series, length=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ma_up = up.rolling(length, min_periods=length).mean()
    ma_down = down.rolling(length, min_periods=length).mean()
    rs = ma_up/ma_down
    rsi = 100 - (100/(1+rs))
    return rsi

def compute_macd(series, fast=12, slow=26, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def compute_indicators(df):
    if df is None or df.empty: return df
    df["rsi"] = df.groupby(level=0)["close"].transform(compute_rsi)
    macd, sig = df.groupby(level=0)["close"].transform(compute_macd)
    df["macd"] = macd
    df["macd_signal"] = sig
    df["ema9"] = df.groupby(level=0)["close"].transform(lambda x: x.ewm(span=9, adjust=False).mean())
    df["ema21"] = df.groupby(level=0)["close"].transform(lambda x: x.ewm(span=21, adjust=False).mean())
    df["vol_ma20"] = df.groupby(level=0)["volume"].transform(lambda x: x.rolling(20,min_periods=20).mean())
    df["bb_mid"] = df.groupby(level=0)["close"].transform(lambda x: x.rolling(20,min_periods=20).mean())
    df["bb_std"] = df.groupby(level=0)["close"].transform(lambda x: x.rolling(20,min_periods=20).std())
    df["bb_upper"] = df["bb_mid"] + 2*df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2*df["bb_std"]
    return df

# ----------------- Target Builders -----------------
def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return true_range(df).rolling(length, min_periods=length).mean()

def prior_day_hlc(df: pd.DataFrame):
    ts = df.index.get_level_values("timestamp")
    dates = pd.to_datetime(ts).date
    last_day = pd.Series(dates).iloc[-1]
    prev_mask = dates < last_day
    if not prev_mask.any(): return None
    prev = df[prev_mask]
    prev_date = pd.to_datetime(prev.index.get_level_values("timestamp")).date[-1]
    day_df = prev[pd.to_datetime(prev.index.get_level_values("timestamp")).date == prev_date]
    if day_df.empty: return None
    return float(day_df["high"].max()), float(day_df["low"].min()), float(day_df["close"].iloc[-1])

def classic_pivots(h,l,c):
    P=(h+l+c)/3.0
    R1=2*P-l; S1=2*P-h; R2=P+(h-l); S2=P-(h-l)
    return {"P":P,"R1":R1,"R2":R2,"S1":S1,"S2":S2}

def recent_swing(prices: pd.Series, lookback: int = 80):
    tail=prices.tail(lookback)
    return float(tail.min()), float(tail.max())

def fib_levels_from_swing(low: float, high: float):
    diff = high-low
    return {
        "23.6%": high-0.236*diff,
        "38.2%": high-0.382*diff,
        "50%": high-0.5*diff,
        "61.8%": high-0.618*diff,
        "78.6%": high-0.786*diff,
        "127.2%": high+0.272*diff,
        "161.8%": high+0.618*diff,
    }

def donchian_levels(df: pd.DataFrame, length: int=20):
    return float(df["high"].rolling(length,min_periods=length).max().iloc[-1]), float(df["low"].rolling(length,min_periods=length).min().iloc[-1])

def median_ignore_nans(vals):
    vals=[v for v in vals if v is not None and pd.notna(v)]
    if not vals: return None
    return float(pd.Series(vals).median())

def build_targets(df_symbol: pd.DataFrame, side: str, last_close: float):
    if df_symbol is None or df_symbol.shape[0] < 30 or pd.isna(last_close): return {}
    a = float(atr(df_symbol,14).iloc[-1]) if df_symbol.shape[0]>=14 else None
    d_hi,d_lo = donchian_levels(df_symbol,20)
    pr = prior_day_hlc(df_symbol)
    piv = classic_pivots(*pr) if pr else None
    sw_lo, sw_hi = recent_swing(df_symbol["close"],80)
    fibs = fib_levels_from_swing(sw_lo,sw_hi) if sw_hi>sw_lo else None
    if side=="BUY":
        stop=min([fibs.get("23.6%") if fibs else None,d_lo,piv.get("S1") if piv else None,last_close-(1.0*a) if a else None],default=None)
        t1=median_ignore_nans([fibs.get("38.2%") if fibs else None,d_hi,piv.get("R1") if piv else None,last_close+(1.0*a) if a else None])
        t2=median_ignore_nans([fibs.get("61.8%") if fibs else None,piv.get("R2") if piv else None,last_close+(1.5*a) if a else None,fibs.get("127.2%") if fibs else None])
    else:
        stop=max([fibs.get("23.6%") if fibs else None,d_hi,piv.get("R1") if piv else None,last_close+(1.0*a) if a else None],default=None)
        t1=median_ignore_nans([fibs.get("61.8%") if fibs else None,d_lo,piv.get("S1") if piv else None,last_close-(1.0*a) if a else None])
        t2=median_ignore_nans([fibs.get("78.6%") if fibs else None,piv.get("S2") if piv else None,last_close-(1.5*a) if a else None,fibs.get("161.8%") if fibs else None])
    return {"stop":stop,"t1":t1,"t2":t2}

# ----------------- Scoring -----------------
def score_row(row, prev):
    score=0; reasons=[]
    if pd.notna(prev["rsi"]) and pd.notna(row["rsi"]):
        if prev["rsi"]<30 and row["rsi"]>=30: score+=25; reasons.append("RSI↑30")
        if prev["rsi"]>70 and row["rsi"]<=70: score+=25; reasons.append("RSI↓70")
    if pd.notna(row["macd"]) and pd.notna(row["macd_signal"]):
        if row["macd"]>row["macd_signal"]: score+=20; reasons.append("MACD×")
        else: score+=20; reasons.append("MACD×")
    if pd.notna(row["ema9"]) and pd.notna(row["ema21"]):
        if row["ema9"]>row["ema21"]: score+=15; reasons.append("EMA9>21")
        else: score+=15; reasons.append("EMA9<21")
    if pd.notna(prev["vol_ma20"]) and pd.notna(row["volume"]):
        if row["volume"]>1.5*row["vol_ma20"]: score+=10; reasons.append("VOL↑")
    if pd.notna(row["close"]) and pd.notna(row["bb_upper"]) and pd.notna(row["bb_lower"]):
        if row["close"]>row["bb_upper"]: score+=10; reasons.append("BB▲")
        if row["close"]<row["bb_lower"]: score+=10; reasons.append("BB▼")
    buy_score=score if "RSI↑30" in reasons or "EMA9>21" in reasons or "MACD×" in reasons else 0
    sell_score=score if "RSI↓70" in reasons or "EMA9<21" in reasons or "MACD×" in reasons else 0
    return (buy_score,reasons),(sell_score,reasons)

# ----------------- Scan -----------------
def scan_once(tickers):
    """
    Scans the provided tickers using Alpaca IEX feed (free tier friendly),
    computes indicators, scores signals, and builds ensemble targets.
    Returns (all_buys, all_sells).
    """
    from datetime import datetime, timedelta
    from zoneinfo import ZoneInfo
    from alpaca.data.requests import StockBarsRequest

    all_buys, all_sells = [], []
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

    # 5-min bars: LOOKBACK bars ~ LOOKBACK*5 minutes of history
    end   = datetime.now(ZoneInfo("America/New_York"))
    start = end - timedelta(minutes=LOOKBACK * 5)

    print(f"[scan] scanning {len(tickers)} tickers, timeframe={TIMEFRAME}, lookback={LOOKBACK}")

    for i in range(0, len(tickers), 120):  # gentle batching
        chunk = tickers[i:i+120]
        print(f"[scan] chunk size={len(chunk)}  first={chunk[0]}  last={chunk[-1]}")

        # ---- Fetch bars from IEX feed (avoids SIP subscription error) ----
        req = StockBarsRequest(
            symbol_or_symbols=chunk,
            timeframe=TIMEFRAME,
            start=start,
            end=end,
            limit=LOOKBACK,
            feed="iex",  # <<<<<< key change
        )
        try:
            bars = client.get_stock_bars(req).df
        except Exception as e:
            print(f"[scan] error fetching bars: {e}")
            time.sleep(0.5)
            continue

        if bars is None or bars.empty:
            time.sleep(0.1)
            continue

        # Normalize to MultiIndex ('symbol','timestamp')
        bars = normalize_bars_index(bars)
        if bars is None or bars.empty:
            time.sleep(0.1)
            continue

        # Compute indicators per symbol
        data = compute_indicators(bars)
        if data is None or data.empty:
            time.sleep(0.1)
            continue

        # Iterate by symbol (level 0 after normalize)
        for sym in data.index.get_level_values(0).unique():
            df = data.xs(sym, level=0)
            if not isinstance(df, pd.DataFrame) or df.shape[0] < 3:
                continue

            last = df.iloc[-1]
            prev = df.iloc[-2]
            if not isinstance(last, pd.Series) or not isinstance(prev, pd.Series):
                continue

            (b_score, b_reasons), (s_score, s_reasons) = score_row(last, prev)
            last_px = float(sget(last, "close"))

            if b_score >= BUY_THRESHOLD:
                tg = build_targets(df, side="BUY", last_close=last_px)
                all_buys.append((sym, last.name, last_px, int(b_score), b_reasons, tg))

            if s_score >= SELL_THRESHOLD:
                tg = build_targets(df, side="SELL", last_close=last_px)
                all_sells.append((sym, last.name, last_px, int(s_score), s_reasons, tg))

        time.sleep(0.25)  # backoff to be nice to the API

    return all_buys, all_sells


# ----------------- State -----------------
STATE_FILE="state/state.json"
def load_state():
    if os.path.exists(STATE_FILE):
        try: return json.load(open(STATE_FILE,"r"))
        except: return {}
    return {}
def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE),exist_ok=True)
    json.dump(state,open(STATE_FILE,"w"))

# ----------------- Notify -----------------
def notify_discord(msg: str):
    if not DISCORD_WEBHOOK_URL: return
    try: requests.post(DISCORD_WEBHOOK_URL,json={"content":msg})
    except Exception as e: print("[notify] error",e)

# ----------------- Main -----------------
def run_and_notify():
    tickers=pd.read_csv("nasdaqlisted.csv")["Symbol"].tolist()
    tickers=[t for t in tickers if pd.notna(t)]
    print(f"[tickers] loaded {len(tickers)} symbols from nasdaqlisted.csv")
    buys,sells=scan_once(tickers)
    state=load_state(); lines=["**Confluence Signals**"]
    if buys:
        lines.append("**BUY**"); lines.append("")
        for sym,ts,px,score,reasons,tg in buys:
            reason_str=", ".join(reasons) if reasons else "-"
            tgt=""
            if tg:
                s=tg.get("stop"); t1=tg.get("t1"); t2=tg.get("t2")
                parts=[]
                if t1: parts.append(f"T1 ${t1:.2f}")
                if t2: parts.append(f"T2 ${t2:.2f}")
                if s: parts.append(f"Stop ${s:.2f}")
                if parts: tgt="  "+", ".join(parts)
            lines.append(f"- {sym} @ {ts} — ${px:.2f} (score {score}) [{reason_str}]{tgt}")
            state[sym]={"buy_px":px,"buy_ts":str(ts)}
    if sells:
        lines.append("**SELL**"); lines.append("")
        for sym,ts,px,score,reasons,tg in sells:
            reason_str=", ".join(reasons) if reasons else "-"
            tgt=""
            if tg:
                s=tg.get("stop"); t1=tg.get("t1"); t2=tg.get("t2")
                parts=[]
                if t1: parts.append(f"T1 ${t1:.2f}")
                if t2: parts.append(f"T2 ${t2:.2f}")
                if s: parts.append(f"Stop ${s:.2f}")
                if parts: tgt="  "+", ".join(parts)
            growth_note=""
            if sym in state and "buy_px" in state[sym]:
                buy_px=state[sym]["buy_px"]
                if buy_px and buy_px>0:
                    pct=(px/buy_px-1.0)*100.0
                    buy_ts=state[sym].get("buy_ts","")
                    sign="+" if pct>=0 else ""
                    growth_note=f" [{sign}{pct:.2f}% since BUY @ ${buy_px:.2f} on {buy_ts}]"
                state.pop(sym,None)
            lines.append(f"- {sym} @ {ts} — ${px:.2f} (score {score}) [{reason_str}]{growth_note}{tgt}")
    if len(lines)>1: notify_discord("\n".join(lines))
    else: print("[notify] No signals this run.")
    save_state(state)

if __name__=="__main__":
    run_and_notify()
