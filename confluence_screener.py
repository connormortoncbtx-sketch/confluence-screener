# Confluence Screener — Discord Alerts (no pandas-ta; indicators via pandas)
# Features kept:
# - Confluence alerts (RSI/MACD/EMA/Bollinger/Volume) with 50-SMA trend bias
# - Discord alerts with reason codes
# - ETF/SPAC filter (if name/desc column exists)
# - Gentler batching + backoff (rate-limit friendly)
# - Daily digest mode: `python confluence_screener.py --digest`
# - Persists last BUY; on SELL reports % change since BUY (state/ cache)

import os, time, ssl, smtplib, csv, requests, sys, json
from email.mime.text import MIMEText
from collections import defaultdict
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# -------------------- ENV & PATHS --------------------
load_dotenv()
ALPACA_API_KEY     = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY  = os.getenv("ALPACA_SECRET_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise SystemExit("Missing ALPACA_API_KEY/ALPACA_SECRET_KEY (set in .env or GitHub Secrets)")
if not DISCORD_WEBHOOK_URL:
    print("Warning: DISCORD_WEBHOOK_URL not set — discord alerts disabled.")

CSV_PATH  = Path("nasdaqlisted.csv")   # must be in repo root
STATE_DIR = Path("state"); STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = STATE_DIR / "state.json"  # { "AAPL": {"buy_px": ..., "buy_ts": "..." } }

# -------------------- SCAN CONFIG --------------------
TIMEFRAME = TimeFrame.FiveMinutes
LOOKBACK  = 400

W = dict(RSI=25, MACD=25, EMA=25, BB=15, VOL=10)
BUY_THRESHOLD  = 60
SELL_THRESHOLD = 60
COOLDOWN_SEC   = 30 * 60  # per ticker per direction (in-process)

# -------------------- OPTIONAL EMAIL (OFF) --------------------
USE_EMAIL = False
SMTP_HOST, SMTP_PORT = "smtp.gmail.com", 465
EMAIL_FROM, EMAIL_TO = "you@example.com", ["you@example.com"]
EMAIL_USER, EMAIL_PASS = "you@example.com", "APP_PASSWORD_HERE"

def email(subject, html):
    if not USE_EMAIL: return
    msg = MIMEText(html, "html")
    msg["Subject"] = subject; msg["From"] = EMAIL_FROM; msg["To"] = ", ".join(EMAIL_TO)
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx) as s:
        s.login(EMAIL_USER, EMAIL_PASS)
        s.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())

def discord(msg: str):
    if not DISCORD_WEBHOOK_URL: return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": msg}, timeout=10)
    except Exception as e:
        print(f"[discord] post failed: {e}")

# -------------------- STATE --------------------
def load_state() -> dict:
    if STATE_PATH.exists():
        try: return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception as e: print(f"[state] read error: {e}")
    return {}

def save_state(state: dict):
    try: STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e: print(f"[state] write error: {e}")

# -------------------- TICKERS --------------------
def read_tickers_from_csv(path: Path):
    if not path.exists():
        raise SystemExit(f"Ticker file not found: {path}")
    with open(path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        delim = csv.Sniffer().sniff(sample, delimiters="|,\t;").delimiter
    df = pd.read_csv(path, delimiter=delim)

    # symbol column
    cands = [c for c in df.columns if c.lower() in ["symbol","ticker","symbols","tickers"]]
    col = cands[0] if cands else df.columns[0]

    # optional ETF/SPAC/etc filter by name/desc
    name_cols = [c for c in df.columns if c.lower() in ["security name","name","description","company name","issuer name"]]
    if name_cols:
        nm = df[name_cols[0]].astype(str).str.lower()
        mask = ~nm.str.contains(r"\b(etf|trust|fund|warrant|unit|spac)\b", regex=True)
        df = df[mask]

    ser = (df[col].astype(str).str.strip()
           .str.replace(r"[^\w\.-]", "", regex=True)
           .replace("", pd.NA).dropna().drop_duplicates())
    tk = ser.tolist()
    print(f"[tickers] loaded {len(tk)} symbols from {path.name}")
    return tk

# -------------------- INDICATORS (manual) --------------------
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=length).mean()

def rsi_wilder(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def macd_series(close: pd.Series, fast=12, slow=26, signal=9):
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd = fast_ema - slow_ema
    macds = ema(macd, signal)
    return macd, macds

def bollinger(close: pd.Series, length=20, mult=2.0):
    basis = sma(close, length)
    dev = close.rolling(length, min_periods=length).std()
    upper = basis + mult * dev
    lower = basis - mult * dev
    return lower, upper

def compute_indicators(df_multi: pd.DataFrame) -> pd.DataFrame:
    out = []
    for sym in df_multi.index.get_level_values(0).unique():
        sub = df_multi.xs(sym).copy()
        if len(sub) < 50: 
            continue
        sub["rsi"] = rsi_wilder(sub["close"], 14)
        sub["macd"], sub["macds"] = macd_series(sub["close"], 12, 26, 9)
        sub["ema9"]  = ema(sub["close"], 9)
        sub["ema21"] = ema(sub["close"], 21)
        sub["bbL"], sub["bbU"] = bollinger(sub["close"], 20, 2.0)
        sub["volSma20"] = sma(sub["volume"], 20)
        sub["sma50"] = sma(sub["close"], 50)
        sub["symbol"] = sym
        out.append(sub)
    return pd.concat(out).sort_index() if out else pd.DataFrame()

# -------------------- DATA --------------------
def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def fetch_history(client, tickers):
    req = StockBarsRequest(symbol_or_symbols=tickers, timeframe=TIMEFRAME, limit=LOOKBACK)
    return client.get_stock_bars(req).df

# -------------------- SIGNALS --------------------
def score_row(row, prev):
    # BUY
    rsiReclaim30  = pd.notna(prev["rsi"]) and pd.notna(row["rsi"]) and prev["rsi"] < 30 and row["rsi"] >= 30
    macdBullCross = pd.notna(prev["macd"]) and pd.notna(prev["macds"]) and pd.notna(row["macd"]) and pd.notna(row["macds"]) and prev["macd"] <= prev["macds"] and row["macd"] > row["macds"]
    emaBullCross  = pd.notna(prev["ema9"]) and pd.notna(prev["ema21"]) and pd.notna(row["ema9"]) and pd.notna(row["ema21"]) and prev["ema9"] <= prev["ema21"] and row["ema9"] > row["ema21"]
    bbBullBounce  = pd.notna(prev["bbL"]) and pd.notna(row["bbL"]) and prev["close"] < prev["bbL"] and row["close"] > row["bbL"]
    volExp        = pd.notna(row["volSma20"]) and row["volume"] > row["volSma20"]

    buy_score = (W["RSI"] if rsiReclaim30 else 0) + (W["MACD"] if macdBullCross else 0) + \
                (W["EMA"] if emaBullCross else 0) + (W["BB"] if bbBullBounce else 0) + (W["VOL"] if volExp else 0)
    buy_reasons = []
    if rsiReclaim30:  buy_reasons.append("RSI↑30")
    if macdBullCross: buy_reasons.append("MACD×")
    if emaBullCross:  buy_reasons.append("EMA9>21")
    if bbBullBounce:  buy_reasons.append("BB▲")
    if volExp:        buy_reasons.append("VOL↑")

    # SELL
    rsiFall70     = pd.notna(prev["rsi"]) and pd.notna(row["rsi"]) and prev["rsi"] > 70 and row["rsi"] <= 70
    macdBearCross = pd.notna(prev["macd"]) and pd.notna(prev["macds"]) and pd.notna(row["macd"]) and pd.notna(row["macds"]) and prev["macd"] >= prev["macds"] and row["macd"] < row["macds"]
    emaBearCross  = pd.notna(prev["ema9"]) and pd.notna(prev["ema21"]) and pd.notna(row["ema9"]) and pd.notna(row["ema21"]) and prev["ema9"] >= prev["ema21"] and row["ema9"] < row["ema21"]
    bbBearReject  = pd.notna(prev["bbU"]) and pd.notna(row["bbU"]) and prev["close"] > prev["bbU"] and row["close"] < row["bbU"]

    sell_score = (W["RSI"] if rsiFall70 else 0) + (W["MACD"] if macdBearCross else 0) + \
                 (W["EMA"] if emaBearCross else 0) + (W["BB"] if bbBearReject else 0) + (W["VOL"] if volExp else 0)
    sell_reasons = []
    if rsiFall70:     sell_reasons.append("RSI↓70")
    if macdBearCross: sell_reasons.append("MACD×")
    if emaBearCross:  sell_reasons.append("EMA9<21")
    if bbBearReject:  sell_reasons.append("BB▼")
    if volExp:        sell_reasons.append("VOL↑")

    # Trend bias
    trend_up   = pd.notna(row.get("sma50")) and pd.notna(row["close"]) and row["close"] > row["sma50"]
    trend_down = pd.notna(row.get("sma50")) and pd.notna(row["close"]) and row["close"] < row["sma50"]
    if not trend_up:   buy_score  = 0
    if not trend_down: sell_score = 0

    return (buy_score, buy_reasons), (sell_score, sell_reasons)

# -------------------- SCAN --------------------
def scan_once(tickers):
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    all_buys, all_sells = [], []
    print(f"[scan] scanning {len(tickers)} tickers, timeframe={TIMEFRAME}, lookback={LOOKBACK}")

    for chunk in chunked(tickers, 120):
        try:
            print(f"[scan] chunk size={len(chunk)}  first={chunk[0]}  last={chunk[-1]}")
            raw = fetch_history(client, chunk)
        except Exception as e:
            print(f"[scan] fetch error: {e}"); time.sleep(1.0); continue

        if raw is None or raw.empty:
            time.sleep(0.25); continue

        data = compute_indicators(raw)
        if data is None or data.empty:
            time.sleep(0.25); continue

        for sym in data.index.get_level_values(0).unique():
            df = data.xs(sym)
            if len(df) < 3: continue
            last, prev = df.iloc[-1], df.iloc[-2]
            (b_score, b_reasons), (s_score, s_reasons) = score_row(last, prev)
            if b_score >= BUY_THRESHOLD:
                all_buys.append((sym, last.name, float(last["close"]), int(b_score), b_reasons))
            if s_score >= SELL_THRESHOLD:
                all_sells.append((sym, last.name, float(last["close"]), int(s_score), s_reasons))

        time.sleep(0.25)  # backoff

    return all_buys, all_sells

_last_fire = defaultdict(lambda: 0)  # in-process cooldown

# -------------------- NOTIFY (with BUY→SELL P&L) --------------------
def run_and_notify():
    state = load_state()
    tickers = read_tickers_from_csv(CSV_PATH)
    buys, sells = scan_once(tickers)

    now = time.time()
    def filter_cooldown(events, tag):
        out = []
        for sym, ts, px, score, reasons in events:
            key = f"{tag}:{sym}"
            if now - _last_fire[key] >= COOLDOWN_SEC:
                _last_fire[key] = now
                out.append((sym, ts, px, score, reasons))
        return out

    buys  = filter_cooldown(buys,  "BUY")
    sells = filter_cooldown(sells, "SELL")

    # record BUYs
    for sym, ts, px, score, reasons in buys:
        state[sym] = {"buy_px": float(px), "buy_ts": pd.Timestamp(ts).isoformat()}

    lines, has_lines = ["**Confluence Signals**"], False
    if buys:
        has_lines = True; lines.append("**BUY**")
        for sym, ts, px, score, reasons in buys:
            reason_str = ", ".join(reasons) if reasons else "-"
            lines.append(f"- {sym} @ {ts} — ${px:.2f} (score {score}) [{reason_str}]")

    if sells:
        has_lines = True; lines.append("**SELL**")
        for sym, ts, px, score, reasons in sells:
            reason_str = ", ".join(reasons) if reasons else "-"
            growth_note = ""
            if sym in state and "buy_px" in state[sym]:
                buy_px = state[sym]["buy_px"]
                if buy_px and buy_px > 0:
                    pct = (px / buy_px - 1.0) * 100.0
                    buy_ts = state[sym].get("buy_ts", "")
                    growth_note = f" [+{pct:.2f}% since BUY @ ${buy_px:.2f} on {buy_ts}]"
                    del state[sym]  # reset after paired SELL
            lines.append(f"- {sym} @ {ts} — ${px:.2f} (score {score}) [{reason_str}]{growth_note}")

    save_state(state)

    if has_lines:
        msg = "\n".join(lines)
        if len(msg) > 1800:
            discord("**Confluence Signals**")
            chunk = ""
            for line in lines[1:]:
                if len(chunk) + len(line) + 1 > 1800:
                    discord(chunk); chunk = line
                else:
                    chunk = line if not chunk else f"{chunk}\n{line}"
            if chunk: discord(chunk)
        else:
            discord(msg)
    else:
        print("[notify] No signals this run.")

# -------------------- DIGEST --------------------
def digest():
    tickers = read_tickers_from_csv(CSV_PATH)
    buys, sells = scan_once(tickers)
    buys = sorted(buys, key=lambda x: x[3], reverse=True)[:10]
    sells = sorted(sells, key=lambda x: x[3], reverse=True)[:10]
    lines = ["**Close Digest — Top Confluence**"]
    if buys:
        lines.append("**BUY**")
        for s, ts, px, sc, rs in buys:
            reason_str = ", ".join(rs) if rs else "-"
            lines.append(f"- {s} — ${px:.2f} (score {sc}) [{reason_str}]")
    if sells:
        lines.append("**SELL**")
        for s, ts, px, sc, rs in sells:
            reason_str = ", ".join(rs) if rs else "-"
            lines.append(f"- {s} — ${px:.2f} (score {sc}) [{reason_str}]")
    msg = "\n".join(lines) if len(lines) > 1 else "No high scores today."
    discord(msg)

# -------------------- MAIN --------------------
if __name__ == "__main__":
    if "--digest" in sys.argv:
        digest()
    else:
        run_and_notify()
