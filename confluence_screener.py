# Confluence Screener — Discord Alerts
# How to run:
#   1) pip install alpaca-py pandas pandas_ta python-dotenv requests
#   2) Create a .env file with ALPACA_API_KEY=... ALPACA_SECRET_KEY=... DISCORD_WEBHOOK_URL=...
#   3) python confluence_screener.py

import os, time, ssl, smtplib, csv, requests
from email.mime.text import MIMEText
from collections import defaultdict
from pathlib import Path
import pandas as pd
import pandas_ta as ta
from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

load_dotenv()

ALPACA_API_KEY    = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")
if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise SystemExit("Missing ALPACA_API_KEY/ALPACA_SECRET_KEY in environment (.env)")
if not DISCORD_WEBHOOK_URL:
    print("Warning: DISCORD_WEBHOOK_URL not set — alerts will not be sent to Discord.")

# --- Paths ---
CSV_PATH = Path("/mnt/data/nasdaqlisted.csv")

# --- Timeframe & Lookback ---
TIMEFRAME = TimeFrame.FiveMinutes  # Minute, FiveMinutes, FifteenMinutes, Day
LOOKBACK  = 400

# --- Confluence Weights & thresholds ---
W = dict(RSI=25, MACD=25, EMA=25, BB=15, VOL=10)
BUY_THRESHOLD  = 60
SELL_THRESHOLD = 60
COOLDOWN_SEC   = 30 * 60  # 30 minutes

# --- Optional email (still supported, disabled by default) ---
USE_EMAIL = False
SMTP_HOST, SMTP_PORT = "smtp.gmail.com", 465
EMAIL_FROM    = "you@example.com"
EMAIL_TO      = ["you@example.com"]
EMAIL_USER    = "you@example.com"
EMAIL_PASS    = "APP_PASSWORD_HERE"

def email(subject, html):
    if not USE_EMAIL: 
        return
    msg = MIMEText(html, "html")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(EMAIL_TO)
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ctx) as server:
        server.login(EMAIL_USER, EMAIL_PASS)
        server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())

def discord(msg: str):
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": msg}, timeout=5)
    except Exception:
        pass

def read_tickers_from_csv(path: Path):
    # Sniff delimiter and locate a symbol column
    with open(path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        dialect = csv.Sniffer().sniff(sample, delimiters="|,\t;")
        delim = dialect.delimiter
    df = pd.read_csv(path, delimiter=delim)
    cands = [c for c in df.columns if c.lower() in ["symbol","ticker","symbols","tickers"]]
    col = cands[0] if cands else df.columns[0]
    ser = (df[col].astype(str).str.strip()
           .str.replace(r"[^\w\.-]", "", regex=True)
           .replace("", pd.NA).dropna().drop_duplicates())
    return ser.tolist()

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def fetch_history(client, tickers):
    req = StockBarsRequest(symbol_or_symbols=tickers, timeframe=TIMEFRAME, limit=LOOKBACK)
    return client.get_stock_bars(req).df

def compute_indicators(df):
    out = []
    for sym in df.index.get_level_values(0).unique():
        sub = df.xs(sym).copy()
        if len(sub) < 50:
            continue
        sub["rsi"] = ta.rsi(sub["close"], length=14)
        macd = ta.macd(sub["close"], 12, 26, 9)
        sub["macd"], sub["macds"] = macd["MACD_12_26_9"], macd["MACDs_12_26_9"]
        sub["ema9"], sub["ema21"] = ta.ema(sub["close"], 9), ta.ema(sub["close"], 21)
        bb = ta.bbands(sub["close"], length=20, std=2.0)
        sub["bbL"], sub["bbU"] = bb["BBL_20_2.0"], bb["BBU_20_2.0"]
        sub["volSma20"] = sub["volume"].rolling(20).mean()
        sub["symbol"] = sym
        out.append(sub)
    return pd.concat(out).sort_index() if out else pd.DataFrame()

def score_row(row, prev):
    rsiReclaim30  = (pd.notna(prev["rsi"]) and pd.notna(row["rsi"]) and prev["rsi"] < 30 and row["rsi"] >= 30)
    macdBullCross = (pd.notna(prev["macd"]) and pd.notna(prev["macds"]) and pd.notna(row["macd"]) and pd.notna(row["macds"]) and prev["macd"] <= prev["macds"] and row["macd"] > row["macds"])
    emaBullCross  = (pd.notna(prev["ema9"]) and pd.notna(prev["ema21"]) and pd.notna(row["ema9"]) and pd.notna(row["ema21"]) and prev["ema9"] <= prev["ema21"] and row["ema9"] > row["ema21"])
    bbBullBounce  = (pd.notna(prev["bbL"]) and pd.notna(row["bbL"]) and prev["close"] < prev["bbL"] and row["close"] > row["bbL"])
    volExp        = (pd.notna(row["volSma20"]) and row["volume"] > row["volSma20"])

    buy_score = (W["RSI"] if rsiReclaim30 else 0) + (W["MACD"] if macdBullCross else 0) + (W["EMA"] if emaBullCross else 0) + (W["BB"] if bbBullBounce else 0) + (W["VOL"] if volExp else 0)

    rsiFall70     = (pd.notna(prev["rsi"]) and pd.notna(row["rsi"]) and prev["rsi"] > 70 and row["rsi"] <= 70)
    macdBearCross = (pd.notna(prev["macd"]) and pd.notna(prev["macds"]) and pd.notna(row["macd"]) and pd.notna(row["macds"]) and prev["macd"] >= prev["macds"] and row["macd"] < row["macds"])
    emaBearCross  = (pd.notna(prev["ema9"]) and pd.notna(prev["ema21"]) and pd.notna(row["ema9"]) and pd.notna(row["ema21"]) and prev["ema9"] >= prev["ema21"] and row["ema9"] < row["ema21"])
    bbBearReject  = (pd.notna(prev["bbU"]) and pd.notna(row["bbU"]) and prev["close"] > prev["bbU"] and row["close"] < row["bbU"])

    sell_score = (W["RSI"] if rsiFall70 else 0) + (W["MACD"] if macdBearCross else 0) + (W["EMA"] if emaBearCross else 0) + (W["BB"] if bbBearReject else 0) + (W["VOL"] if volExp else 0)
    return buy_score, sell_score

def scan_once(tickers):
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    all_buys, all_sells = [], []
    for chunk in chunked(tickers, 200):
        raw = fetch_history(client, chunk)
        if raw is None or raw.empty:
            continue
        data = compute_indicators(raw)
        if data is None or data.empty:
            continue
        for sym in data.index.get_level_values(0).unique():
            df = data.xs(sym)
            if len(df) < 3: 
                continue
            last, prev = df.iloc[-1], df.iloc[-2]
            b, s = score_row(last, prev)
            if b >= BUY_THRESHOLD:
                all_buys.append((sym, last.name, float(last["close"]), int(b)))
            if s >= SELL_THRESHOLD:
                all_sells.append((sym, last.name, float(last["close"]), int(s)))
    return all_buys, all_sells

_last_fire = defaultdict(lambda: 0)

def run_and_notify():
    tickers = read_tickers_from_csv(CSV_PATH)
    if not tickers:
        raise SystemExit("No tickers parsed from CSV")
    buys, sells = scan_once(tickers)
    now = time.time()

    def filter_cooldown(events, tag):
        out = []
        for sym, ts, px, score in events:
            key = f"{tag}:{sym}"
            if now - _last_fire[key] >= COOLDOWN_SEC:
                _last_fire[key] = now
                out.append((sym, ts, px, score))
        return out

    buys = filter_cooldown(buys, "BUY")
    sells = filter_cooldown(sells, "SELL")

    # Build Discord-friendly message(s)
    if buys or sells:
        MAX_LEN = 1800  # keep under message limit (~2000 incl header)
        lines = ["**Confluence Signals**"]
        if buys:
            lines.append("**BUY**")
            for s, ts, px, sc in buys:
                lines.append(f"- {s} @ {ts} — ${px:.2f} (score {sc})")
        if sells:
            lines.append("**SELL**")
            for s, ts, px, sc in sells:
                lines.append(f"- {s} @ {ts} — ${px:.2f} (score {sc})")
        msg = "\n".join(lines)

        # If too long, chunk it
        if len(msg) > MAX_LEN:
            header = "**Confluence Signals**"
            discord(header)
            chunk = ""
            for line in lines[1:]:
                if len(chunk) + len(line) + 1 > MAX_LEN:
                    discord(chunk)
                    chunk = line
                else:
                    chunk = line if not chunk else f"{chunk}\n{line}"
            if chunk:
                discord(chunk)
        else:
            discord(msg)
    else:
        print("No signals this run.")

if __name__ == "__main__":
    run_and_notify()