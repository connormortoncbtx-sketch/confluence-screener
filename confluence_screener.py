# Confluence Screener — Discord Alerts with niceties
# Requires:
#   pip install alpaca-py pandas pandas_ta python-dotenv requests
#
# Reads tickers from nasdaqlisted.csv, computes RSI/MACD/EMA/Bollinger/vol,
# applies a confluence score with trend bias, and posts alerts to Discord.
# Also supports a daily digest mode: `python confluence_screener.py --digest`

import os, time, ssl, smtplib, csv, requests, sys
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

ALPACA_API_KEY     = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY  = os.getenv("ALPACA_SECRET_KEY")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise SystemExit("Missing ALPACA_API_KEY/ALPACA_SECRET_KEY in environment (.env or GitHub Secrets)")
if not DISCORD_WEBHOOK_URL:
    print("Warning: DISCORD_WEBHOOK_URL not set — alerts will not be sent to Discord.")

# --- Paths ---
CSV_PATH = Path("nasdaqlisted.csv")  # must be present in repo working directory

# --- Timeframe & Lookback ---
# Use 5-minute bars by default to balance signal quality vs. API load.
TIMEFRAME = TimeFrame.FiveMinutes  # Minute, FiveMinutes, FifteenMinutes, Day
LOOKBACK  = 400

# --- Confluence Weights & thresholds ---
W = dict(RSI=25, MACD=25, EMA=25, BB=15, VOL=10)
BUY_THRESHOLD  = 60
SELL_THRESHOLD = 60
COOLDOWN_SEC   = 30 * 60  # 30 minutes per ticker per direction (within a single run)

# --- Optional email (disabled by default) ---
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
        requests.post(DISCORD_WEBHOOK_URL, json={"content": msg}, timeout=10)
    except Exception as e:
        print(f"[discord] post failed: {e}")

def read_tickers_from_csv(path: Path):
    if not path.exists():
        raise SystemExit(f"Ticker file not found: {path}")
    # Sniff delimiter and locate a symbol column
    with open(path, "r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        dialect = csv.Sniffer().sniff(sample, delimiters="|,\t;")
        delim = dialect.delimiter
    df = pd.read_csv(path, delimiter=delim)

    # Choose symbol column
    cands = [c for c in df.columns if c.lower() in ["symbol","ticker","symbols","tickers"]]
    col = cands[0] if cands else df.columns[0]

    # Optional: filter out ETFs/Trusts/Funds/Warrants/Units/SPAC if a name/description column exists
    name_cols = [c for c in df.columns if c.lower() in ["security name","name","description","company name","issuer name"]]
    if name_cols:
        nm = df[name_cols[0]].astype(str).str.lower()
        etf_mask = ~nm.str.contains(r"\b(etf|trust|fund|warrant|unit|spac)\b", regex=True)
        df = df[etf_mask]

    ser = (
        df[col].astype(str).str.strip()
        .str.replace(r"[^\w\.-]", "", regex=True)
        .replace("", pd.NA).dropna().drop_duplicates()
    )
    tickers = ser.tolist()
    print(f"[tickers] loaded {len(tickers)} symbols from {path.name}")
    return tickers

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def fetch_history(client, tickers):
    # Batch request for a chunk of tickers; returns MultiIndex df (symbol, time)
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
        sub["macd"] = macd["MACD_12_26_9"]
        sub["macds"] = macd["MACDs_12_26_9"]
        sub["ema9"] = ta.ema(sub["close"], 9)
        sub["ema21"] = ta.ema(sub["close"], 21)
        bb = ta.bbands(sub["close"], length=20, std=2.0)
        sub["bbL"] = bb["BBL_20_2.0"]
        sub["bbU"] = bb["BBU_20_2.0"]
        sub["volSma20"] = sub["volume"].rolling(20).mean()
        sub["sma50"] = ta.sma(sub["close"], 50)  # for trend bias
        sub["symbol"] = sym
        out.append(sub)
    return pd.concat(out).sort_index() if out else pd.DataFrame()

def score_row(row, prev):
    # BUY side signals
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

    # SELL side signals
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

    # Trend bias gates
    trend_up   = pd.notna(row.get("sma50")) and pd.notna(row["close"]) and row["close"] > row["sma50"]
    trend_down = pd.notna(row.get("sma50")) and pd.notna(row["close"]) and row["close"] < row["sma50"]

    if not trend_up:
        buy_score = 0  # invalidate BUY if not in uptrend
    if not trend_down:
        sell_score = 0  # invalidate SELL if not in downtrend

    return (buy_score, buy_reasons), (sell_score, sell_reasons)

def scan_once(tickers):
    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
    all_buys, all_sells = [], []
    print(f"[scan] scanning {len(tickers)} tickers, timeframe={TIMEFRAME}, lookback={LOOKBACK}")

    for chunk in chunked(tickers, 120):  # gentler chunk size
        try:
            print(f"[scan] chunk size={len(chunk)}  first={chunk[0]}  last={chunk[-1]}")
            raw = fetch_history(client, chunk)
        except Exception as e:
            print(f"[scan] fetch error: {e}")
            time.sleep(1.0)
            continue

        if raw is None or raw.empty:
            time.sleep(0.25)
            continue

        data = compute_indicators(raw)
        if data is None or data.empty:
            time.sleep(0.25)
            continue

        for sym in data.index.get_level_values(0).unique():
            df = data.xs(sym)
            if len(df) < 3:
                continue
            last, prev = df.iloc[-1], df.iloc[-2]
            (b_score, b_reasons), (s_score, s_reasons) = score_row(last, prev)
            if b_score >= BUY_THRESHOLD:
                all_buys.append((sym, last.name, float(last["close"]), int(b_score), b_reasons))
            if s_score >= SELL_THRESHOLD:
                all_sells.append((sym, last.name, float(last["close"]), int(s_score), s_reasons))

        # brief backoff to be kind to API
        time.sleep(0.25)

    return all_buys, all_sells

_last_fire = defaultdict(lambda: 0)

def run_and_notify():
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

    buys = filter_cooldown(buys, "BUY")
    sells = filter_cooldown(sells, "SELL")

    if buys or sells:
        MAX_LEN = 1800
        lines = ["**Confluence Signals**"]
        if buys:
            lines.append("**BUY**")
            for s, ts, px, sc, rs in buys:
                reason_str = ", ".join(rs) if rs else "-"
                lines.append(f"- {s} @ {ts} — ${px:.2f} (score {sc}) [{reason_str}]")
        if sells:
            lines.append("**SELL**")
            for s, ts, px, sc, rs in sells:
                reason_str = ", ".join(rs) if rs else "-"
                lines.append(f"- {s} @ {ts} — ${px:.2f} (score {sc}) [{reason_str}]")
        msg = "\n".join(lines)

        # chunk messages if too long for Discord
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
        print("[notify] No signals this run.")

def digest():
    """Summarize top scores at close (intended to be run by a separate workflow after market)."""
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

if __name__ == "__main__":
    if "--digest" in sys.argv:
        digest()
    else:
        run_and_notify()
