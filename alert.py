"""
Support/Resistance Re-test Finder [ChartPrime] -> Telegram alert
GitHub Actions (free cloud) version: PC ba phone khola lagbe na.

Mode:
    python alert.py once   -> ekbar check kore alert pathay (GitHub eta chalay)
    python alert.py test   -> ager 20ta signal print kore
    python alert.py ping   -> Telegram-e ekta test message pathay

BOT_TOKEN ar CHAT_ID GitHub Secrets theke ashe. Code-e kokhono likho na.
"""
import json
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ================= SETTING (ekhane bodlao) =================
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
CHAT_ID = os.environ.get("CHAT_ID", "")

# Nam : Yahoo symbol
SYMBOLS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
}

INTERVAL = "15m"
PERIOD = "30d"
TIMEZONE = "Asia/Kolkata"   # message-e shomoy ei timezone-e dekhabe

# Indicator setting (tomar screenshot theke)
DETECTION_METHOD = "Wick"   # "Wick" ba "Body"
LEFT_BARS = 10
RIGHT_BARS = 10
MAX_AGE_BARS = 1000         # original script-er fixed limit, bodlio na

LOOKBACK_BARS = 4           # sesh koto ta closed candle dekhbe (GitHub deri korle jate miss na hoy)
STATE_FILE = "state.json"
# ===========================================================

BAR_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60}[INTERVAL]


def send_telegram(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, data={"chat_id": CHAT_ID, "text": text}, timeout=15)
        r.raise_for_status()
        return True
    except Exception as e:
        print("Telegram error:", e)
        return False


def fmt_time(ts):
    try:
        return ts.tz_convert(TIMEZONE).strftime("%d %b %H:%M")
    except Exception:
        return str(ts)


def to_utc_iso(ts):
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%S")


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=1, sort_keys=True)


def get_closed_candles(ticker):
    """Data niye ashe, ar ekhono cholche emon candle ta bad dey."""
    df = yf.download(
        ticker, interval=INTERVAL, period=PERIOD,
        progress=False, auto_adjust=False,
    )
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    if df.empty:
        return None

    now = pd.Timestamp.now(tz=df.index.tz)
    if df.index[-1] + timedelta(minutes=BAR_MINUTES) > now:
        df = df.iloc[:-1]  # sesh candle ekhono closed hoyni
    return df


def find_retests(df):
    """
    Pine Script-er logic hubohu Python-e.
    Return: {bar_position: (support_hits, resistance_hits)}
    prottek hit = (level_price, pivot_bar_position)
    """
    o = df["Open"].to_numpy(dtype=float)
    h = df["High"].to_numpy(dtype=float)
    l = df["Low"].to_numpy(dtype=float)
    c = df["Close"].to_numpy(dtype=float)
    n = len(o)

    if DETECTION_METHOD == "Wick":
        src_hi, src_lo = h, l
    else:
        src_hi = np.maximum(o, c)
        src_lo = np.minimum(o, c)

    first_high, first_low = [], []      # pivot toiri hoyeche, ekhono break hoyni
    second_high, second_low = [], []    # pivot break hoyeche, retest-er opekkhay
    events = {}

    for t in range(n):
        sup_hits, res_hits = [], []

        # ---- notun pivot (RIGHT_BARS ager candle-er jonno) ----
        cand = t - RIGHT_BARS
        if cand - LEFT_BARS >= 0:
            v = src_hi[cand]
            if (src_hi[cand - LEFT_BARS:cand] < v).all() and (src_hi[cand + 1:t + 1] <= v).all():
                first_high.append((v, cand))
            v = src_lo[cand]
            if (src_lo[cand - LEFT_BARS:cand] > v).all() and (src_lo[cand + 1:t + 1] >= v).all():
                first_low.append((v, cand))

        # ---- Support re-test (green R) ----
        for i in range(len(second_high) - 1, -1, -1):
            price, idx = second_high[i]
            if l[t] <= price and c[t] > price and c[t] > o[t]:
                sup_hits.append((price, idx))
                del second_high[i]
            elif t - idx > MAX_AGE_BARS or c[t] < price:
                del second_high[i]

        # ---- Resistance re-test (red R) ----
        for i in range(len(second_low) - 1, -1, -1):
            price, idx = second_low[i]
            if h[t] >= price and c[t] < price and c[t] < o[t]:
                res_hits.append((price, idx))
                del second_low[i]
            elif t - idx > MAX_AGE_BARS or c[t] > price:
                del second_low[i]

        # ---- pivot high upore break hole second phase-e jao ----
        for i in range(len(first_high) - 1, -1, -1):
            price, idx = first_high[i]
            if l[t] > price:
                second_high.append((price, idx))
                del first_high[i]
            elif t - idx > MAX_AGE_BARS:
                del first_high[i]

        # ---- pivot low niche break hole second phase-e jao ----
        for i in range(len(first_low) - 1, -1, -1):
            price, idx = first_low[i]
            if h[t] < price:
                second_low.append((price, idx))
                del first_low[i]
            elif t - idx > MAX_AGE_BARS:
                del first_low[i]

        if sup_hits or res_hits:
            events[t] = (sup_hits, res_hits)

    return events


def run_once():
    state = load_state()
    for name, ticker in SYMBOLS.items():
        try:
            df = get_closed_candles(ticker)
            if df is None or len(df) < 100:
                print(name, "- data paoa jayni")
                continue

            events = find_retests(df)
            last = len(df) - 1
            print(name, "- checked, sesh closed candle:", fmt_time(df.index[last]))

            for t in range(max(0, last - LOOKBACK_BARS + 1), last + 1):
                if t not in events:
                    continue
                sup, res = events[t]
                candle_iso = to_utc_iso(df.index[t])
                close = df["Close"].iloc[t]

                for kind, hits, label in (
                    ("SUPPORT", sup, "🟢 Green R - Support Re-test"),
                    ("RESISTANCE", res, "🔴 Red R - Resistance Re-test"),
                ):
                    key = f"{name}|{kind}"
                    if hits and state.get(key, "") < candle_iso:
                        msg = (
                            f"{label}\n"
                            f"{name} | {INTERVAL}\n"
                            f"Level: {hits[0][0]:.5f}\n"
                            f"Close: {close:.5f}\n"
                            f"Candle: {fmt_time(df.index[t])}"
                        )
                        print(msg, "\n")
                        if send_telegram(msg):
                            state[key] = candle_iso
        except Exception as e:
            print(name, "error:", e)
    save_state(state)


def test_mode():
    for name, ticker in SYMBOLS.items():
        df = get_closed_candles(ticker)
        if df is None:
            print(name, "- data paoa jayni")
            continue
        events = find_retests(df)
        print(f"\n===== {name} | ager 20ta signal =====")
        for t in sorted(events)[-20:]:
            sup, res = events[t]
            for price, _ in sup:
                print(fmt_time(df.index[t]), "| GREEN R (Support) | level", f"{price:.5f}")
            for price, _ in res:
                print(fmt_time(df.index[t]), "| RED R (Resistance) | level", f"{price:.5f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "once"
    if mode == "test":
        test_mode()
    elif mode == "ping":
        ok = send_telegram("✅ GitHub alert bot connected. Ekhon theke alert ashbe.")
        print("ping sent" if ok else "ping FAILED - BOT_TOKEN/CHAT_ID check koro")
    else:
        run_once()
