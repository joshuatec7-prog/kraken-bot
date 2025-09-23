# crypto_bot.py
import os, time, statistics, csv, datetime, requests
from pathlib import Path
from kraken.futures import Trade

# ==== ENV ====
KEY = os.getenv("KRAKENF_KEY", "")
SECRET = os.getenv("KRAKENF_SECRET", "")
PRODUCTS = [p.strip().upper() for p in os.getenv("PRODUCTS","PI_XBTUSD,PI_ETHUSD").split(",") if p.strip()]
TOTAL_BUDGET_EUR = float(os.getenv("TOTAL_BUDGET_EUR","1000"))

SMA_LEN = int(os.getenv("SMA_LEN","20"))
RISE_THR = float(os.getenv("RISE_THR","0.001"))     # >0 → long
FALL_THR = float(os.getenv("FALL_THR","-0.001"))    # <0 → short
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT","0.01"))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT","0.02"))
TRAIL_PCT = float(os.getenv("TRAIL_PCT","0"))       # 0 = uit
LOOP_SECONDS = int(os.getenv("LOOP_SECONDS","5"))
LOG_PATH = Path(os.getenv("LOG_PATH","trades.csv"))

# ==== INIT ====
trade = Trade(key=KEY, secret=SECRET, sandbox=True)
usd_per_market = TOTAL_BUDGET_EUR / max(1, len(PRODUCTS))

prices = {p: [] for p in PRODUCTS}
pos    = {p: None for p in PRODUCTS}      # 'long'|'short'|None
size   = {p: 0    for p in PRODUCTS}
entry  = {p: None for p in PRODUCTS}
trail  = {p: None for p in PRODUCTS}

if not LOG_PATH.exists():
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", newline="") as f:
        csv.writer(f).writerow(["ts","symbol","action","price","size","reason","pos_after"])

def log(sym, action, price, qty, reason, after):
    ts = datetime.datetime.utcnow().isoformat()
    row = [ts, sym, action, f"{price:.2f}", int(qty), reason, after]
    with LOG_PATH.open("a", newline="") as f:
        csv.writer(f).writerow(row)
    print("|".join(map(str,row)), flush=True)

# ==== MARKETS ====
TICKERS_URL = "https://demo-futures.kraken.com/derivatives/api/v3/tickers"

def last_prices():
    r = requests.get(TICKERS_URL, timeout=10)
    r.raise_for_status()
    data = r.json().get("tickers", [])
    mp = {}
    for t in data:
        sym = str(t.get("symbol","")).upper()
        last = t.get("last") or t.get("markPrice") or t.get("bid") or t.get("ask")
        if sym and last is not None:
            mp[sym] = float(last)
    # normalize PF_/PI_ prefix diff
    out = {}
    for want in PRODUCTS:
        if want in mp:
            out[want] = mp[want]
        else:
            # sommige markten heten PF_XBTUSD i.p.v. PI_XBTUSD
            alt = ("PF_" + want.split("_",1)[-1]) if "_" in want else want.replace("PI_","PF_")
            if alt in mp:
                out[want] = mp[alt]
    return out

# ==== ORDERS ====
def qty_from_usd(usd):   # demo perps: 1 contract ≈ $1 notion
    return max(1, round(usd))

def mkt(sym, side, qty, reduce=False):
    trade.send_order(orderType="market", symbol=sym, side=side, size=int(qty), reduceOnly=reduce)

def open_long(sym, px):
    q = qty_from_usd(usd_per_market)
    mkt(sym, "buy", q)
    pos[sym], size[sym], entry[sym] = "long", q, px
    if TRAIL_PCT > 0: trail[sym] = px * (1 - TRAIL_PCT)
    log(sym, "OPEN_LONG", px, q, "signal", "long")

def open_short(sym, px):
    q = qty_from_usd(usd_per_market)
    mkt(sym, "sell", q)
    pos[sym], size[sym], entry[sym] = "short", q, px
    if TRAIL_PCT > 0: trail[sym] = px * (1 + TRAIL_PCT)
    log(sym, "OPEN_SHORT", px, q, "signal", "short")

def close_position(sym, px, reason):
    if pos[sym] is None: return
    side = "sell" if pos[sym] == "long" else "buy"
    mkt(sym, side, size[sym], reduce=True)
    log(sym, "CLOSE_"+pos[sym].upper(), px, size[sym], reason, "flat")
    pos[sym], size[sym], entry[sym], trail[sym] = None, 0, None, None

# ==== LOOP ====
while True:
    try:
        pxs = last_prices()
        for sym in PRODUCTS:
            if sym not in pxs: 
                continue
            px = pxs[sym]
            # buffer
            prices[sym].append(px)
            if len(prices[sym]) > SMA_LEN: prices[sym].pop(0)
            if len(prices[sym]) < SMA_LEN: continue

            sma = statistics.fmean(prices[sym])
            dev = (px - sma) / max(1e-12, sma)

            # trailing stop bijstellen
            if TRAIL_PCT > 0 and pos[sym] == "long":
                trail[sym] = max(trail[sym] or -1e9, px * (1 - TRAIL_PCT))
            elif TRAIL_PCT > 0 and pos[sym] == "short":
                trail[sym] = min(trail[sym] or 1e9, px * (1 + TRAIL_PCT))

            # SL/TP en trail
            if pos[sym] == "long":
                if px <= entry[sym]*(1 - STOP_LOSS_PCT): close_position(sym, px, "STOP_LOSS"); continue
                if px >= entry[sym]*(1 + TAKE_PROFIT_PCT): close_position(sym, px, "TAKE_PROFIT"); continue
                if TRAIL_PCT > 0 and px <= trail[sym]: close_position(sym, px, "TRAIL_STOP"); continue
            elif pos[sym] == "short":
                if px >= entry[sym]*(1 + STOP_LOSS_PCT): close_position(sym, px, "STOP_LOSS"); continue
                if px <= entry[sym]*(1 - TAKE_PROFIT_PCT): close_position(sym, px, "TAKE_PROFIT"); continue
                if TRAIL_PCT > 0 and px >= trail[sym]: close_position(sym, px, "TRAIL_STOP"); continue

            # signal logica
            if pos[sym] is None:
                if dev > RISE_THR: open_long(sym, px)
                elif dev < FALL_THR: open_short(sym, px)
            elif pos[sym] == "long" and dev < 0:
                close_position(sym, px, "flip_long_to_short")
                open_short(sym, px)
            elif pos[sym] == "short" and dev > 0:
                close_position(sym, px, "flip_short_to_long")
                open_long(sym, px)

    except Exception as e:
        print(f"ERROR|{e}", flush=True)

    time.sleep(LOOP_SECONDS)
