import os, time, statistics, csv, datetime
from pathlib import Path
from kraken.futures import Trade, Market

# API keys uit environment
KEY = os.getenv("KRAKENF_KEY")
SECRET = os.getenv("KRAKENF_SECRET")

# Coins en budget
PRODUCTS = [p.strip() for p in os.getenv("PRODUCTS","").split(",") if p.strip()]
TOTAL_BUDGET_EUR = float(os.getenv("TOTAL_BUDGET_EUR","1000"))

# Strategie instellingen
SMA_LEN = int(os.getenv("SMA_LEN","20"))
RISE_THR = float(os.getenv("RISE_THR","0.001"))
FALL_THR = float(os.getenv("FALL_THR","-0.001"))
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT","0.01"))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT","0.02"))
LOOP_SECONDS = int(os.getenv("LOOP_SECONDS","5"))

# Init Kraken demo
market = Market(sandbox=True)
trade  = Trade(key=KEY, secret=SECRET, sandbox=True)

usd_per_market = TOTAL_BUDGET_EUR / max(1, len(PRODUCTS))

prices, pos, size, entry = {}, {}, {}, {}
for p in PRODUCTS:
    prices[p], pos[p], size[p], entry[p] = [], None, 0, None

# Logfile
log_path = Path("trades.csv")
if not log_path.exists():
    with log_path.open("w", newline="") as f:
        csv.writer(f).writerow(
            ["ts","symbol","action","price","size","reason","pos_after"]
        )

def log(sym, action, price, qty, reason, after):
    ts = datetime.datetime.utcnow().isoformat()
    row = [ts, sym, action, f"{price:.2f}", int(qty), reason, after]
    with log_path.open("a", newline="") as f:
        csv.writer(f).writerow(row)
    print("|".join(map(str,row)), flush=True)

def last_prices():
    t = market.tickers()
    return {p: float(t["tickers"][p]["last"]) for p in PRODUCTS}

def qty_from_usd(usd): return max(1, round(usd))

def mkt(sym, side, qty, reduce=False):
    trade.send_order(orderType="market", symbol=sym, side=side, size=qty, reduceOnly=reduce)

def open_long(sym, px):
    q = qty_from_usd(usd_per_market)
    mkt(sym, "buy", q)
    pos[sym], size[sym], entry[sym] = "long", q, px
    log(sym, "OPEN_LONG", px, q, "signal", "long")

def open_short(sym, px):
    q = qty_from_usd(usd_per_market)
    mkt(sym, "sell", q)
    pos[sym], size[sym], entry[sym] = "short", q, px
    log(sym, "OPEN_SHORT", px, q, "signal", "short")

def close_position(sym, px, reason):
    if pos[sym] is None: return
    side = "sell" if pos[sym]=="long" else "buy"
    mkt(sym, side, size[sym], reduce=True)
    log(sym, "CLOSE_"+pos[sym].upper(), px, size[sym], reason, "flat")
    pos[sym], size[sym], entry[sym] = None, 0, None

# Hoofdlus
while True:
    try:
        pxs = last_prices()
        for sym, px in pxs.items():
            prices[sym].append(px)
            if len(prices[sym]) > SMA_LEN: prices[sym].pop(0)
            if len(prices[sym]) < SMA_LEN: continue

            sma = statistics.fmean(prices[sym])
            dev = (px - sma) / sma

            # Stop-loss & take-profit
            if pos[sym] == "long":
                if px <= entry[sym]*(1 - STOP_LOSS_PCT):
                    close_position(sym, px, "STOP_LOSS"); continue
                if px >= entry[sym]*(1 + TAKE_PROFIT_PCT):
                    close_position(sym, px, "TAKE_PROFIT"); continue
            elif pos[sym] == "short":
                if px >= entry[sym]*(1 + STOP_LOSS_PCT):
                    close_position(sym, px, "STOP_LOSS"); continue
                if px <= entry[sym]*(1 - TAKE_PROFIT_PCT):
                    close_position(sym, px, "TAKE_PROFIT"); continue

            # Signalen
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
