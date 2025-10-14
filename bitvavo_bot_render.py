#!/usr/bin/env python3
# Spot-bot voor Bitvavo met SMA-cross. Verkoopt alleen wat de bot zelf kocht.
# Vereist: ccxt, PyYAML, pandas, numpy
# Env: BITVAVO_API_KEY, BITVAVO_API_SECRET, optioneel STATE_FILE=/data/state.json

import os, time, json, math, signal, logging
from datetime import datetime, timedelta

import ccxt
import numpy as np
import pandas as pd
import yaml

STATE_FILE = os.getenv("STATE_FILE", "state.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                s = json.load(f)
        except Exception:
            s = {}
    else:
        s = {}
    s.setdefault("last_trade_ts", {})
    s.setdefault("owned", {})  # per market: True als door deze bot gekocht
    return s

def save_state(state):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logging.warning(f"Kon state niet wegschrijven: {e}")

def init_exchange():
    key = os.getenv("BITVAVO_API_KEY")
    sec = os.getenv("BITVAVO_API_SECRET")
    if not key or not sec:
        raise RuntimeError("Set BITVAVO_API_KEY en BITVAVO_API_SECRET")
    ex = ccxt.bitvavo({"apiKey": key, "secret": sec, "enableRateLimit": True})
    ex.load_markets()
    return ex

def pair(base, quote): return f"{base}/{quote}"

def fetch_ohlcv_df(ex, market, timeframe="15m", limit=150):
    o = ex.fetch_ohlcv(market, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(o, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def sma(series, n): return series.rolling(n).mean()

def crossover(a, b):
    return len(a) >= 2 and len(b) >= 2 and a.iloc[-2] <= b.iloc[-2] and a.iloc[-1] > b.iloc[-1]

def crossunder(a, b):
    return len(a) >= 2 and len(b) >= 2 and a.iloc[-2] >= b.iloc[-2] and a.iloc[-1] < b.iloc[-1]

def get_free_quote(ex, quote):
    bal = ex.fetch_balance()
    return float(bal.get(quote, {}).get("free", 0.0) or 0.0)

def get_open_orders(ex, market):
    try: return ex.fetch_open_orders(market)
    except Exception: return []

def has_open_buy(ex, market):
    return any(o.get("side") == "buy" for o in get_open_orders(ex, market))

def get_base_amount(ex, base):
    bal = ex.fetch_balance()
    return float(bal.get(base, {}).get("total", 0.0) or 0.0)

def count_positions(ex, markets):
    c = 0
    for m in markets:
        base = m.split("/")[0]
        if get_base_amount(ex, base) > 0.0000001:
            c += 1
    return c

def calc_per_coin_limit(max_open_positions, pct):
    return max(1, math.floor(max_open_positions * pct / 100.0))

def allowed_to_trade(state, market, cooldown_min):
    ts = state["last_trade_ts"].get(market)
    if not ts: return True
    return datetime.utcnow() >= datetime.fromisoformat(ts) + timedelta(minutes=cooldown_min)

def mark_traded(state, market):
    state["last_trade_ts"][market] = datetime.utcnow().isoformat(timespec="seconds")

def place_market_buy(ex, market, amount_quote):
    price = float(ex.fetch_ticker(market)["last"])
    amount_base = amount_quote / price
    amount_base = ex.amount_to_precision(market, amount_base)
    logging.info(f"BUY {market} ~{amount_base} base for ~{amount_quote}")
    return ex.create_order(market, "market", "buy", amount_base)

def place_market_sell_all(ex, market):
    base = market.split("/")[0]
    amt = ex.amount_to_precision(market, get_base_amount(ex, base))
    if float(amt) <= 0:
        return None
    logging.info(f"SELL {market} amount {amt}")
    return ex.create_order(market, "market", "sell", amt)

def run():
    cfg = load_config()
    state = load_state()
    ex = init_exchange()

    quote = cfg["quote"]
    markets = [pair(sym, quote) for sym in cfg["symbols"] if pair(sym, quote) in ex.markets]
    if not markets:
        raise RuntimeError("Geen geldige markten voor de opgegeven symbols/quote")

    per_coin_limit = calc_per_coin_limit(cfg["max_open_positions"], cfg["max_pct_per_coin"])
    logging.info(f"Markets: {', '.join(markets)} | per-coin limit: {per_coin_limit}")

    running = True
    def stop(*_): 
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    while running:
        try:
            total_positions = count_positions(ex, markets)
            free_q = get_free_quote(ex, quote)
            logging.info(f"Positions {total_positions}/{cfg['max_open_positions']} | Free {quote}: {free_q:.2f}")

            for m in markets:
                base = m.split("/")[0]

                # Koop-guards
                if cfg["only_buy_if_not_in_position"] and get_base_amount(ex, base) > 0.0000001:
                    continue
                if cfg["single_open_buy_per_coin"] and has_open_buy(ex, m):
                    continue
                if not allowed_to_trade(state, m, cfg["cooldown_minutes"]):
                    continue
                if total_positions >= cfg["max_open_positions"]:
                    break

                # Signaal
                df = fetch_ohlcv_df(ex, m, timeframe=cfg["timeframe"], limit=120)
                df["sma_fast"] = sma(df["close"], cfg["sma_fast"])
                df["sma_slow"] = sma(df["close"], cfg["sma_slow"])

                if crossover(df["sma_fast"], df["sma_slow"]):
                    free_q = get_free_quote(ex, quote)
                    stake = cfg["fixed_stake_quote"] if cfg["fixed_stake_quote"] > 0 else free_q * cfg["stake_fraction"]
                    min_cost = (ex.markets[m]["limits"].get("cost", {}) or {}).get("min") or 5.0
                    amount_quote = max(min_cost, min(stake, free_q))
                    if amount_quote >= min_cost and amount_quote > 0:
                        try:
                            place_market_buy(ex, m, amount_quote)
                            mark_traded(state, m)
                            state["owned"][m] = True         # markeer als door deze bot gekocht
                            save_state(state)
                            total_positions += 1
                        except Exception as e:
                            logging.error(f"BUY failed {m}: {e}")

                # Verkoop alleen als toegestaan en alleen eigen posities
                if not cfg["sell_enabled"]:
                    continue
                if not state["owned"].get(m, False):
                    continue
                if get_base_amount(ex, base) <= 0:
                    continue
                if crossunder(df["sma_fast"], df["sma_slow"]):
                    try:
                        place_market_sell_all(ex, m)
                        mark_traded(state, m)
                        state["owned"][m] = False
                        save_state(state)
                    except Exception as e:
                        logging.error(f"SELL failed {m}: {e}")

            time.sleep(cfg["sleep_seconds"])
        except Exception as e:
            logging.error(repr(e))
            time.sleep(5)

if __name__ == "__main__":
    run()
