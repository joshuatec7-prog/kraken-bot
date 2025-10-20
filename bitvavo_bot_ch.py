#!/usr/bin/env python3
# Bitvavo worker met CH-achtige features + duidelijke diagnose-logs

import os, time, json, math, signal, logging
from datetime import datetime, timedelta

import ccxt
import numpy as np
import pandas as pd
import yaml
import requests

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
    s = {}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                s = json.load(f)
        except Exception:
            s = {}
    s.setdefault("last_trade_ts", {})
    s.setdefault("positions", {})  # market -> {owned, entry_price, peak_price, dca_level, last_buy_price}
    s.setdefault("tsb", {})        # market -> {armed, min_price}
    return s

def save_state(state):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        logging.warning(f"Kon state niet opslaan: {e}")

def init_exchange():
    key = os.getenv("BITVAVO_API_KEY")
    sec = os.getenv("BITVAVO_API_SECRET")
    if not key or not sec:
        raise RuntimeError("Set BITVAVO_API_KEY en BITVAVO_API_SECRET")
    ex = ccxt.bitvavo({"apiKey": key, "secret": sec, "enableRateLimit": True})
    ex.load_markets()
    return ex

def pair(base, quote): 
    return f"{base}/{quote}"

def fetch_ohlcv_df(ex, market, timeframe="5m", limit=150):
    o = ex.fetch_ohlcv(market, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(o, columns=["ts","open","high","low","close","volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def sma(series, n): 
    return series.rolling(n).mean()

def crossover(a, b):
    return len(a)>=2 and len(b)>=2 and a.iloc[-2] <= b.iloc[-2] and a.iloc[-1] > b.iloc[-1]

def crossunder(a, b):
    return len(a)>=2 and len(b)>=2 and a.iloc[-2] >= b.iloc[-2] and a.iloc[-1] < b.iloc[-1]

def get_free_quote(ex, quote):
    bal = ex.fetch_balance()
    return float(bal.get(quote, {}).get("free", 0.0) or 0.0)

def get_open_orders(ex, market):
    try:
        return ex.fetch_open_orders(market)
    except Exception:
        return []

def has_open_buy(ex, market):
    return any(o.get("side") == "buy" for o in get_open_orders(ex, market))

def get_base_amount(ex, base):
    bal = ex.fetch_balance()
    return float(bal.get(base, {}).get("total", 0.0) or 0.0)

def min_cost_for_market(ex, market, default=5.0):
    try:
        return float((ex.markets[market].get("limits", {}).get("cost", {}) or {}).get("min") or default)
    except Exception:
        return default

def in_position(ex, market):
    """Beschouw alleen echte posities; 'dust' telt niet."""
    base = market.split("/")[0]
    amt = get_base_amount(ex, base)
    if amt <= 0:
        return False
    price = float(ex.fetch_ticker(market)["last"])
    notional = amt * price
    threshold = max(5.0, 0.9 * min_cost_for_market(ex, market))
    return notional >= threshold

def count_positions(ex, markets):
    c = 0
    for m in markets:
        if in_position(ex, m):
            c += 1
    return c

def allowed_to_trade(state, market, cooldown_min):
    ts = state["last_trade_ts"].get(market)
    if not ts:
        return True
    try:
        last = datetime.fromisoformat(ts)
    except Exception:
        return True
    return datetime.utcnow() >= last + timedelta(minutes=cooldown_min)

def mark_traded(state, market):
    state["last_trade_ts"][market] = datetime.utcnow().isoformat(timespec="seconds")

def place_market_buy(ex, market, amount_quote):
    price = float(ex.fetch_ticker(market)["last"])
    amount_base = ex.amount_to_precision(market, amount_quote / price)
    logging.info(f"BUY {market} ~{amount_base} base for ~{amount_quote:.2f}")
    return ex.create_order(market, "market", "buy", amount_base), price

def place_market_sell_all(ex, market):
    base = market.split("/")[0]
    amt = ex.amount_to_precision(market, get_base_amount(ex, base))
    if float(amt) <= 0:
        return None, None
    price = float(ex.fetch_ticker(market)["last"])
    logging.info(f"SELL {market} amount {amt}")
    return ex.create_order(market, "market", "sell", amt), price

def fetch_external_signals(url):
    try:
        r = requests.get(url, timeout=5)
        j = r.json()
        return set(j.get("buy", [])), set(j.get("sell", []))
    except Exception as e:
        logging.warning(f"Kon externe signalen niet laden: {e}")
        return set(), set()

def run():
    cfg = load_config()
    state = load_state()
    ex = init_exchange()

    quote = cfg["quote"]
    # Alleen geldige markten opnemen
    markets = [pair(sym, quote) for sym in cfg["symbols"] if pair(sym, quote) in ex.markets]
    if not markets:
        raise RuntimeError("Geen geldige markten voor de opgegeven symbols/quote")

    logging.info(f"Loaded markets: {', '.join(markets)}")
    logging.info(f"Config: tf={cfg['timeframe']} SMA={cfg['sma_fast']}/{cfg['sma_slow']} TP={cfg['tp_pct']}% TSL={cfg['tsl_pct']}%")

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

            # externe signalen
            ext_buy, ext_sell = set(), set()
            if cfg.get("signals_mode", "sma") == "external" and cfg.get("signals_url"):
                ext_buy, ext_sell = fetch_external_signals(cfg["signals_url"])

            for m in markets:
                base = m.split("/")[0]
                price = float(ex.fetch_ticker(m)["last"])
                min_cost = min_cost_for_market(ex, m)

                pos = state["positions"].setdefault(
                    m, {"owned": False, "entry_price": None, "peak_price": None, "dca_level": 0, "last_buy_price": None}
                )
                tsb = state["tsb"].setdefault(m, {"armed": False, "min_price": None})

                # KOOPFLOW
                already_in_pos = in_position(ex, m)
                if cfg["only_buy_if_not_in_position"] and already_in_pos:
                    logging.info(f"skip BUY {m} reason=already_in_position")
                elif cfg["single_open_buy_per_coin"] and has_open_buy(ex, m):
                    logging.info(f"skip BUY {m} reason=open_buy_order_present")
                elif not allowed_to_trade(state, m, cfg["cooldown_minutes"]):
                    logging.info(f"skip BUY {m} reason=cooldown")
                elif total_positions >= cfg["max_open_positions"]:
                    logging.info(f"skip BUY {m} reason=max_open_positions_reached")
                elif free_q < min_cost:
                    logging.info(f"skip BUY {m} reason=insufficient_free_{quote} need>={min_cost:.2f}")
                else:
                    # signaal bepalen
                    want_buy = False
                    if cfg.get("signals_mode", "sma") == "external":
                        want_buy = m in ext_buy
                        logging.info(f"{m} external_buy={want_buy}")
                    else:
                        df = fetch_ohlcv_df(ex, m, timeframe=cfg["timeframe"], limit=120)
                        df["sma_fast"] = sma(df["close"], cfg["sma_fast"])
                        df["sma_slow"] = sma(df["close"], cfg["sma_slow"])
                        want_buy = crossover(df["sma_fast"], df["sma_slow"])
                        logging.info(f"{m} sma_crossover={want_buy}")

                    # Trailing Stop-Buy
                    if want_buy and cfg.get("tsb_enabled", False):
                        if not tsb["armed"]:
                            tsb["armed"] = True
                            tsb["min_price"] = price
                            want_buy = False
                            logging.info(f"{m} TSB armed at {price}")
                        else:
                            tsb["min_price"] = min(tsb["min_price"], price)
                            trigger = price >= tsb["min_price"] * (1 + cfg["tsb_trail_pct"]/100.0)
                            want_buy = trigger
                            if trigger:
                                logging.info(f"{m} TSB triggered min={tsb['min_price']}")
                                tsb["armed"] = False
                                tsb["min_price"] = None

                    if want_buy:
                        # inzet bepalen
                        free_q = get_free_quote(ex, quote)
                        stake = cfg["fixed_stake_quote"] if cfg["fixed_stake_quote"] > 0 else free_q * cfg["stake_fraction"]
                        amount_quote = max(min_cost, min(stake, free_q))
                        try:
                            _, fill_price = place_market_buy(ex, m, amount_quote)
                            mark_traded(state, m)
                            pos["owned"] = True
                            pos["entry_price"] = fill_price if pos["entry_price"] is None else pos["entry_price"]
                            pos["peak_price"] = max(pos["peak_price"] or fill_price, fill_price)
                            pos["last_buy_price"] = fill_price
                            save_state(state)
                            if not already_in_pos:
                                total_positions += 1
                        except Exception as e:
                            logging.error(f"BUY failed {m}: {e}")
                    else:
                        logging.info(f"skip BUY {m} reason=no_signal")

                # DCA
                if pos["owned"] and cfg.get("dca_enabled", False) and pos["dca_level"] < cfg.get("dca_steps", 0):
                    ref = pos["last_buy_price"] or pos["entry_price"] or price
                    if price <= ref * (1 - cfg.get("dca_drop_pct", 5.0)/100.0):
                        free_q = get_free_quote(ex, quote)
                        base_stake = cfg["fixed_stake_quote"] if cfg["fixed_stake_quote"] > 0 else free_q * cfg["stake_fraction"]
                        mult = cfg.get("dca_stake_multiplier", 1.0) ** pos["dca_level"]
                        stake_q = base_stake * mult
                        amount_quote = max(min_cost, min(stake_q, free_q))
                        if amount_quote >= min_cost and amount_quote > 0:
                            try:
                                _, fill_price = place_market_buy(ex, m, amount_quote)
                                mark_traded(state, m)
                                pos["last_buy_price"] = fill_price
                                pos["dca_level"] += 1
                                save_state(state)
                                logging.info(f"DCA BUY {m} level={pos['dca_level']} at {fill_price}")
                            except Exception as e:
                                logging.error(f"DCA BUY failed {m}: {e}")

                # VERKOOPFLOW
                if not cfg["sell_enabled"]:
                    continue
                if cfg.get("manage_only_own_positions", True) and not pos["owned"]:
                    continue
                if not in_position(ex, m):
                    continue

                pos["peak_price"] = max(pos["peak_price"] or price, price)
                want_sell = False
                reason = None

                if cfg.get("tp_pct", 0) > 0 and pos["entry_price"]:
                    if price >= pos["entry_price"] * (1 + cfg["tp_pct"]/100.0):
                        want_sell_
