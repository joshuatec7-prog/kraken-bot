# Bitvavo CH-Style Worker

## Render
- Service type: Background Worker
- Build: `pip install -r requirements.txt`
- Start: `python bitvavo_bot_ch.py`
- Env: `BITVAVO_API_KEY`, `BITVAVO_API_SECRET`
- Optioneel: Disk `/data`, env `STATE_FILE=/data/state.json`

## Functies
- SMA of externe signalen
- Trailing stop-buy/stop-loss, take profit
- DCA stappen
- Limieten: max posities, per-munt, cooldown, enkel eigen posities
- Standaard geen verkopen (`sell_enabled: false`)
