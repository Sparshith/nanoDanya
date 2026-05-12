# nanoDanya Lichess bot

This is the production Lichess listener for nanoDanya.

The listener is a cheap Modal function that stays connected to Lichess and calls
the separate nanoDanya model endpoint only when it needs to make a move.

```bash
modal deploy lichess_bot/app.py
```

To start one run immediately after deploy:

```bash
python - <<'PY'
import modal

modal.Function.from_name("nanodanya-lichess-bot", "run_bot").spawn()
PY
```

