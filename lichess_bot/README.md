# nanoDanya Lichess bot

This is the production Lichess listener for nanoDanya.

The listener is a cheap Modal function that stays connected to Lichess and calls
the separate nanoDanya model endpoint only when it needs to make a move.

The endpoint serves the L16 whole-game checkpoint
`plain/whole-game-20261016/l16_final` (step 250,000), deployed on 2026-10-19.
Move requests use temperature **0.2**, legal masking, and canonical SAN history.
The server verifies the checkpoint SHA-256 before loading it and uses the same
GPT implementation as training.

`budget.daily` in Modal Dict `nanodanya-lichess-state` targets up to **100 game
starts per rolling 24 hours**, with five simultaneous rated 10+0 rapid games and
at least 60 seconds between starts. Incoming challenges share these limits and
must also be rated 10+0. Aborted games still count toward the daily start cap.
Lichess cooldowns take precedence. The listener reconciles recent game history
on startup so restarts cannot reset the allowance.

`budget.evaluation` stores the model identity, game IDs and outcomes, and rapid
rating snapshots. This rollout started at **1566 rapid after 832 rated rapid
games**. The latest rating is refreshed after completed games; daily snapshots
use IST dates. Previous trials remain under `budget.past_trials`. Removing
`budget.daily` restores the normal five-game daily limit and two-hour spacing.

Inference accepts 10 concurrent inputs on one T4; the HTTP endpoint awaits the
remote model asynchronously. Ten simultaneous legal-move requests are checked
before enabling matchmaking. Serving and listener deployments use the
`sparshithsampath` Modal profile (the benchmark sweep used `second` / `workspace2`).
Failed move submissions close the game stream and reconnect after five seconds
to read the current position before retrying (65 seconds for HTTP 429). An
unchanged position reuses the selected move; an accepted move is not replayed.
Serving changes deploy separately with `modal deploy inference/serve.py`;
an already running listener picks them up through the same endpoint.

```bash
modal deploy lichess_bot/app.py
```

Use `modal run lichess_bot/app.py` to check the account and effective limits
without starting a listener or sending challenges. Before replacing the live
listener, set the Dict's `drain` key to true, wait for existing games to finish,
and stop the old deployment. Clear `drain` before starting its replacement; do not run two
listeners against the same Lichess token.

To start one run immediately after deploy:

```bash
python - <<'PY'
import modal

modal.Function.from_name("nanodanya-lichess-bot", "run_bot").spawn()
PY
```
