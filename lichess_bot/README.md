# nanoDanya Lichess bot

This is the production Lichess listener for nanoDanya.

The listener is a cheap Modal function that stays connected to Lichess and calls
the separate nanoDanya model endpoint only when it needs to make a move.

The endpoint serves `grpo/own-games-exact-10k-20260918`, selected step 5,500,
since 2026-09-19. Move requests now use temperature **0.2**.
The temperature trial starts at **1412 rapid, 572 rated rapid games** and targets
100 completed rated 10+0 games, with up to 10 games simultaneously. Aborts do not
count. Completed plus ongoing games reserve the total budget; matchmaking stops
at 100 completed games and remains stopped across restarts and midnight. Incoming
challenges must also be rated 10+0 and share the same budget and concurrency cap.
The trial state, baseline, game outcomes, and final rapid rating persist under
`budget.trial` in Modal Dict `nanodanya-lichess-state`. Startup reconciles against
Lichess history. One listener owns all games; its matchmaker checks every 10 seconds,
keeps only one pending challenge, and respects Lichess rate-limit cooldowns.
Without a trial, normal limits are five games per IST day with a two-hour gap.

A separate `budget.bullet_trial` now targets 100 completed rated **1+1 bullet**
games at temperature 0.2, one bullet game at a time within the overall 10-game
cap. It starts with zero rated bullet games (the displayed 3000 rating is
provisional, RD 500). `bullet_config` seeds this counter once; it never resets
an existing trial. Game events/history route each result by speed. Each trial
has its own final rating and stops independently. Before sending or accepting
a bullet challenge, the listener makes a warm-up inference request. Server
challenge limits are shared across both trials and may delay completion.

Inference accepts 10 concurrent inputs on one T4; the HTTP endpoint awaits the
remote model asynchronously. Ten simultaneous legal-move requests are checked
before enabling the trial. Serving and listener deployments use the
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
