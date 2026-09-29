# nanoDanya Lichess bot

This is the production Lichess listener for nanoDanya.

The listener is a cheap Modal function that stays connected to Lichess and calls
the separate nanoDanya model endpoint only when it needs to make a move.

The endpoint serves `grpo/own-games-exact-10k-20260918`, selected step 5,500,
since 2026-09-19. Move requests now use temperature **0.2**.
A rapid trial (`budget.trial` in Modal Dict `nanodanya-lichess-state`) targets
100 completed rated 10+0 games, with up to 10 games simultaneously. Aborts do not
count. Completed plus ongoing games reserve the total budget. Incoming challenges
must also be rated 10+0 and share the same budget and concurrency cap. The trial
state, baseline, game outcomes, and final rapid rating persist in the Dict.
Startup reconciles against Lichess history. When a trial completes, it moves to
`budget.past_trials` after its final rating is recorded, and the bot returns to
normal limits: five games per IST day with a two-hour gap. One listener owns all
games; its matchmaker checks every 10 seconds, keeps only one pending challenge,
and respects Lichess rate-limit cooldowns.

The current trial started 2026-09-29 at **1525 rapid, 650 rated rapid games**.
The earlier 0.2 trials (rapid 78/100, 1412 -> 1525; bullet 28/100) are in
`past_trials`. A `bullet_trial` key (seeded from `bullet_config`) runs 1+1 bullet
the same way, one game at a time with a warm-up inference request first; none
is set now.

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
