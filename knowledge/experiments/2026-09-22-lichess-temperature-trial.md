# Lichess temperature 0.2 trial

Started 2026-09-22T11:21:31.460770+05:30. Status: matchmaking paused after repeated HTTP 429 responses.

- Serving model: `grpo/own-games-exact-10k-20260918`, selected step 5,500. Only sampling temperature changes to 0.2.
- Baseline: rapid rating 1412, RD 45, 572 rated rapid games.
- Target: 100 completed rated standard 10+0 games against online bots, up to 10 simultaneous games. Aborted/no-start games are excluded. Incoming challenges must match the same rated clock.
- Completed plus ongoing games reserve the total budget. The persistent trial cap survives restarts and midnight. After game 100, no new games start and the final rapid rating is stored.
- Matchmaking checks every 10 seconds, retains one pending challenge, and honors server cooldowns. Actual concurrency depends on opponents accepting.
- Production Modal profile: `sparshithsampath`. The old listener was stopped while idle before replacing it. The new listener call is `fc-01M33TSG8RH5C7APG4AWYP0ADA`.
- Model service: one T4 accepting up to 10 concurrent calls; the HTTP handler awaits inference asynchronously.

Verification: 20 scheduling/game tests passed, including reservations near the total cap, 10-game capacity, aborted games, restart/midnight behavior, and terminal-result accounting. Ten simultaneous short-history predictions completed in 2.169 seconds, and ten 100-ply predictions in 2.096 seconds after warm-up. Every returned move was legal. Startup logs confirm temperature 0.2 and trial limits; the first game `PmVLqhxf` received model moves and its terminal result was persisted.

Live source of truth: Modal Dict `nanodanya-lichess-state`, key `budget`, field `trial`. It stores game IDs/status/winner, baseline, and (after completion) `finished_at` and `final_rapid`. Initial local snapshots are `benchmark/results/lichess_temperature02_trial_{start,status}.json`; these are not live counters.

This tests the deployed fine-tune, whereas the offline temperature sweep tested the original champion. A before/after live rating change also reflects opponent mix and rating uncertainty.

## Matchmaking paused

2026-09-22T13:48:51+05:30: Set the existing Modal Dict `drain` flag to `True` after repeated generic HTTP 429 responses. Trial counters remain saved (rapid 78/100, bullet 28/100); automatic matchmaking stays paused until the flag is explicitly cleared. Before resuming, use a shared bot-game budget across time controls and increasing backoff after repeated rate limits. Official guidance: https://lichess.org/page/api-tips ; server bot-game cap: https://github.com/lichess-org/lila/blob/master/modules/bot/src/main/BotLimit.scala . The generic responses did not identify the triggering limit or reset time.
