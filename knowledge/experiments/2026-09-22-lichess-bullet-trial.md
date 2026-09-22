# Bullet temperature 0.2 trial

Started 2026-09-22 alongside the rapid trial. Target: 100 completed rated standard 1+1 bullet games, temperature 0.2, on `grpo/own-games-exact-10k-20260918`. One bullet game at a time within the existing 10-game overall limit. The original rapid trial continues with its original counter.

Baseline bullet: zero rated games, provisional rating 3000, RD 500. This is not an established strength estimate. The listener warms the model with an inference request before sending or accepting bullet challenges. Separate durable counters reserve ongoing slots, exclude aborted/no-start games, and capture each final rating at 100 completed games.

State: `nanodanya-lichess-state` in the `sparshithsampath` Modal profile, `budget.bullet_trial`; initial configuration under `bullet_config`. Local snapshots: `benchmark/results/lichess_bullet_trial_{start,status}.json`.

Validation: 23 scheduling/game tests, including speed separation, independent caps across restarts, and unconfigured bullet rejection. The replacement listener recovered four active rapid games in about six seconds. First bullet game `eiLbB6GG` completed by checkmate and was counted only in the bullet trial; a second bullet game started while rapid games continued. Listener call: `fc-01M33VVYRMT476G6ZDEYYC84JN`.

Future deployments can use the new Dict `drain` flag to stop challenges, finish active games, and exit before replacement. Shared server rate limits are honored across both trials.

Temperature benchmark dashboard: https://app.artifactbin.dev/a/OyGiWE
This shows the original champion versus SF1500, not direct temperature-versus-temperature matches or live Lichess results.

## Matchmaking paused

2026-09-22T13:48:51+05:30: Set the existing Modal Dict `drain` flag to `True` after repeated generic HTTP 429 responses. Trial counters remain saved (rapid 78/100, bullet 28/100); automatic matchmaking stays paused until the flag is explicitly cleared. Before resuming, use a shared bot-game budget across time controls and increasing backoff after repeated rate limits. Official guidance: https://lichess.org/page/api-tips ; server bot-game cap: https://github.com/lichess-org/lila/blob/master/modules/bot/src/main/BotLimit.scala . The generic responses did not identify the triggering limit or reset time.
