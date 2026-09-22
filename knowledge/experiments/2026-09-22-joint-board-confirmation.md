# Joint board-loss confirmation benchmark

Completed 4,000 fresh games on workspace2 H100s, following the inconclusive
[400-game joint-control comparison](2026-09-21-joint-board-training.md).
The joint model scores **50.5%**, with a **95% interval of 49.1–51.94%**.
The larger sample does not establish a playing-strength benefit from board loss.

The fixed comparison is the existing joint board-supervised checkpoint versus
the matched next-move-only continuation. No additional training or checkpoint
selection occurs. Every worker verifies both checkpoint hashes against the
original experiment before playing.

## Protocol fixed before results

- Exactly 4,000 additional games, 2,000 per color, standard starting position.
- Same runtime settings as the pilot: temperature 0.8, legal masking enabled,
  EOS suppressed, KV cache, 200-ply cap, unfinished capped games scored as draws.
- Eighty shards of 50 games, at most four H100 workers concurrently.
- Game offsets 400 to 4,350 in increments of 50; seed `20260922 + offset`.
  Game numbers and seeds do not overlap the earlier 400-game comparison.
- Finish all 4,000 games without changing the stopping rule based on interim scores.
- Primary result uses only the 4,000 fresh games. Any pooled 4,400-game result
  is descriptive and includes the exploratory sample that motivated this run.
- Score is `(wins + 0.5 * draws) / games`. The 95% percentile bootstrap interval
  uses 20,000 resamples within each color, preserving the fixed color balance.
- Replay every game independently to verify legality, result, termination,
  move count, identity, color, and benchmark settings. Count capped games separately.

The uncertainty interval measures game sampling uncertainty for this checkpoint
pair and runtime configuration. It does not measure variability across training
seeds or performance against a broader opponent population.

## Results

| Metric | Joint model versus next-move-only control |
|---|---:|
| Fresh games | 4,000 |
| Wins / draws / losses | 1,676 / 688 / 1,636 |
| Score | 50.50% |
| 95% color-stratified bootstrap interval | 49.10–51.94% |
| White score, 2,000 games | 52.35% |
| Black score, 2,000 games | 48.65% |
| Games reaching the 200-ply cap | 35 (0.875%) |
| Mean game length | 101.089 plies |

All 4,000 games replay legally. Results, terminations, lengths, unique IDs,
balanced colors, fresh seeds, fixed settings, and original checkpoint hashes
passed verification. Terminations: 3,312 checkmates, 242 stalemates, 301 fivefold
repetitions, 110 insufficient-material draws, and 35 move-limit draws.

If every capped game were counted as a joint loss or every one as a joint win,
the score would range from 50.06% to 50.94%. Capped games are too few to explain
the lack of a demonstrated gain. The descriptive pooled 4,400-game result is
1,854 wins, 750 draws, 1,796 losses, a 50.66% score. The fresh sample remains
the primary result.

The earlier 52.25% lead was not reproduced. This does not prove exact equality:
small advantages or disadvantages remain compatible with the interval. It does
mean this experiment provides no convincing basis to scale board-supervised
training for playing strength. The earlier board-readout improvement remains
valid, but improved reconstruction has not translated into a demonstrated policy
gain over the matched control.

Recommendation: retain the board head as a diagnostic, and investigate a broad
held-out mate-threat/tactical evaluation before choosing the next training
objective. Do not train specifically on the single traced `36...g5` failure.

App: https://modal.com/apps/workspace2/main/ap-3KC8TIZiLy7pTZ35LlWkKI
Summed worker evaluation time: 1,632 seconds (27.2 H100 worker-minutes), excluding
container startup and orchestration. Python compilation and whitespace checks
passed.

## Reproduction and results

```sh
MODAL_PROFILE=second modal run --detach modal_board_benchmark.py --confirmation
```

- Local results: `benchmark/results/joint-board-confirmation-20260922/result.json`.
- Local audit: `benchmark/results/joint-board-confirmation-20260922/audit.json`.
- Remote receipts and game logs: volume `nanodanya-data`, directory
  `/data/experiments/involved-pieces-20260921/benchmarks/joint-board/`, filenames
  `confirmation-joint-vs-control-0400` through `confirmation-joint-vs-control-4350`.
- Joint checkpoint: `/data/experiments/involved-pieces-20260921/joint-board/final.pt`.
- Control checkpoint: `/data/experiments/involved-pieces-20260921/joint-board-control/final.pt`.

The runner reuses completed receipts only when requested settings and hashes
match. No deployment, commits, or pushes are part of this experiment.
