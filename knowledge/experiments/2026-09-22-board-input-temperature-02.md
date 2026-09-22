# Board-input comparison at temperature 0.2

Running 4,000 fresh games on workspace2 H100s, comparing the existing actual-board
model against its constant-board control at temperature **0.2 for both models**.
No further training. This follows the
[temperature-0.8 comparison](2026-09-22-oracle-board-input.md), which scored
50.35% for actual-board inputs.

The user selected 0.2 based on the earlier
[champion versus SF1500 sweep](2026-09-21-champion-temperature-sweep.md).
That sweep found substantially stronger champion play at lower temperatures.
This test asks whether the board input adds strength when both models use 0.2;
it does not measure either model's absolute improvement against a fixed opponent.

## Fixed protocol

- Exactly 4,000 games, 2,000 White and 2,000 Black; no interim stopping.
- Same two final checkpoints from the 1,000-step board-input training experiment.
  Each worker verifies their original SHA256 hashes before play.
- Temperature 0.2, legal masking, EOS suppressed, KV cache, standard starting
  position, 200-ply cap scored as a draw. All settings except temperature and
  fresh random seeds match the preceding comparison.
- Eighty 50-game shards, up to four H100 workers concurrently. Offsets 4,000 to
  7,950, seed `20270000 + offset`, disjoint from the prior sample.
- Primary result is the actual-board model's score `(wins + 0.5 * draws) / games`
  at 0.2. Do not pool games across temperatures.
- 95% percentile bootstrap interval, 20,000 resamples within each color.
  The change from 0.8 uses independent resampling of both samples; this concerns
  the relative matchup advantage, not absolute playing strength.
- Replay all games and verify legal moves, results, terminations, counts, colors,
  unique IDs, new seeds, settings, and unchanged checkpoint hashes. Count move-limit
  draws and distinct full move sequences separately.
- Recompute `g5` probability from the same critical position at 0.2, 0.8, and 1.0
  for the actual-board model, control, and champion. Positive temperature scaling
  preserves the greedy top choice; it changes sampled move probabilities.

Raw legality and greedy engine move-quality metrics are unchanged by this
temperature setting, so they are not rerun. No champion head-to-head games are
included in this matched board-input comparison.

## Outputs

App: https://modal.com/apps/workspace2/main/ap-PhMoMIvcNxVdhxa6oIYtMg

```sh
MODAL_PROFILE=second modal run --detach modal_board_input.py --temperature-check --temperature 0.2
```

- Local results and audit: `benchmark/results/board-input-20260922-t0.2/`.
- Remote games: `nanodanya-data`, directory
  `/data/experiments/board-input-20260922/benchmarks/`, receipts/logs named
  `t0.2-oracle-vs-control-4000` through `t0.2-oracle-vs-control-7950`.
- Remote diagnostic: `/data/experiments/board-input-20260922/temperature-t0.2.json`.
- Checkpoint hashes are the same as in the preceding experiment report.

No deployment, commits, or pushes.
