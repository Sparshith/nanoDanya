# Board-input comparison at temperature 0.2

Completed 4,000 fresh games on workspace2 H100s, comparing the existing actual-board
model against its constant-board control at temperature **0.2 for both models**.
The actual-board model scores **46.54%** (95% interval **45.24–47.88%**), a clear
disadvantage in this matchup. Its relative score is lower than at 0.8 by 3.81
percentage points. Lower temperature does not rescue the board-input intervention.
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

## Results

| Both models' temperature | Actual-board wins / draws / losses | Score | 95% interval |
|---|---:|---:|---:|
| 0.8, previous sample | 1,666 / 696 / 1,638 | 50.35% | 48.95–51.76% |
| 0.2, fresh sample | 1,416 / 891 / 1,693 | 46.54% | 45.24–47.88% |

The change in relative score is **−3.81 points**, with an independent-sample
bootstrap 95% interval **−5.71 to −1.89 points**. This is a change in the
board-aware model's advantage over its control when both temperatures change;
it does not mean lowering temperature weakens either player against a fixed
external opponent. The champion's earlier SF1500 improvement is compatible
with this result.

The actual-board model scores 36.875% as White (515 wins, 445 draws, 1,040 losses)
and 56.20% as Black (901 wins, 446 draws, 653 losses). Both colors contribute
exactly 2,000 games. Lower-temperature play concentrates on fewer lines:
3,385 distinct full move sequences, versus 4,000 at 0.8. As a post-hoc diagnostic,
the most common full line ends in a Black checkmate and appears 169 times with
the oracle as White and 125 times with it as Black. These are independently
sampled games with unique IDs and fresh seeds, not duplicated worker results.
The results describe this starting-position matchup, not broad chess strength.

All 4,000 games replay legally. Terminations: 3,109 checkmates, 641 fivefold
repetitions, 161 stalemates, 80 insufficient-material draws, and 9 move-limit
draws. Counting all 9 capped games as oracle losses or as oracle wins gives
scores of 46.425–46.650%, so move-limit scoring cannot explain the disadvantage.
Counts, colors, IDs, seeds, checkpoint hashes, and runtime settings all passed
verification. Python compilation and whitespace checks passed.

## Traced failure at matching temperatures

All three models still rank `g5` first in the position before `36...g5`.
The same move permits `Rh2#`. The board-aware model receives correct piece
locations. The saved temperature-1 probabilities match the previous diagnostic.

| Model | P(g5), temperature 0.8 | P(g5), temperature 0.2 |
|---|---:|---:|
| Original champion | 55.29% | 76.53% |
| Constant-board control | 66.69% | 96.25% |
| Actual-board model | 70.51% | 98.30% |

Legal masking and EOS suppression are applied in both columns. In this specific
position, lower temperature makes the already-favored mistake more likely.
This does not contradict stronger average play by the champion at low temperature.

Recommendation: do not promote the actual-board checkpoint. It has no demonstrated
advantage at 0.8 and loses to its matched control at 0.2. Any next test of broad
strength should use varied starting positions or external opponents at matched
settings, rather than assume this one matchup establishes a universal ranking.

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
