# Joint board-supervised continuation

Completed on workspace2 H100, 2026-09-21. Joint training improves board readout and
scores 56.25% against the original champion in this benchmark. Its 52.25% score
against matched next-move-only continuation is inconclusive. The traced game's
fatal `36...g5` remains the greedy choice. Keep this as an experimental candidate;
the added chess-strength benefit of board loss is not yet established.

## Protocol

Both arms initialize from `all-pieces/final.pt`: the original champion transformer
plus the fitted head from [all-living-piece supervision](2026-09-21-all-living-piece-head.md).
The transformer is unfrozen for both arms.

- Control: next-move cross-entropy only; fitted piece head frozen.
- Joint: next-move CE + `0.1 * all-living location CE` + `0.1 * collision loss`.
  Both transformer and piece head update.
- Same 1,000 steps, sampled batches, seed 42, fresh optimizer, policy LR `3e-6`,
  head LR `1e-3`, batch 32, sequence 512, and existing 100k training games.
- Compare final step 1,000 checkpoints, independent of validation selection.
- Same ten fixed validation batches from the 5k validation games. These contain
  145,624 scored positions and 3,431,724 living-piece instances; sampled windows
  may repeat positions. Missing game prefixes receive no board supervision.
- Accuracy scores living-piece identities, with no empty-square credit. Truth
  still determines which identities are alive. Capture detection is not measured.
- Training app: https://modal.com/apps/workspace2/main/ap-YFHyfnBELDmyvhbomjyFGD

## Training results

| Metric | Champion + fitted head | Next-move-only control | Joint board loss |
|---|---:|---:|---:|
| Next-move CE | 1.405723 | 1.393091 | 1.393165 |
| Living-piece accuracy | 97.51% | 97.43% | 97.86% |
| Exactly correct living-piece placements | 95,321/145,624 (65.46%) | 94,162/145,624 (64.66%) | 100,504/145,624 (69.02%) |
| Positions with overlapping predictions | 26.44% | 26.59% | 24.04% |

Board supervision improves the fitted decoder's validation accuracy while
next-move CE is essentially tied with the control. This is not independently
proof that the transformer stores more board information: the joint head also
receives another 1,000 supervised updates, whereas the control head is frozen.
The game comparison tests whether the extra board objective improves policy
behavior. Overlapping predictions remain common; the penalty is not a hard
one-piece-per-square constraint.

Remote training function times were 304 seconds control and 241 seconds joint,
including compilation, validation, and checkpoint work.

## Chess benchmark

The canonical `benchmark/run.py` implementation evaluates original champion,
next-move-only control, and joint model with identical settings:

- Raw legality and EOS propensity: 4,096 positions from the separate probe split.
- Legal greedy move quality: 1,024 positions, Stockfish depth 10, identical engine
  binary. Report median centipawn loss and 300+ cp mistake rate. The mean is
  dominated by the benchmark's 100,000 cp encoding of mate scores.
- Head-to-head: 400 games per pairing, 1,200 total, 200 White and 200 Black per
  pairing, temperature 0.8, legal masking, EOS suppressed, KV cache, 200-ply cap.
  Eight shards per pairing use seeds `20260921 + game_offset`, offsets 0 to 350
  in increments of 50. Games reaching the cap are scored as draws and counted
  separately. Score means `(wins + 0.5 * draws) / games`.
- Independent-game percentile bootstrap score intervals use 20,000 resamples.
  They measure game sampling uncertainty, not variation across training seeds.
- Joint versus control is the primary comparison for the effect of board loss.
  Champion comparisons also include the effect of extra next-move training.

Probe games were excluded from this continuation's training and checkpoint
selection. Position collection takes the first N eligible positions from at
most 200 probe games, rather than sampling evenly across all games. Positions
from the same game are correlated. This is not a broad strength or Elo estimate.

The 74-position GZrvqz3j trace is a diagnostic already examined in earlier
experiments. It is excluded from training, but should not be treated as a fresh
test of the research decisions made across this sequence of experiments.

## Benchmark results

Benchmark app: https://modal.com/apps/workspace2/main/ap-zzzlfuMFI5utLWawjH5vE8

| First model versus opponent | W / D / L | Score | 95% bootstrap interval | Capped games |
|---|---:|---:|---:|---:|
| Joint vs control | 178 / 62 / 160 | 52.25% | 47.75–56.75% | 3/400 |
| Joint vs champion | 189 / 72 / 139 | 56.25% | 51.88–60.62% | 5/400 |
| Control vs champion | 176 / 66 / 158 | 52.25% | 47.75–56.75% | 5/400 |

The direct joint-control interval includes 50%, so the data do not establish an
added strength benefit from board loss. The joint-champion result supports an
improvement under these game settings, but cannot isolate its cause. The fact
that one champion comparison clears 50% and the other does not is not itself
evidence that joint training is better than the control.

All 1,200 saved games replay legally; results, terminations, ply counts, unique
game IDs, and 200/200 colors per matchup were verified. Only 13 games hit the
200-ply cap. Counting every capped game as a loss or every one as a win yields
score bounds of 51.88–52.62% joint-control, 55.62–56.88% joint-champion, and
51.62–52.88% control-champion. Sampling uncertainty remains larger than this
cap sensitivity.

| Fixed-position metric | Champion | Control | Joint |
|---|---:|---:|---:|
| Genuine illegal raw top choices / 4,096 | 9 | 6 | 6 |
| Strictly illegal raw top choices, including under-disambiguation / 4,096 | 14 | 12 | 12 |
| Mean probability mass on legal moves | 98.029% | 98.219% | 98.219% |
| EOS top choices / 4,096 | 13 | 10 | 9 |
| 300+ cp mistakes / 1,024 | 31 (3.03%) | 26 (2.54%) | 27 (2.64%) |
| Median centipawn loss | 12 | 12 | 12 |
| Engine best-move agreement | 49.51% | 48.73% | 48.34% |

All 1,024 move-quality FENs and engine binary hashes match across arms; no
positions failed scoring. The fixed-position results do not show a clear joint
advantage over the control. Both continuations modestly improve legality and
large-mistake counts relative to the champion on this sample.

## Traced game

| Metric, all 74 positions | Champion + fitted head | Control | Joint |
|---|---:|---:|---:|
| Correct living-piece locations / 1,676 | 1,636 | 1,635 | 1,645 |
| Exactly correct living-piece placements / 74 | 47 | 46 | 51 |
| Positions with overlapping predictions / 74 | 19 | 19 | 18 |
| Involved-piece locations correct / 96 | 95 | 95 | 96 |
| Correct locations away from starting square / 866 | 833 | 832 | 841 |

Before `36...g5`, all three decoders still get 9/10 living-piece locations right.
The black rook is incorrectly predicted on c2 instead of a6, overlapping the
white rook. Joint training increases the probability on its correct square
from 18.44% to 28.88%, but c2 remains the argmax.

An exhaustive legal reply check identifies `g5` as the only move that permits
mate in one. It remains the legal greedy choice for all three policies. At
temperature 1, after legal masking and EOS suppression, its probability is
51.32% champion, 58.71% control, and 58.78% joint. These diagnostic probabilities
use temperature 1 rather than the head-to-head temperature 0.8. Board supervision
has not repaired this tactical failure.

## Saved results and reproduction

Volume `nanodanya-data`, root `/data/experiments/involved-pieces-20260921`:

- `joint-board-control/final.pt` and `joint-board/final.pt`.
- Training metrics and checkpoints under each arm's directory.
- Benchmark requests, checkpoint hashes, game logs, and trace under
  `benchmarks/joint-board/`.
- Local training result: `data/probes/involved-pieces-20260921/joint-board-training.json`.
- Local benchmark results: `benchmark/results/joint-board-20260921/result.json`.
- Local replay audit and aggregates: `benchmark/results/joint-board-20260921/audit.json`.

Final checkpoint SHA256:

- Champion: `7e72148ed260564d671a67f5e0bc9bc4d82fd11695a34d9f4708858033955b0e`.
- Control: `e2298703c6719cf9e7709af1b9c0cd0ea602ef991086042eebbb5404f5e2ba10`.
- Joint: `e231e07541dc828fbfc751565e6c1b112fc14d178a7062653b2f26b9b6baa454`.

```sh
MODAL_PROFILE=second modal run --detach modal_involved_pilot.py --action joint-board
MODAL_PROFILE=second modal run --detach modal_board_benchmark.py
```

Training checkpoints resume within their existing directories. Benchmark receipts
are reused only when requested arguments and checkpoint hashes match.

Validation: 28 relevant tests passed; Python compilation and whitespace checks
passed. No deployment, commits, or pushes.
