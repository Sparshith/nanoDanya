# Eval Head: Boxed Win-Prob Classification

Date: 2026-06-16. Backfilled from session notes during the 2026-07-04 restructure.

## Problem

The original scalar eval head (centipawns clipped to +/-1500, scaled to [-1,1], MSE)
collapsed to predicting the mean, corr ~0 with Stockfish. Root cause was the target
representation, not the backbone.

## Fix

ChessBench-style classification (DeepMind used 128 bins; their ablation plateaus
after ~32):

- centipawns to win% via logistic `1/(1+exp(-0.00368208*cp))`
- bin win% into 64 uniform boxes
- `Linear(n_embd, n_bins)` head, cross-entropy, `ignore_index` for NaN evals
- inference: expected win% = softmax @ bin_centers

Fine-tuned `eval-aware/v2` (top 3 layers + heads unfrozen) on `datasets/eval/base`
(vocab 4519). Script: `training/train_eval_boxes.py`.

Note (2026-07-04): the script, the eval-aware checkpoints, and the `eval_boxes_test`
volume dir were deleted for a from-scratch rewrite. The recipe above and git history
are what survives; `datasets/eval/base` was kept.

## Result

Corr 0.83 with Stockfish win%, win%-MAE ~0.087, reached by step 200. The model is a
good position judge.

## Value-guided move selection: negative

Tested 1-ply lookahead: policy proposes top-k legal moves, each scored one ply ahead
by the win% head, pick best. Implemented as `--selection value-guided --top-k` in
`benchmark/run.py` with `modal_benchmark.py` pass-through.

- H2H value-guided vs greedy, same checkpoint: lost 39.5 vs 60.5. Confounded: greedy
  sampled at temp 0.8 while value-guided played deterministic argmax.
- vs Stockfish-500: 97 vs 96, a tie; SF500 too weak to discriminate.

## Move-quality follow-up (2026-06-25)

Stockfish-scored per-move comparison on 600 `datasets/eval/base` val positions, same
checkpoint, greedy vs value-guided selection
(`benchmark/results/modal_move_quality_{greedy,value-guided}_20260625_*.jsonl`):

| Metric | greedy | value-guided |
|---|---:|---:|
| median cp loss | 24 | 31 |
| blunder rate (>=300cp) | 11.8% | 16.0% |
| best-move match | 40.8% | 38.3% |
| avg cp loss | 1427 | 1171 |

Value-guided is worse on median loss, blunders, and best-move match; only the
outlier-dominated average favors it. This confirms the wash-to-negative verdict on
1-ply value guidance, independent of the earlier H2H determinism confound.

Related: `eval-aware/weighted-ft` lost to `plain/puzzles-5m` 36% over 50 games
(2026-05-19, `benchmark/results/eval_weighted_ft_vs_puzzle_5m_50.jsonl`), so no
eval-supervised line is competitive with the plain champions on strength.

Verdict: judging works, 1-ply value-greedy on a state-value head is a
wash-to-negative (horizon effect). Untested: search deeper than 1 ply, or using the
judge only for blunder-checking rather than move selection.
