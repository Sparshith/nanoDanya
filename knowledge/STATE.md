# nanoDanya State

Last updated: 2026-07-04

The project asks how much chess (legal continuation, tactics, termination) a pure
next-token model internalizes from data and objectives. Runtime scaffolds (legality
masks, EOS suppression, resign heuristics) keep games running but are never evidence
of learned capability; every result should say whether it measures the raw model or
model plus harness.

Model aliases come from `model_registry.py`. Public snapshot of this story:
https://www.sparshith.com/posts/nanodanya.html (numbers verified against
`benchmark/results/` on 2026-07-04).

## Current story

Scale under plain CE is the lever; data source and architecture are second-order
([scale matrix](experiments/2026-05-scale-matrix-h2h-and-stockfish.md)):

- **Source question settled.** At matched 5M scale, actual games vs puzzle-linked
  games is a tie (47.25 / 52.75 H2H). The old "puzzle data is better" story is dead;
  earlier puzzle-model wins were scale in disguise.
- **Champion: the 15M-game model.** The registry default is `plain/games-15m`
  (L16_H8_E1024 best, step 200k, `checkpoints/plain/games-15m/l16_best.pt`), which is
  also `modal_benchmark.py`'s default; the size-matched L12 is `plain/games-15m/l12`.
  The H2H matrix, Stockfish ladder, and blog numbers are from the L12: ~68-70% vs
  both 5M models, and 97% / 78.25% / 42.5% vs sf500/sf1000/sf1500. Architecture is a
  small effect on top: the L16 beats the L12 only 54.45% over 1000 games. The lichess
  bot serves the champion (redeployed 2026-07-04; `inference/serve.py`).
- Raw legality scales the same way: 3.9% real illegal (500k) to 0.44% (5M) to 0.24%
  (15M) ([raw legality](experiments/2026-05-raw-legality-by-scale.md)). The 15M model
  is at 99.71% raw-legal, better than every frontier API model tested on the same
  positions; best API model was gemini-3.1-flash-lite at 98.95%
  ([API comparison](experiments/2026-05-27-api-model-legality.md)).
- Errors, both nanoDanya's and the API models', concentrate in middlegame and
  endgame; openings are near-perfect for everyone.

Settled negative results:

- `weighted/puzzles-5m`: tactical upweighting hurt raw legality, didn't explain gains.
  Checkpoint, registry entry, and `training/train_weighted.py` deleted 2026-07-04
  (plain puzzle controls retrain via `training/train.py`, which ignores weight bins).
- King-safety SFT on `plain/games-5m`: improved the targeted in-check/late slices but
  slightly regressed overall legality; superseded by 15M pretrain
  ([king-safety SFT](experiments/2026-05-19-king-safety-sft.md)). Code and artifacts
  (`sft/`, `benchmark/king_safety_legality.py`, the modal_benchmark mode, volume
  `king_safety_sft/` data + checkpoints) deleted 2026-07-04; findings stand.
- 1-ply value-guided move selection: worse median cp loss and blunder rate than
  greedy ([eval head](experiments/2026-06-16-eval-head-boxed-classification.md)).
- Eval-supervised lines are not strength-competitive (`eval-aware/weighted-ft` lost
  to `plain/puzzles-5m` at 36%). The whole eval-aware line (scripts and checkpoints:
  v1, v2, weighted-ft, eval_boxes_test, ab_test) was deleted 2026-07-04 for a
  rewrite; the findings stand, the artifacts are gone.

What does work besides scale:

- **Position judging.** Boxed win-prob classification (64 bins, cross-entropy) fixed
  the eval-head mean collapse: corr 0.83 with Stockfish win%. The model judges well;
  using that judgment for move selection is the unsolved part.
- **Puzzle-solve eval.** Rating-binned solve rates decline monotonically (full-solve
  77% at 600-700 down to 2.5% at 2700-2800), so the metric discriminates; hard
  puzzles fail on the deep line, not move one
  ([puzzle eval](experiments/2026-06-16-puzzle-solve-eval.md)). GO for the
  tool-assisted phase.

Linear probes support the story from inside: board occupancy and legal-move sets are
linearly readable from hidden states and improve with data
([probes](experiments/2026-05-01-linear-probes.md)).

## Measurement conventions

- **Raw top-1 legality**: is the highest-logit token legal before any mask. Report
  pre-`<eos>` and excluding `<bos>`. `raw_top1_illegal_rate` is strict SAN-token
  illegality; `raw_top1_real_illegal_rate` excludes under-disambiguated SAN (e.g.
  `Ng4` when SAN requires `Nfg4`), a notation failure, not a board-state one.
- **Masked/game results** measure model plus harness, never raw legality.
- H2H at temperature 0 is degenerate (deterministic games collapse); use temp 0.8.
- sf500 is saturated for 5M+ models; discriminate at sf1000/sf1500.
- Benchmark files are only comparable when protocol matches. Checkpoints with
  identical arch can be incompatible if their `meta.pkl` tokenizer contracts differ.
  Checkpoint-selection metric matters: `*_best_move.pt` is often more honest than
  best-combined-objective for legality questions.

## Open questions

- Game endings: the model still relies on the max-move guardrail; `<eos>` behavior
  should be learned, not suppressed around.
- Middlegame tactics and conversions are now the real gap (blog framing: the
  problems left are chess problems, not legality problems). sf1500 score 42.5% is
  the number to move.
- Tool-assisted / action-value direction (FEN-snapshot rebuild): flagged as the next
  big phase after the puzzle-eval GO, not started.
- Judge-without-selection uses for the eval head (blunder-check veto, resign/draw
  decisions) are unexplored.
- Eval-aware training is being rewritten from scratch (old scripts and checkpoints
  deleted 2026-07-04). Building blocks that survive in `training/common.py`:
  `forward_hidden`/`move_logits` for backbone access; the boxed win-prob recipe
  (64 bins, win_k=0.00368208) is in the experiment note and git history.
