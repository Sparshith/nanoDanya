# Log

Append-only, newest last. Big runs get their own file in `experiments/`; the current
picture lives in `STATE.md`.

## 2026-05-01 initial wiki scaffold

Created the original `knowledge/` wiki from `AGENTS.md`, `ARCHITECTURE.md`,
`TRAINING_EXPERIMENT_LOG.md`, and `model_registry.py`. Filed the linear probe results.

## 2026-05-04 retired MODEL_FAMILIES.md

Moved the durable content of root `MODEL_FAMILIES.md` into the wiki; `model_registry.py`
stays the source of truth for aliases and checkpoint paths.

## 2026-05-11 scale H2H suite changed the story

Ran the four-edge games-vs-puzzles H2H suite on Modal. Reframed the narrative from
"puzzle weighting helped" to "plain CE plus scale helped". `plain/puzzles-5m` stays
champion; `plain/games-5m` vs `plain/puzzles-5m` flagged as the missing clean test.
See `experiments/2026-05-11-games-vs-puzzles-scale-h2h.md`.

## 2026-07-04 flattened the wiki

Restructured `knowledge/` from the 9-directory wiki to STATE.md + log.md + experiments/.
Backfilled work the wiki had missed: raw legality by scale (5M/15M models),
eval-head boxed classification (corr 0.83), and the rating-binned puzzle-solve eval
(go/no-go: GO). Dropped model-family stub pages (duplicated `model_registry.py`),
templates, and operations docs. Fixed stale references (deleted `training/train_eval.py`,
`game_runner.py`) and unified naming to registry aliases.

## 2026-07-04 ingested the May benchmark suite from benchmark/results/

The blog post was ahead of this folder; backfilled from artifacts (all numbers
re-extracted from JSONL summaries, all match the blog). New: the full 5x5 H2H matrix
and Stockfish ladder (champion is now the 15M-game model; 5M games vs 5M puzzles is
a tie, settling the source question), the API model legality comparison, and the
king-safety SFT results. Updated the raw-legality table with phase splits and the
L16-vs-L12 arch comparison, and added the 2026-06-25 move-quality run to the
eval-head note (value-guided selection confirmed negative). Rewrote STATE.md:
champion changed, both former open questions resolved, new open questions filed
(15M not in registry, endings/`<eos>`, middlegame tactics, tool-assisted phase).

## 2026-07-04 consolidated training/ and deleted the eval-aware line

Deduplicated the seven copy-drifted trainers into `training/common.py` (hardened
checkpoint resume with full config validation, shared `train_loop` with AMP, grad
accum, best/periodic checkpoints, early stopping). `train.py` and
`train_weighted.py` are now 66/82-line scripts holding only data and loss; both
smoke-tested on Modal including resume (loss trajectories match pre-refactor).
Deleted `train_eval.py` and `train_combined.py` (dead regression-head lines), then
the entire eval-aware branch for a rewrite: `train_from_scratch.py`,
`train_ab_eval.py`, `train_eval_boxes.py`, volume checkpoints
`checkpoints/eval-aware/` (v1, v2, weighted-ft), `eval_boxes_test/`, `ab_test/`,
and the three eval-aware registry entries. `datasets/eval/base` kept for the
rewrite. New env vars: `CKPT_DIR` picks the checkpoint directory (the old
per-dataset default paths predate the volume reorg); `WEIGHT_DECAY` and
`EVAL_BATCH_SIZE` removed (never used).

## 2026-07-04 deleted the king-safety SFT line

Continuing the "keep only the blog story" cleanup: removed `sft/` (data builder,
trainer, validator), `benchmark/king_safety_legality.py`, the king-safety mode and
functions in `modal_benchmark.py`, and the volume `king_safety_sft/` directory
(SFT data bins plus both checkpoints). Benchmark result JSONLs in
`benchmark/results/` are kept as evidence for the experiment note. The repo now
carries only the plain scale-story models plus the weighted ablation; eval-aware
and king-safety exist as findings in knowledge/ and recipes in git history.

## 2026-07-04 refresh pass after registry/cleanup changes

The 15M checkpoints are now registered: `plain/games-15m` (L16, marked champion,
also `modal_benchmark.py`'s default) and `plain/games-15m/l12`; checkpoints moved
from `/data/actual_15m/` to `checkpoints/plain/games-15m/`. Updated STATE.md
(champion bullet reconciled with the registry, registration open question closed,
noted the lichess bot still serves `plain/puzzles-5m`) and fixed now-stale "not in
registry" and old-path claims in the scale-matrix and raw-legality notes. Added
deletion pointers to the eval-head and king-safety notes so they don't reference
removed scripts as if present.

## 2026-07-04 dropped the 3m model and bootstrap notebooks

Deleted `checkpoints/plain/games-3m/` from the volume and its registry entry; the
scale story keeps 500k/5m/15m and the 3m rung was never in the blog numbers.
`datasets/games/3m` deleted from the volume too (raw sources remain). Also deleted the four February
bootstrap notebooks (DetermineElo, Inference, Preprocess, Train), superseded by
`training/`, `inference/`, and the prep scripts.

## 2026-07-04 lichess bot now serves the champion

Switched `inference/serve.py` from `plain/puzzles-5m` to `plain/games-15m`
(`checkpoints/plain/games-15m/l16_best.pt`) and redeployed the `nanodanya-chess`
Modal app. Verified via `/health` (reports the champion) and a live `/move` call
(legal reply in a Petrov). The bot app needed no change; it calls the endpoint.

## 2026-07-04 dropped the L8 checkpoint and the weighted line

Deleted `plain/games-500k/l8` (the old chess_min.pt) and `weighted/puzzles-5m`:
volume checkpoints, registry entries, and `training/train_weighted.py`. The
weighted ablation is a settled negative result; the plain puzzle controls it
also trained are reproducible with `training/train.py` (plain CE ignores the
weight bins in the puzzle datasets). `training/` is now common.py + train.py.
README_LAYOUT.md rewritten to match the volume (eval-aware and 3m references
were stale) and re-uploaded.

## 2026-07-04 retired the arena rig

The January `arena/` dir (untracked LLM full-game rig, RESULTS.md, stray .env) is
retired; the 9-0 result is preserved in
[experiments/2026-01-31-llm-arena.md](experiments/2026-01-31-llm-arena.md).
API-model comparison now happens only via the benchmark legality mode.

## 2026-07-04 folded the standalone local docs into knowledge/

ARCHITECTURE.md and TRAINING_EXPERIMENT_LOG.md are deleted; the parts knowledge/
didn't already cover moved into [architecture.md](architecture.md) (pipeline mental
model, token contract, which-knob-moves-which-metric, experiment checklist) and
[training-history.md](training-history.md) (timeline, recipes, dataset facts,
reproducibility notes). New [modal.md](modal.md) documents how we use Modal (apps,
volume layout, training/benchmark commands, env knobs, smoke-test pattern); the
local README_LAYOUT.md is deleted, its volume-side copy stays. knowledge/ is now
the only knowledge location.

## 2026-07-04 tracked the dataset prep scripts in data_prep/

Moved the four live dataset builders out of gitignored dirs into tracked
`data_prep/`: `prepare_actual_data.py`, `prepare_eval_data.py`,
`download_games.py`, `build_weighted_data.py`. Deleted the abandoned
board-token puzzle format scripts (download_puzzles, convert_puzzle,
generate_puzzle_vocab) and `puzzles/puzzle_training.md` after lifting its untried
eval-as-tokens idea into STATE.md open questions. `rc_extract_actual_games.py`
lives on RapidCanvas, not here; its filters are recorded in the modal.md lineage
section. Note: the extracted game shards went away with `archive/`, so rebuilding
a games dataset differently means re-extracting from the public Lichess dumps.
