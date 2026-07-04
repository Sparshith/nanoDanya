# Training history and recipes (Dec 2025 - May 2026)

Distilled from the old TRAINING_EXPERIMENT_LOG.md (deleted 2026-07-04). Commands
reference pre-cleanup script names and dataset paths (train_weighted.py etc. are in
git history; datasets moved to `datasets/...` on the volume). Conclusions and H2H
numbers here were superseded by the May scale matrix; see STATE.md and
experiments/. What this file preserves is the recipes and dataset facts.

## Timeline

| When | Line | Idea | Outcome |
| --- | --- | --- | --- |
| Dec 2025 - Jan 2026 | baseline | plain next-token LM on 500k games | works; project thesis established |
| Feb - Mar 2026 | weighted | upweight puzzle solution moves (5x) | legality regressed badly, no strength gain |
| Mar 2026 | eval-head ft / scratch / combined | eval supervision variants | representation-interesting, never strength-competitive |
| Apr 2026 | plain/puzzles-5m, plain/puzzles-500k | scale + source controls | gain was scale, not source or weighting |
| Apr - May 2026 | plain/games-3m | scale actual games | confirmed scale lever (3m rung later deleted) |
| May 2026 | scale H2H suite | 200-game Modal matrix | story settles: plain CE + scale |
| Jun 2026 | 15M models | L12 + L16 on 15M games | champion; see STATE.md |

## Recipes and dataset facts

**Baseline (plain/games-500k)**: L12/H6/E768, AdamW lr 1e-4, batch 32, 50k iters,
plain CE. Early result 67W/3L/30D vs SF500.

**plain/puzzles-500k control**: built by `data_prep/build_weighted_data.py`,
trained with `UNIFORM_WEIGHTS=1`, batch 64, early stopping on uniform val loss
(patience 20, min steps 3000, min delta 0.001; best hit at step 9300). Dataset:
500k games, 40.8M tokens (36.8M train / 4.1M val).

**plain/games-3m**: recovered January Lichess shards, 3,092,525 games, 215.7M train
/ 23.9M val tokens. Plain CE, same backbone. (Model deleted 2026-07-04.)

**plain/puzzles-5m**: same builder, full corpus: 4,925,800 games, 402.3M tokens.
`UNIFORM_WEIGHTS=1`, batch 64, 100k iters.

**weighted ablation**: same corpus, solution moves weighted 5.0 via
`build_weighted_data.py --weight 5.0`, weighted CE
(`(per_token * w).sum() / w.sum()`). Negative result.

**eval dataset** (`datasets/eval/base`): built by `data_prep/prepare_eval_data.py` from
the 3.19M-game ndjson; strips `+`/`#`, prunes SAN tokens with freq < 5, context 512.
Weight rule: `w[t] = 1 + 4.0 * |eval[t] - eval[t-1]|` with evals clipped to
[-1500, 1500] and normalized to [-1, 1], times an endgame ramp after move 30.
Full-corpus stats: 3,190,825 games kept, 260.2M tokens (234.1M train / 26.0M val),
unpruned from-scratch vocab 12,678 (pruned+stripped vocab used in practice: 4,519).
First build attempt died on local disk writing `train_weights.bin`; build on Modal.

**eval-aware trainers** (all deleted; git history): eval-head fine-tune froze
embeddings + layers 0-8, trained 9-11 + lm_head + head, lambda_eval 1.0; from-scratch
used Muon (blocks) + AdamW (embeddings/heads), warmup + cosine, batch 64, grad
accum 4, lambda_eval up to 10.0 (which is how best-combined diverged from
best-move; keep selection metrics separate).

## Reproducibility notes

- Step counts are not comparable across trainers: tokens/optimizer-step ranged
  16,384 (batch 32) to 131,072 (batch 64 x accum 4), so "30k steps" of one line can
  cost more than "100k steps" of another. Compare token budgets, not steps.
- Track strength, raw legality, and eval quality as separate metrics per line;
  collapsing them into one scalar is how the scratch_v2 regression hid.
