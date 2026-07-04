# King-Safety SFT

Date: 2026-05-19 (run `created_at` in artifact). Backfilled 2026-07-04 from
`benchmark/results/`. Workflow added in commit 1e2abb7 (`sft/` scripts:
build/validate data, train, plus `benchmark/king_safety_legality.py`); all of it
deleted 2026-07-04 (commit 59d63ab) along with the volume `king_safety_sft/` data
and checkpoints. The result JSONLs in `benchmark/results/` remain the evidence.

## Setup

Fine-tuned `plain/games-5m` on a king-safety-focused SFT mix (70/30 per the
checkpoint name, `/data/king_safety_sft/chess_actual_5m_king_safety_sft_70_30_L12_H6_E768_best.pt`),
targeting the failure slices where legality errors concentrate: positions in check,
late positions, low-legal-move-count positions.

## Results

Targeted slices improved (`king_safety_legality_{plain_games_5m,sft_best}_4096.jsonl`,
4096 positions per slice-set):

| Slice | base exact illegal | SFT exact illegal |
|---|---:|---:|
| in_check | 52/2026 = 2.57% | 37/2026 = 1.83% |
| late_position | 76/2379 = 3.19% | 53/2379 = 2.23% |

But overall raw legality on the standard `actual_5m` val 4096 slightly regressed
(`legality_sft_best_actual_5m_4096.jsonl`): strict illegal 0.610% vs the base
model's 0.537%, with endgame worst hit (1.36% vs 1.02% illegal).

## Read

Targeted SFT does move the targeted failure modes, but at this scale it traded a
little general legality for it. Not adopted; the 15M-scale pretrain got to better
overall legality (0.293% strict) without targeted data. Worth revisiting only on
top of the current champion, measuring both slice and overall legality.
