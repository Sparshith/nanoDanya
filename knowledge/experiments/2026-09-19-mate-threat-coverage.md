# More distinct mate-threat positions at a fixed update budget

Status: complete. Small fresh-test probability improvement; no established game-strength gain.

User authorized the follow-up and requested a comparison to DeepMind Searchless
Chess. No production change.

## Frozen follow-up

Reuse the existing 1,000-position targeted checkpoint as the narrow control.
Expand to all **2,208** distinct eligible source games in the old training split,
one position per game in the same seeded hash order; the original 1,000 are a
prefix of this larger pool. The 231,377-position ordinary training source and
all rewards stay the same. Start again from the original pre-RL checkpoint,
not the already fine-tuned narrow model. Both models receive 300 updates of
64 positions (32 uniform + 32 targeted), same LR 3e-6, KL 0.05, temperature 0.8,
full-model AdamW and seed. The intervention is coverage versus repetition.
The expanded run uses `training/train_mate_threats.py`, promoted from scratch
for reuse; mate-label helpers are now `data_prep/prepare_mate_threats.py`.
Sixteen focused tests still pass after the move and dataset-path parameterization.

Existing validation/test probes are development-only, already inspected. Do not
select a checkpoint or change training from their scores. Primary evaluation:
2,000 freshly generated original-champion-v-SF1500 games, seed 27419000,
balanced colors, temperature 0.8, engine 20ms, 200-ply cap. Up to 32 random own
positions/game from ply 16, excluding all boards appearing in any old archive
split. Exact one-reply mate labels; grade candidate threats and a fixed ordinary
sample with the same 50k-node MultiPV engine recipe, exact fatal-move overrides.
Select up to 500 eligible threats and 1,000 ordinary positions by fixed hash,
one per source game, before model inference. Minimum 100 threats. Reuse engine
cp >= -200 eligibility. No synthetic positions or probability-based selection.

Evaluate original, narrow and expanded models together on the fresh positions.
Primary metric: expanded-minus-narrow mate probability with paired game-bootstrap
95% interval. Also report greedy errors, ordinary rewards/blunders, and raw
legality. Finish with 400 fresh expanded-versus-narrow games, seed 27519000.
One training seed remains a limitation; old checkpoint reuse means this is a
matched-recipe comparison, not a bitwise-controlled deterministic replication.

Dataset preparation app: ap-BxhGwr1K3eAI1PVWAzG6BM.
Expanded training: ap-obgQsxWoBWErPzwftuNqY0.
Fresh game collection: ap-SvuOeWzJFy3Pb895csZazH.
Dataset: /data/datasets/own-games/mate-coverage-20260919.
Expanded checkpoint: /data/checkpoints/grpo/mate-coverage-20260919/expanded/final.pt.
Fresh source: /data/datasets/own-games/mate-coverage-fresh-source-20260919.jsonl.
Fresh test: /data/datasets/own-games/mate-coverage-fresh-test-20260919.

## Collection and execution audit

Both 2,000-game collection and 400-game H2H replay audits pass, with 1,000/1,000
and 200/200 colors respectively, unique game IDs and complete game numbers.
Fresh source champion score versus SF1500: 884W/215D/901L, 49.575%, 73 unresolved
200-ply caps. Source SHA256 34b9f3295eee27a3c78fe1a727097b1c10cf68144fb773d4dc1349518e5de55a.
After excluding every old archive board, 57,287 candidate positions remain.
Exact labeling and 50k-node MultiPV grading cover 4,025 threat/ordinary records;
984 eligible threat positions supply the frozen 500-position, 500-game primary
test. Ordinary test contains 1,000 positions, one per game. Labels/selection
were frozen before any candidate inference.

Expanded training completed at step 300. Training mate probability 6.038% ->
1.331%; development validation 5.730% -> 4.779%. The reused old test is strictly
development-only (mate probability 3.559% expanded versus previously measured
3.749% narrow). Original-model development probabilities differ slightly across
runs (validation baseline 5.730% vs previous 5.746%), consistent with numerical
execution variability; do not claim bitwise identity. Fresh evaluation loads all
three checkpoints in one process with identical inputs and batch order.

Fresh expanded-versus-narrow H2H: **151W / 93D / 156L**, **49.375%** score,
approximate independent-game 95% [45.077%, 53.673%], Elo -4.3 [-34.3, +25.6].
Four caps counted as draws; assigning all as losses/wins bounds score at
48.875% / 49.875%. No established strength gain or equivalence.
H2H app ap-WtujFyObbehOVdwDaMfmjN; labeling app ap-xg4zIglSYmXJzk5f3TQBJW.
Artifacts: data/traces/grpo/mate-coverage-20260919/.

## Fresh held-out result

All three checkpoints were evaluated together on the frozen new test:

| Metric | Original | Narrow (1,000) | Expanded (2,208) |
| --- | ---: | ---: | ---: |
| Mate-in-one blunder probability, 500 threats | 6.846% | 5.176% | 4.911% |
| Greedy mate-in-one blunders, 500 threats | 26 | 16 | 18 |
| Ordinary expected reward, 1,000 positions | 0.63121 | 0.63475 | 0.63485 |
| Ordinary blunder probability | 16.886% | 15.892% | 15.819% |
| Ordinary raw top-choice illegal moves | 30 | 33 | 35 |

Primary expanded-minus-narrow mate probability: **-0.265 percentage points**,
paired game-bootstrap 95% **[-0.478, -0.064]**, 10,000 resamples. This supports
a small probability improvement on this fresh source distribution, conditional
on these fitted checkpoints. Greedy error difference +0.4 pp has interval
[-0.6, +1.4], so neither a greedy improvement nor regression is established.
Ordinary reward difference +0.000096 has interval [-0.000175, +0.000361].

More distinct threats produce modest additional transfer at the same update
budget, but this does not establish a fix for decisive move selection or an
Elo gain. The 400-game H2H remains 49.375%; do not promote the checkpoint.
One training seed and one fresh source recipe limit generalization. These
results do not rule out gains from substantially more data or different
supervision; they do not justify assuming that scaling this recipe solves it.
Next diagnostic recommendation: a controlled explicit-board/value-supervision
comparison, retaining full-game evaluation alongside tactical metrics. That
comparison has not been run or deployed.

Inference app: ap-CmyYGyyewMRlerJdGtlUQW. Reports: fresh_test.json, summary.json,
fresh_test_manifest.json and h2h_audit.json under the local artifact directory.
