# Human-only L16 with the corrected recipe

Status: completed October 30. Human-only corrected-recipe training recovered most
of the mixed model's measured strength gain. Both apps stopped, zero tasks.
Conservative accounting $77.48, including queue time and overhead allowances,
within the user-authorized $80 total cap. This is not a provider invoice.


## Completed results

Training reached 200,000 updates after resuming from 179,321 with the same world
size and restored optimizer/sampler state. Supervised 5,912,110,582 human targets.
Final held-out CE 1.305626; move-only CE 1.301837. Lowest validation CE 1.305109
occurred at 190k; game evaluation used the predetermined final 200k checkpoint.

| Model | Fresh internal masked BayesElo |
| --- | ---: |
| Old human L16 | 1818 (+14/-14) |
| Human-only, corrected recipe | 1949 (+15/-14) |
| Mixed human/Leela, corrected recipe | 1954 (+15/-15) |

All three were freshly evaluated in the same 3,600-game pool, SF1500 fixed to
1500. These ratings are not public/human Elo and are not comparable to older
pools. BayesElo intervals are conditional on opponent rating estimates.

Human-fixed versus old-human: 225W/73D/102L, 65.375% in 400 games, paired-opening
95% interval 61.375–69.375%; implied H2H Elo +110 [80,142]. Human-fixed versus
mixed: 153W/83D/164L, 48.625%, interval 44.5–52.75%; implied H2H Elo -10 [-38,19].
The mixed model scores 67.875% against old-human in the same fresh evaluation.

The corrected human-only training setup delivers a clear gain over the old
human model; an additional mixed-data advantage is not established here. This
supports recipe/training changes as a major explanation for the earlier gain.
It does not prove data equivalence: one seed, current-vs-historical implementation
changes, and about 2.7% more supervised targets than mixed-200k remain limitations.

Held-out 4,096-position diagnostics: raw move illegality counts 3 human-fixed,
8 old-human, 6 mixed. EOS top-1 counts are separately 12, 9, and 8. These are
fixed-position diagnostics, not raw-game reliability or forfeit measurements.

Remote replay checks passed all 3,600 games. Local reconciliation confirmed all
nine 400-game groups, unique per-pair game numbers, scores and rating game counts.
Receipts: `output/human-fixed-20261030/{summary.json,elo.jsonl,completion-audit.json}`.
The conservative $77.48 estimate even retains the first call's queue time at the
full GPU resource rate and adds $0.50 CPU and $2 overhead allowances.

The missing control for the successful mixed human/Leela L16 is a fresh human-only
model trained with the corrected recipe. This run uses the existing 15M human-game
corpus, the mixed run's seed and initial weights, L16/1024, context 512, global
batch 64, 200,000 updates, AdamW and cosine 3e-4 to 3e-5 with 2,000 warmup updates.
Random windows, missing-history masking, BOS masking and retained EOS match the
mixed recipe. All 64 windows now come from humans. This is not a coverage pass;
actual supervised-token counts will be reported.

Uses the current recipe trainer, including FP32 loss, combined model/loss compilation
and single-source DDP. These implementation changes and one-seed uncertainty remain
limitations of comparison with the historical mixed run. Primary evaluation uses
`final.pt`, not the checkpoint selected by human-only validation.

Remote CPU preflight: all 20 pretraining tests passed, including human-only random
sampling under two-worker DDP, equality with single-worker updates and exact CPU
resume. Human train/validation hashes, both control checkpoint hashes, model shapes,
the mixed checkpoint's 200k step and the fresh initial-weight hash passed.

Evaluation runs automatically after successful completion: 200 fixed openings,
both colors, all three model pairs and each model against SF1500/SF2000, 3,600
fresh games. Greedy legal-masked play, canonical histories, depth-1 Fairy-Stockfish,
10ms/1300cp adjudication, 400-ply cap. Paired-opening bootstrap intervals and legal
PGN replay checks. Raw held-out legality and termination are reported separately.
A local collector downloads results and fits a fresh BayesElo pool anchored at
SF1500; these internal ratings are not human Elo and are not pooled with historical
stochastic start-position matches.

Reservations: training $67.2284 (7,200-second external wall-clock cap), evaluation
$5.3459 (4,200 seconds), verification $0.0518, controller $0.1579, overhead $5.
Total $77.7840. Training stops early enough to save before its deadline. If it
cannot finish 200k within that reservation, preserve the checkpoint and report an
incomplete comparison. No automatic error retry or additional unbudgeted training.
GPU preemption restarts remain within the controller's deadline. Estimates are
rate-based upper reservations, not provider invoices.

App: https://modal.com/apps/minusx/main/ap-wVorBwKP4GZTgspDjRG49s

Local protocol, logs, receipts and eventual results: `output/human-fixed-20261030/`.
Remote experiment: `/data/experiments/human-fixed-20261030/` on `nanodanya-data`.
Checkpoint directory: `/data/checkpoints/plain/human-fixed-20261030/`.
Recipe: `training/recipes/l16-human-fixed-windows.json`.
Runner/evaluation/collector: `scratch/modal_human_fixed.py`,
`scratch/human_fixed_eval.py`, `scratch/collect_human_fixed.py`.
Source snapshots are preserved with the remote experiment. Profile `second`.
No promotion, commit, push, or resumption of the paused full-Leela run.

## Deadline adjustment at 119k updates

The 8-H100 allocation started at 16:18:35 IST, about 10.5 minutes after the
training call was queued. Training is healthy, but validation/checkpointing adds
roughly 5-6ms/update to the 29ms steady updates. The original controller counts
queue time against its wall-clock limit, so a bounded continuation was arranged
without disturbing the live job or changing its recipe.

The continuation waits for the original pipeline. If training stops short, it
resumes `latest.pt` with eight workers, then evaluates the fixed 200k endpoint.
Original training is conservatively reserved from its recorded allocation time
through the original deadline, plus 130 seconds startup/shutdown allowance.
The new total reservation is $79.7508: original training <=$62.5676,
continuation <=$9.3373 (1,000 seconds including queue/startup), evaluation
<=$5.3459, all CPU checks/controllers $0.50, other overhead $2.00. The user cap
remains $80. Queue reservation was released; no measured GPU execution was removed.
If the continuation cannot finish within its own deadline, it saves and stops.

Continuation app: https://modal.com/apps/minusx/main/ap-O2I2Tt17Y6q46gPIT5Q5KE
Runner: `scratch/modal_human_fixed_finish.py`; request and call receipts:
`output/human-fixed-20261030/finish-*.json`. Remote continuation ledger and source
snapshot: `/data/experiments/human-fixed-20261030/finish/`.
