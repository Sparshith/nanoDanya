# Human-only L16 with the corrected recipe

Status: launched October 30. Training/evaluation results pending. User authorized
8 H100s and an $80 total cap, including startup, retries, CPU, memory and evaluation.

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
