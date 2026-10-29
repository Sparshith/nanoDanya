# Full Leela coverage, mixed L16

Status: paused at the user's request on October 28 for checkpoint strength testing.
Training app stopped at 18:39 IST. The latest durable checkpoint is step 310,000;
the last logged update was 319,000. About 9k unsaved updates would be replayed on
resume. User authorized training on all existing Leela training data with a new
$100 total compute cap, then explicitly requested this pause. Leave training paused.

Checkpoint audit: 4,344,771,004 / 6,841,912,852 Leela targets (63.50%), cursor
9,920,000 / 15,621,668 windows. Model, optimizer and RNG state preserved. Latest
validation move CE is 1.43645 (human 1.33511, Leela 1.53780). Frozen evaluation
copy and 2,600-game fresh comparison against the successful mixed 200k model are
under `output/leela-full-20261028/pause-eval/`. Evaluation and independent legal
replay audit completed; no strength improvement at this checkpoint.

## Paused-checkpoint results

| Metric | Previous mixed 200k | Full-coverage recipe at 310k |
| --- | ---: | ---: |
| Internal BayesElo in this fresh pool | 1811 ±11 | 1796 ±11 |
| Score vs SF1500, 400 games each | 85.625% | 83.625% |
| Score vs SF2000, 400 games each | 78.000% | 77.375% |

Direct H2H: new checkpoint 277W/394D/329L, 47.4% across 1,000 games.
Color-stratified bootstrap 95% score interval [45.0%, 49.8%]. Logistic
score-equivalent Elo is -18.1, interval [-34.9, -1.4]. Joint four-player BayesElo
fit gives a -15-point difference. The individual ±11 BayesElo intervals condition
on opponent rating estimates and are not a confidence interval for the difference.
SF1500 is anchored at 1500 by convention; ratings are internal masked ratings,
not calibrated human or Lichess Elo. Do not compare absolute ratings across pools.

Both models had 6/4,096 nonlegal raw top choices on the fixed-position probe;
real illegal moves excluding under-disambiguation were 4 new versus 3 old. Early
EOS top choices were 4 versus 8. These counts do not establish unmasked game strength.
The harness scored capped games and one old-model SF2000 opponent-vocabulary
ending as draws; H2H had five capped games and no vocabulary endings.

All 2,600 saved games independently replayed legally; model identities, settings,
outcomes, group totals and rating-game counts checked. Final evidence:
`output/leela-full-20261028/pause-eval/audit.json`, `comparison.pgn`,
`bayeselo-output.txt`, and `records/benchmark/`.

Training and both evaluation-launch apps are stopped. Approximately $47.99 in
resource-time estimates through the pause/test, before unmeasured startup and
minor controller overhead, within the original $100 cap. The GPU evaluation cost
approximately $0.49. The initial CPU-only controller launch failed before GPU work
and was stopped; its overhead is covered by the reserved margin. No promotion or
resume. This result concerns the 63.5%-covered checkpoint, not a completed pass.

The earlier mixed run sampled windows with replacement: 2.80B supervised Leela
tokens at 200k updates and 4.20B at 300k, against a 6.894B-token training corpus.
These were exposure counts, not unique coverage. The extension also retained the
terminal 3e-5 learning rate. This run tests a complete pass with a schedule planned
for that pass; it is not an isolated data-size ablation.

## Protocol

- Fresh L16, width 1024, eight heads, 512-token inputs; same initialization seed
  and architecture as the successful mixed 200k model.
- All 52,146,724 Leela training games. Validation remains held out. Shuffle a
  verified partition without replacement; supervise every move and EOS exactly
  once. BOS is context, not a prediction target.
- Audit: 6,894,059,576 corpus tokens, 6,841,912,852 supervised targets,
  15,621,668 packed windows, 488,178 updates. Zero excluded games. The existing
  cleaned training corpus contains no games exceeding 512 input tokens.
- Pack adjacent complete games into windows. Games exceeding the context limit
  retain every target through overlapping windows with up to 256 preceding context
  tokens on continuation segments. Mask repeated context targets; exclude no games.
- Existing human random-window sampler, 32 windows per domain per update,
  equally weighted domain-normalized cross entropy. Human windows retain the
  existing missing-history and BOS masks.
- AdamW, BF16, cosine 3e-4 to 3e-5, 2k warmup. Update count comes from the full
  coverage index. Save sampler cursor, optimizer and RNG states for recovery.
- The final checkpoint is the primary evaluation artifact. A validation-best
  checkpoint is retained separately and cannot substitute for full coverage.
- Automatically evaluate 1,000 masked H2H games against the successful mixed
  200k L16, plus 400 games per model against each of SF1500/SF2000. Fixed-position
  raw legality and EOS probes accompany the 2,600 games. Engine labels are settings,
  not human ratings; masked strength does not establish unmasked reliability.

## Budget and verification

Use the workspace2 Modal workspace (`second` profile). Full timeout reservations total
$97.463354, including a $5 startup/overhead reserve. Training has a 72,000-second
external timeout and persistent internal deadline; preparation, preflight and
evaluation have separate caps. GPU preflight must project completion with 10%
margin plus 1,200 seconds overhead inside the training allowance. No automatic
retries, promotion, commits or pushes.

Ten local tests passed, including exact coverage, partial batches, long games,
identical interrupted/uninterrupted training and failure to label a budget-stopped
checkpoint complete. An additional 100 randomized coverage cases passed.

H100 preflight: 0.11655 seconds/update, 56,898 seconds (15.8h) projected training;
63,788 seconds (17.7h) with 10% margin and 1,200 seconds overhead. The 64 temporary
updates were discarded. Initial weights match the successful mixed 200k run.

Local receipts: `output/leela-full-20261028/`.
Remote receipts: `/data/experiments/leela-full-20261028/` on `nanodanya-data`.
Checkpoints: `/data/checkpoints/plain/leela-full-20261028/`.
Training app: https://modal.com/apps/workspace2/main/ap-dTLXMYpBm0epu2hkCqOsGq
Pipeline call: `fc-01M6059KA40XZ80B3WNEXVBW0H`.
Runner: `scratch/modal_leela_full.py`; trainer: `scratch/leela_full_train.py`;
sampler: `training/coverage_batches.py`.
