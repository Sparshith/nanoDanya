# Full Leela coverage, mixed L16

Status: dataset audit passed on October 28; GPU preflight running. User authorized
training on all existing Leela training data with a new $100 total compute cap.
Results pending.

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

Local receipts: `output/leela-full-20261028/`.
Remote receipts: `/data/experiments/leela-full-20261028/` on `nanodanya-data`.
Checkpoints: `/data/checkpoints/plain/leela-full-20261028/`.
Runner: `scratch/modal_leela_full.py`; trainer: `scratch/leela_full_train.py`;
sampler: `training/coverage_batches.py`.
