# Move-involved piece pilot

Completed on workspace2, 2026-09-21. No clear representation improvement over the
matched next-move-only control. The critical missing-piece problem remains.

Follow-up: [faster head with frozen-policy warmup](2026-09-21-involved-piece-head-warmup.md)
finds both missing black pieces in the unchanged champion's hidden state.

## Run

- Modal profile `second`, workspace `workspace2`, H100.
- App: https://modal.com/apps/workspace2/main/ap-VbnmzCkL4Drx1pJ2z8hliW
- Initialization: existing `/data/checkpoints/plain/games-15m/l16_best.pt`,
  step 148,400, L16/H8/E1024, context 512, vocabulary 14,750.
- Initial checkpoint SHA256:
  `7e72148ed260564d671a67f5e0bc9bc4d82fd11695a34d9f4708858033955b0e`.
- This is the original 15M champion, not the own-game model previously inspected.
- Data: 100,000 unique games from the February 2025 portion of the existing
  combined SAN corpus. Token IDs converted to the checkpoint vocabulary.
  5,000 separate validation games and 200 additional probe games.
  No selected moves were unrepresentable. The traced game is absent.
- Both arms: 1,000 updates, batch 32, AdamW 3e-6, seed 42, identical sampled
  windows. Joint backbone/head training; involved-piece weight 0.1 versus 0.
- Only mover, captured piece, and castling rook locations receive auxiliary
  supervision. Targets are pre-move positions; future tokens select targets
  without entering their prediction. Incomplete histories receive no piece loss.
- Comparison uses fixed final step 1,000, not a checkpoint chosen on the trace.

## Independent representation comparison

For each frozen checkpoint, fit new linear board readouts at all 16 layers on
the same 160 probe games, validate on 40 other games. Identical per-layer seeds,
20 epochs, batch 256, LR 1e-3. There are 15,287 probe positions total.
The traced game is never used to fit or select a readout.

Score correctly recovered colored pieces, with no accuracy credit for empty
squares. GZrvqz3j is evaluated at 36 black-to-move positions, plies 1,3,...,71.
The denominator below is 823 piece instances across those positions.

| Model | Validation-selected layer | Trace recovered pieces | Trace precision | Before 36...g5 |
|---|---:|---:|---:|---:|
| Starting champion | 12 | 738/823 (89.67%) | 96.09% | 8/10 |
| Next-move-only control | 11 | 750/823 (91.13%) | 97.02% | 8/10 |
| Involved-piece objective | 11 | 748/823 (90.89%) | 96.89% | 8/10 |

Held-out recovered-piece recall at each selected layer is essentially flat:
86.07% champion, 86.02% control, 86.08% involved.

Layer selection explains the apparent trace improvement over the starting
champion. Holding layer 12 fixed, recall is 89.67%, 89.43%, and 89.67%, respectively.
At layer 11 it is 91.37%, 91.13%, and 90.89%. At the final layer it is 87.61%,
87.24%, and 87.12%. None establishes an overall improvement from this objective.

Before `36...g5`, the champion's selected readout omits the black king on h4
and rook on a6. Both trained models' selected readouts still omit the rook and
decode h4 as a white pawn. All recover 8 of the 10 actual pieces.

## Training outcome and next decision

- New head held-out location accuracy: 44.57% overall, 47.74% movers,
  24.58% captured pieces, 98.18% castling rooks.
- Final next-move loss: 1.39308 control versus 1.40024 involved (lower is better).
- The control's auxiliary head is untrained, so its accuracy is not a valid
  representation baseline. The comparisons above use separately fitted probes.
- Readout errors do not prove what the model believes. One trace and one seed
  cannot settle whether the objective can help. Playing strength was not tested.

Do not scale this configuration yet. A useful next diagnostic is fitting the
fresh identity-location head on a frozen champion at a higher head-specific
learning rate, then testing joint training against that calibrated starting point.
This would distinguish a poorly fitted new head from insufficient state information.

## Artifacts

- Volume root: `/data/experiments/involved-pieces-20260921` on `nanodanya-data`.
- Models: `control/final.pt`, `involved/final.pt`; original champion unchanged.
- Provenance: `preparation.json`; per-arm `metrics.jsonl`, `result.json`.
- Local results: `data/probes/involved-pieces-20260921/result.json` and
  `fixed-layer-comparison.json` (includes decoded boards at every traced position).
- Runner: `modal_involved_pilot.py`; independent evaluator: `probes/compare_involved.py`.

## Protocol and run commands (moved from training/README.md, 2026-09-22)

Replay standard-start SAN games once to label the mover and, when applicable,
the captured piece or castling rook. Each target is the piece's location **before**
the next move. The 32 identities follow pieces from their starting squares;
promoted pieces retain their pawn identity. No other pieces receive supervision.

```sh
.venv/bin/python data_prep/prepare_involved_pieces.py \
  --data-dir data/processed --out-dir data/involved-pieces

PYTHONPATH=.:nanochat DATASET_DIR=processed PIECE_LABELS_DIR=data/involved-pieces \
  CKPT_DIR=output/involved-pieces PIECE_LOSS_WEIGHT=0.1 \
  MAX_ITERS=1000 .venv/bin/python training/train_involved_pieces.py
```

Set `INIT_CHECKPOINT` to continue an existing policy; otherwise architecture
settings match `train.py`. For a matched next-move-only control, use a separate
`CKPT_DIR` and `PIECE_LOSS_WEIGHT=0`, keeping the seed, initialization, data,
batch size, steps, and learning rate identical. The weight 0.1 is an initial
experiment setting, not a tuned result.

`PIECE_HEAD_LR` sets the head's learning rate independently (defaults to `LR`).
Use `FREEZE_POLICY=1` to fit only the head on an unchanged policy. To follow with
joint training, point `INIT_CHECKPOINT` at that checkpoint and set
`INIT_PIECE_HEAD=1`, keeping a separate output directory. This loads both weights
with fresh optimizer state. Reusing a `CKPT_DIR` containing `latest.pt` instead
resumes that run with its original protocol and optimizer state.

The head predicts a distribution over 64 locations per identity. The next move
only selects which outputs receive loss; it is not an input to those predictions.
Windows missing a game's start receive no piece loss until the next visible BOS.
The objective averages cross-entropy over supervised pieces and adds the weighted
result to next-move loss. Empty squares earn no accuracy credit.

`metrics.jsonl` reports next-move loss and location accuracy/counts for movers,
captured pieces, and castling rooks. These are auxiliary-head metrics, not proof
of stronger play; compare held-out tactics and strength separately. `best.pt` is
selected by piece loss when the policy is frozen, otherwise by next-move loss.
`latest.pt` supports resume. Both load through
ordinary inference; the auxiliary head is stored separately from policy weights.

The existing Modal launcher supports both scripts. For preparation, set
`MODAL_GPU=none`, pass `--script data_prep/prepare_involved_pieces.py`, and supply
`DATASET_DIR` and a `/data/...` `PIECE_LABELS_DIR` through `--env-overrides`.
For training, pass `--script training/train_involved_pieces.py` with those same
paths and a distinct `/data/...` `CKPT_DIR`. In both cases, `--datasets` names
the source dataset on the volume.

`scratch/modal_involved_pilot.py` runs the matched 1,000-update H100 pilot on the
`second` profile (workspace2), using its existing 15M checkpoint. It prepares a fixed
100k-game subset in the checkpoint vocabulary, reserves 5k validation games and
200 separate probe games, and compares independent frozen-model readouts along
GZrvqz3j. The control's unused auxiliary head is untrained; its head accuracy is
not a representation baseline. Compare the independently fitted readouts.

The runner's `--action head-warmup` fits the head for 1,000 updates at `1e-3`
with the policy frozen, then runs 1,000 joint updates with policy LR `3e-6`
and piece loss weight `0.1`. It verifies unchanged policy weights after warmup
and repeats the independent readout comparison. After completion,
`--action head-trace` scores the trained identity heads on the same trace.
This diagnostic uses ground truth to select living identities for scoring;
it does not measure whether the head can detect captures or form a valid board.

For a collision penalty, provide `PIECE_OCCUPANCY_DIR` and set
`COLLISION_LOSS_WEIGHT` (default `0`). `prepare_piece_occupancy.prepare` derives
pre-move living-identity masks from the validated involved-piece labels and
replays validation games for all-living-piece location metrics. The head still
receives only move-involved location supervision.

The penalty is expected colliding pairs divided by living-piece count, averaged
over valid positions. It excludes captured identities and incomplete histories.
The loss becomes next-move CE + weighted involved-piece CE + weighted collision
penalty. This discourages duplicate locations without guaranteeing uniqueness;
evaluate raw argmax collisions and all-living-piece accuracy together.

`modal_involved_pilot.py --action collision` compares 1,000 additional frozen-policy
head updates with collision weights `0` and `0.1`, both initialized from the fitted
`head-warmup` checkpoint. It verifies exact policy preservation and evaluates
all 74 traced-game positions. With occupancy metrics enabled, frozen-head
`best.pt` selection uses the combined head loss; the pilot compares `final.pt`.

Set `PIECE_LOCATION_SCOPE=all_living` to replace involved-piece CE with mean
location CE over every living identity. It uses `train_locations.npy` and its
provenance file in `PIECE_OCCUPANCY_DIR`, produced by
`prepare_piece_occupancy.prepare_training_locations`. Captured identities and
incomplete prefixes remain masked. The collision term is unchanged. Metrics
report both involved-piece accuracy and all-living-piece accuracy/loss.

`modal_involved_pilot.py --action all-pieces` starts from the fitted
`collision-penalty` head and compares 1,000 further updates of the existing
involved-only objective against 1,000 updates of all-living location loss.
Both use a frozen champion, head LR `1e-3`, location weight `0.1`, collision
weight `0.1`, and identical batches. The results include all 74 traced positions.

`modal_involved_pilot.py --action joint-board` continues from `all-pieces/final.pt`
for 1,000 updates with the transformer unfrozen. The control uses next-move CE
alone and preserves the fitted head. The treatment adds `0.1 * all-living CE`
and `0.1 * collision loss`, updating both policy and head. Both use policy LR
`3e-6`, head LR `1e-3`, fresh optimizers, and identical batches. After training,
`MODAL_PROFILE=second modal run --detach modal_board_benchmark.py` compares their
final checkpoints with the original champion using the canonical benchmarks:
held-out legality, termination, engine move quality, and 400 games per pairing.
