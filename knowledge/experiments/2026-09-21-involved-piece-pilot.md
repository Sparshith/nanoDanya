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
