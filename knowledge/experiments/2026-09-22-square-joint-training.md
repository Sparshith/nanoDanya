# Balanced direct-square supervision during continuation

Completed on workspace2 H100s. Fresh-probe board reconstruction improved modestly
over matched continuation. The fixed chess benchmark does not establish a playing-
strength gain: 50.6% score, 95% interval 48.8%–52.45%.

Question: does adding balanced board reconstruction to next-move training improve
recoverable board placement and playing strength relative to matched next-move-only
continuation? Both models begin at the same latest board-input checkpoint. This is
not a test of SAN-only board memory; both keep their normal true board inputs.

## Protocol fixed before results

- Base checkpoint `/data/experiments/board-input-20260922/oracle/final.pt`, SHA256
  `b07f4d6ac0a7330faad2937954e9dee6f4dca4a0251191cda073196da34828be`.
- Both use the same already fitted direct-square linear head (64 × 13 classes).
- Joint arm: next-move CE + **0.1 × board loss**, updating policy, board encoder and
  head. Board loss is 50/50 occupied/empty CE, normalized separately per board.
- Control arm: same next-move loss and head loss, but detach the hidden state before
  the board head. Its board loss updates only the head. A test verifies bitwise
  equality of the control backbone update with a pure next-move-only update.
- Identical data, initial tensors, fresh optimizer, random seed 42 and sampled
  batches. Uniform token-window starts, no move-stage oversampling or material
  weights. Existing 100k-game train split, 5k validation split, complete-prefix
  masking for board targets; next-token loss uses all tokens as in prior runs.
- **1,000 fixed updates**, batch 32, context 512, AdamW. Policy and existing board
  encoder LR 3e-6, head LR 1e-4, weight decay 0.01. Clip backbone and head gradients
  separately at 1.0 so head gradients cannot influence the control's clipping.
- Ten fixed validation batches (seed 43), every 100 steps and step 1, plus baseline.
  Report next-move CE, complete-prefix next-move CE, balanced board CE and attached-
  head exact-board accuracy. Compare final checkpoints, no best-step selection.

## Representation evaluation

Score both attached heads, then freeze each resulting backbone and independently
fit a **fresh** direct-square linear probe for 3,000 updates with identical seed,
head initialization, data and LR 0.001. Fresh-probe exact-board accuracy by move
stage is primary, avoiding an inference about representation from the auxiliary
head's own improvement alone. Use the same 200 development probe games and trace.
Average within each game/band then across games; also retain pooled counts and
0/1/2/3+ piece errors. No material-weighted metric. The probe games exclude this
continuation/readout training but have been inspected repeatedly; they are not a
fresh final test and their original pretraining overlap is unknown.

## Playing-strength evaluation

- Exactly **2,000** joint-versus-control games, 1,000 per color, standard starting
  position, temperature **0.2** for both, legal masking, EOS off, KV cache.
- Fixed 200-ply cap, capped games scored as draws and reported separately.
- Forty shards of 50, seed `20280922 + offset`, offsets 0 through 1950.
  No early stopping or sample expansion based on intermediate scores.
- Score `(wins + 0.5 draws) / 2000`, with color-stratified game bootstrap.
  This is a pilot against one matched opponent and a single training seed, not
  a universal Elo measurement or evidence of an effect across training seeds.
- Secondary: greedy move-quality diagnostic on 1,024 deterministic development
  positions, Stockfish depth 10; matched FENs and engine hash required.
- Revisit the traced pre-g5 position for both decoding and move probabilities;
  do not use this already known example for selection or training targets.

## Checks and outputs

40 focused tests passed, including joint gradient flow into policy and encoder,
control gradient isolation, equality to a next-move-only control update, standard
inference logits, checkpoint save/load/resume, masks and per-board balanced loss.

```sh
MODAL_PROFILE=second modal run --detach modal_square_joint.py
```

Remote: `/data/experiments/square-joint-20260922/{control,joint}/`.
Local full result: `benchmark/results/square-joint-20260922/result.json`.
Fresh/attached predictions and independent game/representation audits will be
saved with the result. The auxiliary head is saved as training metadata; standard
inference uses the unchanged model-loading path and no square decoder.

No deployment, commits or pushes are part of this experiment.

## Results

App: https://modal.com/apps/workspace2/main/ap-qeqNns8WE5HG7tVzw6ZT0q

All training, fresh-readout fits, 2,000 games and both Stockfish diagnostics finished.
The independent audit passed. The two arms have identical initialization and
baseline validation values. The board head's control gradient is detached; the
fresh evaluation heads are independently initialized and fitted identically.

### Fresh probes: primary representation result

Each position is scored on full colored piece placement, then averaged within its
game/move band and equally across contributing games. Initial boards are excluded.
Intervals below are paired bootstrap differences over games (20,000 draws, seed 42).

| Move band | Games | Control | Board loss | Difference (percentage points) | 95% interval |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1–10 | 200 | 88.10% | 89.15% | +1.05 | +0.75 to +1.37 |
| 11–30 | 200 | 71.05% | 73.30% | +2.25 | +1.86 to +2.67 |
| 31+ | 121 | 56.19% | 58.10% | +1.90 | +1.34 to +2.51 |

This supports better linear recoverability of piece placement for this trained
checkpoint pair. It does not establish an effect across training seeds, layer
choices or all possible representations. Normal true board inputs remain enabled
for both models. The three interval estimates are marginal, not multiplicity-
adjusted. These are development games that have been inspected before.

### Attached heads and losses

| Measure | Control | Board loss |
| --- | ---: | ---: |
| Validation next-token CE | 1.388210 | 1.387960 |
| Validation complete-prefix next-move CE | 1.340144 | 1.339862 |
| Validation attached-head balanced board CE | 0.032819 | 0.029667 |
| Validation attached-head exact-board rate | 0.756033 | 0.771981 |
| Attached head, moves 1–10 | 88.20% | 89.72% |
| Attached head, moves 11–30 | 71.92% | 73.69% |
| Attached head, moves 31+ | 58.05% | 59.76% |

The starting validation move CE was 1.38900378 and balanced board CE was 0.03322996
for both arms. Board loss reduces the final attached-head board CE by about 9.6%
relative to control. Next-move loss differs very little. Attached-head validation
includes initial positions; the primary full-history probe metric excludes them.

### Chess games

Joint-model outcomes: **780 wins, 464 draws, 756 losses** out of 2,000. Score
**50.60%**, color-stratified bootstrap 95% interval **48.80%–52.45%** (20,000 draws,
seed 20260922). No demonstrated playing-strength improvement in this pilot.

White score 37.7%; Black score 63.5%, with exactly 1,000 games each. There were
1,612 unique complete move sequences. At temperature 0.2, recurrent lines and
color asymmetry make this a specific matchup result, not general chess strength.

Eight games reached the 200-ply cap. Counting all capped games as joint losses or
all as wins gives 50.4%–50.8%, so caps do not explain the result. Terminations:
1,536 checkmates, 331 fivefold repetitions, 48 insufficient-material draws,
77 stalemates and 8 caps. All games replay legally; IDs, numbers, seeds, colors,
model hashes, outcomes, terminations and lengths match the fixed protocol.

### Secondary move-quality and trace

On the same 1,024 development FENs, median Stockfish loss is 13 cp for both models.
The control has 25 losses of at least 300 cp; the joint model has 26. Best-move
match rates are 46.29% and 46.78%, respectively. Engine binaries match. Mean cp
loss is 910.11 versus 814.04, but mate-score sentinels make this a poor standalone
summary; do not infer a strength gain from that mean. This engine diagnostic uses
greedy moves, unlike the temperature-0.2 games.

Both models still greedily choose g5 in the known traced position. At temperature
0.2, g5 probability is 98.58% for control and 98.63% for joint. This example was
not used to tune the run. Representation readout and move choice remain separate.

## Saved results and verification

- Full result: `benchmark/results/square-joint-20260922/result.json`.
- Audit: `benchmark/results/square-joint-20260922/audit.json`.
- Raw predictions: `{control,joint}-{attached,fresh}-predictions.npz` in that folder.
- Re-run audit: `PYTHONPATH=.:nanochat .venv/bin/python probes/audit_square_joint.py`.
- All exported arrays agree on games, plies and true labels. Aggregate, per-game,
  stage and trace counts were independently recomputed using piece-entry sets.
- Every trace FEN matches independently parsed chess positions. Fresh-probe
  frozen-state hashes match their corresponding trained checkpoint tensors.
- Training took 229.02 seconds for control and 198.70 seconds for joint, excluding
  container startup. No model was deployed or promoted.

Final checkpoint hashes:

```
control 30ed280ab9c8eb9d110bb4efbe0bb34ff745c59c4bc3e7082672c606dd26214e
joint   038d7df248493977dbc21eaf5c88aed2a3c5042b8e02ecfb09d3a5dc2e49010c
```
