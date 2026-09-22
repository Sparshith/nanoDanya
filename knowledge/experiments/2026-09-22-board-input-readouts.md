# Board representation after board-input training

Completed on workspace2 H100s. The new board-input checkpoint does not improve final-state
linear board decoding over the original champion under this matched protocol. It
slightly improves over the continuation control. This measures recoverable locations,
separately from next-move loss and play.

## Matched readout protocol

- Freeze all checkpoint parameters, including the board encoder where present.
  The actual-board model receives its normal correct board inputs; the control
  receives constant blank inputs and the original champion receives SAN only.
- Fit a separate fresh linear head on each model's final normalized transformer
  state, exactly the hidden state used by its move-output layer.
- Every head maps 1,024 features to 32 identity-specific distributions over 64
  locations. Identical initial head weights (seed 42), sampled batches, optimizer,
  learning rate `1e-3`, batch 32, sequence 512, and 3,000 updates.
- Only the fresh readout learns. Mean location CE covers all living identities
  after each observed input token, with no collision loss. Captured identities,
  EOS padding, and positions before the first visible BOS in a cropped window
  receive no loss or accuracy credit. Final boards and initial boards are included.
- Train on the existing 100k-game training split. Monitor ten fixed validation
  windows per evaluation (seed 43, 5k validation games). Compare the final step,
  with no layer selection, checkpoint selection, or trace-specific tuning.
- Primary evaluation: all full-history positions from the 200 separate probe
  games excluded from continuation and readout training. A game longer than
  the context limit is truncated at 512 tokens, never cropped from the left.
- Separately decode all 74 positions of GZrvqz3j, including the initial board
  and terminal mate. This is an already examined diagnostic, not a fresh test.
- Score living-piece location accuracy, exact living-piece placements, raw
  overlap frequency, and location accuracy away from an identity's starting square.
  No empty-square accuracy. Captured-piece detection and piece-kind prediction
  are not measured; truth determines which identities are alive and their kind.
- Held-out probe/trace scoring uses float32. Training and validation monitoring
  use bfloat16 autocast, matching the earlier head training workflow.

The readout is given hidden states only. It never directly receives the board
labels. The actual-board model's transformer does receive its normal board input.
Thus this asks whether locations are easier to recover under the checkpoint's
normal inputs, not whether it remembers those locations when board inputs are removed.

The older fitted champion head had a different sequence of involved-piece,
collision, and all-living objectives. It is not reused: all three fresh heads
here get the same objective and training budget.

## Verification

- Hash original checkpoint files before loading and frozen model state tensors
  before/after fitting. Assert frozen parameters have no gradients.
- Compare hidden-state readout against the actual move-layer input, including
  real and constant board inputs. Verify capture/padding and incomplete-prefix masks.
- Test a complete tiny fit, checkpoint resume, independent-game scoring, and trace.
- 43 relevant local tests passed; Python compilation and whitespace checks passed.

## Outputs

```sh
MODAL_PROFILE=second modal run --detach modal_board_readout.py
```

- Volume `nanodanya-data`, root `/data/experiments/board-input-20260922/readouts/`.
- Per arm `champion`, `control`, `oracle`: `latest.pt`, `head.pt`, `metrics.jsonl`,
  and `result.json`. Only small readout/optimizer states are saved; policy weights
  remain in their original files.
- Local results: `data/probes/board-input-20260922/readout-result.json`.
- Per-game raw counts support paired bootstrap comparisons across the same games.
- Every traced position includes the current FEN and all identity predictions.

No policy training, deployment, commits, or pushes.

## Results

200 games, 15,287 positions, 347,572 living-piece targets. No games were truncated.
These games were excluded from continuation and readout training; original champion
pretraining overlap is not established.

| Checkpoint | Living-piece location accuracy | All living locations correct | Positions with overlaps | Away-from-start accuracy |
| --- | ---: | ---: | ---: | ---: |
| Original champion | 97.4972% | 65.6113% | 29.0116% | 95.5923% |
| Constant-board control | 97.4365% | 64.9964% | 29.4237% | 95.4782% |
| New actual-board checkpoint | 97.4860% | 65.3824% | 29.0705% | 95.5224% |

Paired bootstrap over games, 20,000 resamples, seed 42:

- New minus champion location accuracy: -0.0112 percentage points,
  95% interval [-0.0410, +0.0194]. No demonstrated improvement.
- New minus control location accuracy: +0.0495 percentage points,
  95% interval [+0.0222, +0.0774]. Small gain over the control.
- New minus champion exact-placement rate: -0.2290 percentage points,
  95% interval [-0.6321, +0.1797].

On all 74 traced positions, champion recovered 1,640/1,676 living locations
(97.85%) and 46 exact placements; control 1,644 (98.09%) and 49; new checkpoint
1,636 (97.61%) and 45. Before 36...g5, all three recover 9/10 living pieces:
the black rook on a6 is decoded at c2, overlapping the white rook. The new
checkpoint does not fix this readout error.

These are fresh independent linear heads with no collision constraint. The
result establishes no improvement in this readout, not that every possible
representation or decoder is unchanged. Actual-board input was enabled for
the new checkpoint throughout evaluation.

All protocol, initial-head, frozen-weight, aggregate-count, trace-FEN and
per-game alignment checks passed. Runtime per H100 fit including evaluation:
139 seconds champion, 140 control, 136 new checkpoint, excluding startup.
An initial remote module packaging failure was fixed and its failed app stopped.
Successful run: https://modal.com/apps/workspace2/main/ap-rnyZUPIhtZ4CqZdQjZGxsr

Audit: `data/probes/board-input-20260922/readout-audit.json`.
Audit script: `scratch/involved-pilot/audit_board_readout.py`.
