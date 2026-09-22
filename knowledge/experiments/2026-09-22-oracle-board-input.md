# Supplying correct piece locations to the move predictor

Completed training and benchmarking on workspace2 H100s. The actual-board model
scores **50.35%** against the matched constant-board control over 4,000 games
(95% interval **48.95–51.76%**). Direct board inputs did not establish stronger
play in this 1,000-update pilot. The model uses the input, but still chooses
the fatal `36...g5` with the correct board supplied. This follows the
[board-loss confirmation](2026-09-22-joint-board-confirmation.md), where better
piece readout did not establish stronger play. The new intervention supplies
correct piece locations directly to the move predictor during training and play.

## Matched experiment

- Initialize both arms from the original 15M champion, SHA256
  `7e72148ed260564d671a67f5e0bc9bc4d82fd11695a34d9f4708858033955b0e`.
- Both have the same additional board encoder. Oracle gets actual piece features;
  control gets the same constant blank features at every position. The control's
  encoder is trainable, so it can learn a constant offset but no position information.
- Features: 32 stable starting-piece identities, each with square 0–63 or captured
  64, and current piece kind 1–6 or absent 0. Promotion retains the pawn identity
  and updates its kind. Color is fixed by identity. No empty-square accuracy metric.
- An identity-specific location embedding and a kind embedding produce 32 numbers
  per piece. Concatenate all 32 pieces and project to 1,024 numbers. Add that vector
  to the SAN token embedding before the existing input normalization and transformer.
- The projection starts at zero, preserving the initial policy exactly. Move
  vocabulary/output remain 14,750 tokens. The encoder adds 1,115,360 parameters.
- Standard next-move CE only. No piece-location or collision objective.
- Both update the transformer and encoder for 1,000 steps, same seed 42, batch 32,
  sequence 512, fresh AdamW optimizer, policy LR `3e-6`, encoder LR `1e-4`.
- Same existing 100k training games and 5k validation games. Compare final step
  1,000 checkpoints, not a checkpoint chosen from benchmark scores.

The input describes the board **after the observed input token**, before the
target move. BOS provides the initial board. EOS inputs get blank features.
Features do not depend on whether the next token is EOS or a move. No future
move selects which pieces are exposed. At inference, observed SAN is replayed
with python-chess and each current board is supplied alongside its token, including
when KV caching is used. Complete standard-start histories are required.

This input supplies piece locations and kinds, not an engine evaluation or best
move. Castling rights, en passant eligibility, and draw history are available
through SAN history and the shared runtime legal mask, not separate new features.
Training windows may start mid-game; oracle inputs still provide the actual board
there. Report full-prefix move loss separately, and use full histories in games.

## Training results

Both arms completed the fixed 1,000 updates. Initial metrics match exactly,
including next-move CE `1.405723667` and zero board-vector magnitude.

| Validation metric | Constant-board control | Actual-board model |
|---|---:|---:|
| All-token next-move CE | 1.395844 | 1.389004 |
| All-token CE with board blanked | 1.395844 | 1.410916 |
| Complete-prefix move CE | 1.338415 | 1.339659 |
| Complete-prefix move CE with board blanked | 1.338415 | 1.349069 |

The complete-prefix metric covers 145,624 move targets in the fixed validation
windows, excluding BOS/EOS targets and predictions before the first visible BOS.
The all-token metric includes those other cases. The oracle's lower aggregate
loss does not extend to this complete-prefix move subset. Removing its board
input worsens prediction, but useful reliance must still be assessed against
the matched control and in generated games.

## Validation and benchmark protocol

- Feature expansion uses previously validated mover/capture/castling labels.
  Independent chess replay checked 500 sampled training games (37,193 tokens)
  and all 5,000 validation games (387,133 tokens), with exact agreement.
- 53 relevant local tests passed, including both colors' special moves, promotion,
  capture, future-input isolation, initial-policy preservation, encoder gradients,
  control invariance, full versus cached inference, cache reset, checkpoint loading,
  and a complete tiny training/resume run. Python compilation/whitespace checks pass.
- Fixed 4,000 oracle-versus-control games, 2,000 per color, no interim stopping.
  Eighty shards of 50, seed `20270000 + offset`, offsets 0–3,950 by 50.
  Temperature 0.8, legal masking, EOS suppressed, KV cache, 200-ply cap.
- Raw legality/EOS checks on 4,096 probe positions; legal greedy move quality on
  1,024 probe positions, Stockfish depth 10, same engine binary across arms.
- Probe ablation on 200 separate games, at most 140 plies each: compare real-board
  and blank-board next-move CE and raw greedy choices. This tests dependence on the
  supplied input; blank boards are out of distribution for an oracle in active play,
  so this alone does not establish useful board reasoning.
- Revisit GZrvqz3j before `36...g5`, with real/blank board inputs. It is a previously
  examined diagnostic, not a fresh generalization test.
- Replay all generated games to check results, terminations, lengths, colors, IDs,
  and settings. Verify checkpoint hashes and matching move-quality FENs/engine hash.
- Score counts draws as half a point. Report 20,000-resample color-stratified 95%
  bootstrap intervals; they cover game sampling, not variation across training seeds.

## Playing strength

| Metric | Actual-board model versus constant-board control |
|---|---:|
| Games | 4,000 |
| Wins / draws / losses | 1,666 / 696 / 1,638 |
| Score | 50.35% |
| 95% color-stratified bootstrap interval | 48.95–51.76% |
| White score, 2,000 games | 52.425% |
| Black score, 2,000 games | 48.275% |
| Games reaching 200-ply cap | 43 (1.075%) |

Every game replayed legally, with outcomes, terminations, lengths, IDs, balanced
colors, settings, and checkpoint hashes verified. Terminations: 3,304 checkmates,
306 fivefold repetitions, 241 stalemates, 106 insufficient-material draws, and
43 capped games. Counting every capped game as an oracle loss or every one as an
oracle win gives scores of 49.81–50.89%. There is no demonstrated strength gain.

| Fixed-position metric | Constant-board control | Actual-board model |
|---|---:|---:|
| Genuine illegal raw top choices / 4,096 | 7 | 8 |
| Strict illegal choices, including under-disambiguation / 4,096 | 12 | 13 |
| Mean legal probability mass | 98.190% | 98.185% |
| EOS top choices / 4,096 | 10 | 10 |
| 300+ cp mistakes / 1,024 | 26 (2.54%) | 33 (3.22%) |
| Median centipawn loss | 13 | 14 |
| Engine best-move agreement | 47.56% | 47.85% |

All 1,024 move-quality FENs and engine binary hashes match; no positions failed
scoring. These finite-depth, correlated positions do not show a clear move-quality
improvement either. Average cp loss is not emphasized because mate scores are
encoded as 100,000 cp in the canonical benchmark.

## Does the model use the board?

On 200 separate probe games, 14,859 move targets with full prefixes:

| Condition | Move CE | Raw greedy choices changed by blanking |
|---|---:|---:|
| Constant-board control | 1.334928 | 0/14,859 |
| Actual-board model, correct boards | 1.338192 | 1,188/14,859 (8.00%) |
| Same actual-board model, boards blanked | 1.348262 | Same comparison |

The model depends on the supplied input: removing it changes predictions and
worsens its own CE. However, with real boards its CE remains slightly worse than
the matched control. Dependence on board inputs is not evidence that they improve
move choice or that every supplied fact is reliably used.

## The traced `g5` failure

Before `36...g5`, the actual-board input includes the black rook correctly on a6,
along with every other living identity and its current kind. It does not use the
earlier head's incorrect rook prediction. Exhaustive legal replies identify
`g5` as the only move permitting immediate mate, via `Rh2#`.

| Model / input | Greedy move | Probability of `g5` |
|---|---|---:|
| Original champion | g5 | 51.32% |
| Constant-board control | g5 | 60.15% |
| Actual-board model, correct board | g5 | 63.12% |
| Same actual-board model, blank board | g5 | 52.39% |

Probabilities use temperature 1, legal masking, and EOS suppression. The game
benchmark uses temperature 0.8. In this position, the correct board input actually
raises the model's probability of the fatal move relative to its blank-input
ablation. Supplying accurate positions has not repaired the tactical failure.

This weakens the practical hypothesis that explicit positions alone would unlock
the existing policy's tactical ability. It does not establish that board inputs
cannot help with a different interface, more adaptation, or another objective.
For this implementation and training budget, the model uses the board without
a demonstrated playing-strength gain. Do not promote this checkpoint on these
results. A next investigation should evaluate threats and consequences across
a broader held-out tactical set rather than train on this one traced position.

## Saved outputs

App: https://modal.com/apps/workspace2/main/ap-133ufSm2UfIZD3wo6PfROS

Volume `nanodanya-data`, root `/data/experiments/board-input-20260922`:

- `boards/`: causal board arrays and provenance.
- `control/` and `oracle/`: baseline, training metrics, resumable `latest.pt`,
  inference `final.pt`, and training receipts.
- `benchmarks/`: benchmark requests, checkpoint hashes, game logs, and receipts.
- `diagnostics.json`: held-out board ablation and critical-position choices.
- Local results and replay audit: `benchmark/results/board-input-20260922/`.

Final checkpoint SHA256:

- Control: `d30440939a8e64634169cefecc6f7254b3e646c893b6b1f2f378fd6a0d9402e8`.
- Actual-board model: `b07f4d6ac0a7330faad2937954e9dee6f4dca4a0251191cda073196da34828be`.

Training function times, including compilation/validation/checkpoint work:
251 seconds control and 202 seconds actual-board model.

```sh
MODAL_PROFILE=second modal run --detach modal_board_input.py
# Resume only evaluation after completed training:
MODAL_PROFILE=second modal run --detach modal_board_input.py --benchmark-only
```

New checkpoints need the board-aware `model_loading.load_model` path for inference.
Existing ordinary checkpoints still use the original GPT path. Source changes
include an optional embedding addition in the nanochat submodule. No deployment,
commits, or pushes are included.
