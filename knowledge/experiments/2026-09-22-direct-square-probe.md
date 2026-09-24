# Direct-square probe with normal sampling

One frozen checkpoint: latest actual-board-input model, SHA256
`b07f4d6ac0a7330faad2937954e9dee6f4dca4a0251191cda073196da34828be`.
It receives its normal correct board inputs. Only a fresh linear readout learns.
This experiment measures recoverable piece placement, not playing strength or
board memory without those inputs. No checkpoint comparison is part of the run.

## Fixed protocol

- Read the final normalized hidden state used by the move-output layer.
- Linear 1024 → 64 × 13 outputs: empty, white P/N/B/R/Q/K, black P/N/B/R/Q/K.
- Per board, loss is 0.5 × mean occupied-square CE + 0.5 × mean empty-square CE.
  Average those board losses over valid positions. Ground truth defines the two
  groups. No material values or piece-type weights. No collision loss.
- Normal sampling: uniformly sampled token-window starts in the existing 100k-game
  training split. No stage oversampling. Batch 32, context 512, 3,000 updates,
  AdamW LR 0.001, default weight decay 0.01, head/data seeds 42, clip norm 1.0.
- Include initial boards during training, exclude EOS padding and positions before
  the first visible BOS in a cropped window. Each scored position has full game
  history within the window. This preserves the existing training sampler.
- Ten fixed validation windows per evaluation, seed 43, including step zero.
  Train/validation hidden-state forwards use bfloat16 on H100; held-out game and
  trace evaluation use float32. No layer or step selection using probe games.
- Use the existing 200 held-out probe games and the previously traced game.
  Those probe games were excluded from continuation/readout fitting but have been
  repeatedly inspected, so this is a development evaluation, not a fresh final
  test. Original champion pretraining overlap is not established.

## Metrics

Primary: exact-board reconstruction by completed move number (1–10, 11–30, 31+).
Average position accuracy within each game/band, then equally across contributing
games. Report game counts and game-bootstrap 95% intervals (10,000 draws, seed 42).
Also retain pooled-position rates, mean piece errors, 0/1/2/3+ error distributions,
and unweighted piece precision/recall. No material-weighted metric.

A piece entry matches only if colour, kind and square match. Error count is the
larger of missing and extra entries, so relocating one piece counts once. Predict
by independent square argmax without truth filtering or legality correction.
Identical piece identities are irrelevant. One square cannot contain multiple
predictions, but extra kings and other illegal placements remain possible and
are errors. Initial boards are reported separately from post-move headline scores.

The primary per-game averaging differs from the earlier pooled-position charts;
pooled results remain in the JSON for consistent comparisons if needed later.
These metrics cover piece placement, not turn, castling rights or en passant.

## Validation and reproduction

67 relevant tests passed. Tests establish equal occupied/empty loss weight at
2, 10 and 32 pieces, per-board rather than batch-wide normalization, a large loss
for empty-everywhere predictions on sparse boards, loss masks, labels after
captures/castling/en passant/promotions, frozen-backbone behavior and complete
fit/resume/evaluation. Compilation and whitespace checks passed.

```sh
MODAL_PROFILE=second modal run --detach modal_board_readout.py --square-board --arm oracle
```

Remote: `/data/experiments/board-input-20260922/square-readouts/oracle/`.
Local: `data/probes/board-input-20260922/square-readout-result.json`.
Independent audit: `probes/audit_square_readout.py`, using exported predictions
and independently checked labels from the previous readout run.

## Results

Completed on workspace2 H100:
https://modal.com/apps/workspace2/main/ap-TDGUibWdOSWfk5cZQbd3gp

Final fixed-step validation loss: **0.033230**, from **2.748715** before fitting.
Occupied-square component: 0.053508; empty-square component: 0.012952.
The probe has 852,800 trainable parameters. Frozen-backbone tensor hashes match
both before/after fitting and the previous experiment. Runtime including scoring
was 130.76 seconds on H100, excluding container startup.

Primary results, equal weight per contributing game:

| Move band | Games | Positions | Exact board | Game-bootstrap 95% interval | Mean piece errors |
| --- | ---: | ---: | ---: | ---: | ---: |

| 1–10 | 200 | 4,000 | 88.83% | 87.53%–90.10% | 0.127 |
| 11–30 | 200 | 7,054 | 72.23% | 70.14%–74.22% | 0.396 |
| 31+ | 121 | 4,033 | 57.13% | 53.08%–61.07% | 0.752 |

Error distribution, also averaging within games before averaging across games:

| Move band | 0 errors | 1 | 2 | 3+ |
| --- | ---: | ---: | ---: | ---: |
| 1–10 | 88.83% | 10.00% | 0.90% | 0.28% |
| 11–30 | 72.23% | 20.46% | 4.93% | 2.38% |
| 31+ | 57.13% | 26.15% | 10.40% | 6.32% |

The pooled-position exact rates are 88.83%, 72.20%, and 57.92%, respectively.
The primary per-game averages above stop longer games receiving more weight.
All 200 initial boards were correct and are excluded from post-move metrics.
No games were truncated. Final evaluation contains 15,087 post-move positions.

On the traced game, 44/73 post-move boards are reconstructed exactly. The position
immediately before 36...g5 is completely correct: all ten pieces match, with no
missing or invented piece. This measures extraction by the new probe, not a change
to the chess model or evidence that its move predictor uses every decoded fact.

The loss was still decreasing at the fixed endpoint, so this is not an established
readout ceiling. This single-seed experiment changes both the output format and
loss relative to the earlier identity readout; it does not isolate their individual
effects. No new chess-strength benchmark or model comparison was run.

Independent set-based audit passed for every saved prediction: source/checkpoint
hashes, frozen-state equality, identity-label to square-label conversion, game/ply
alignment, aggregate/per-game/per-stage counts, per-game averaging, loss summaries,
all 74 trace FENs, and the complete logged validation sequence. Audit receipt:
`data/probes/board-input-20260922/square-readout-audit.json`.

Probe learning curve:

| Updates | Balanced validation loss | Occupied loss | Empty loss |
| --- | ---: | ---: | ---: |
| 0 | 2.748715 | 2.748818 | 2.748611 |
| 1 | 2.363694 | 2.419650 | 2.307738 |
| 500 | 0.087963 | 0.136132 | 0.039794 |
| 1000 | 0.056365 | 0.088963 | 0.023768 |
| 1500 | 0.045143 | 0.072074 | 0.018212 |
| 2000 | 0.039202 | 0.062888 | 0.015516 |
| 2500 | 0.035464 | 0.056914 | 0.014013 |
| 3000 | 0.033230 | 0.053508 | 0.012952 |
