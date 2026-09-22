# Complete board reconstruction and material weighting

Primary metric: exact colored piece placement, with no supplied alive mask or
piece kinds. Missing, invented, misplaced and incorrectly promoted pieces count.
Identical same-color pieces are interchangeable. Overlaps are retained and scored,
not repaired or deduplicated. This does not test turn, castling or en passant rights.

## Protocol

Three frozen checkpoints: original champion, constant-board continuation control,
and actual-board-input continuation. The latter receives its normal board inputs.
New matched linear heads on the final normalized hidden state predict 32 identities,
each with 65 locations (including captured) and 6 possible current piece kinds.
Color is fixed by identity. Decode independent argmax location and kind; predicted
captured identities are omitted using the prediction alone. No truth-based filtering.

All three heads start from identical weights and get identical batches: seed 42,
AdamW learning rate 0.001, batch 32, sequence 512, 3,000 updates on the same 100k-game
training split. Loss is mean all-identity location CE plus mean living-identity kind
CE. No material weighting in training, and no collision penalty. Exclude EOS and
incomplete prefixes from both losses. This changes the readout and its supervision,
not any chess policy weights. New results are not directly comparable with the
previous 64-location heads that were supplied the set of living pieces.

Evaluate all post-move positions of the same 200 games excluded from continuation
and readout training; original champion pretraining overlap is not established.
Initial positions are reported separately. Use FP32 for final evaluation. Also
reconstruct the previously examined 73-ply trace. Compare final head weights after
3,000 steps, with no selection on probe games or trace.

## Metrics

- Exact board: multisets of (color, current kind, square) agree completely.
- Error count: maximum of missing and extra piece entries, after exact matches.
  A single relocated or mistyped piece counts once. Report 0, 1, 2, 3+ errors.
- Piece precision/recall/F1: exact matched entries; empty squares earn no credit.
- Material-weighted F1: twice correctly located material divided by total actual
  plus predicted material. Pawn 1, knight/bishop 3, rook 5, queen 9. This penalizes
  invented material as well as missing material. Weights are a diagnostic choice,
  not estimates of tactical importance.
- Kings: excluded from the material weighting, reported as a separate both-kings-
  correct rate that also rejects invented kings. Exact-board score includes kings.
- Break down by moves 1–10, 11–30, and 31+, using completed ply. These are move-number
  bands, not a claim to classify opening/middlegame/endgame from position content.
- Per-piece precision and recall identify which kinds cause errors.

## Reproduction and checks

```sh
MODAL_PROFILE=second modal run --detach modal_board_readout.py --full-board
```

Remote output: `/data/experiments/board-input-20260922/full-board-readouts/{arm}/`.
Local result: `data/probes/board-input-20260922/full-board-readout-result.json`.
Each arm exports compressed actual/predicted arrays for independent histogram-based
audit in `probes/audit_full_board.py`, including paired bootstrap across games.

52 relevant local tests passed, covering identity swaps, colors, captures, ghosts,
overlaps, promotions, king scoring, weighted scoring, move bands, loss masks,
frozen backbone behavior and a complete tiny fit/resume/evaluation. No policy
training or deployment is part of this comparison.

## Results

Completed: https://modal.com/apps/workspace2/main/ap-QLheruHeuV21NpdTsyK2HT

200 games, 15,087 post-move positions, no truncated games. New checkpoint did not
show a clear gain over champion on exact-board accuracy or material-weighted F1.
The stricter full reconstruction score is substantially below the earlier score
that supplied alive status and kind. This is a changed test and changed readout,
not evidence that the underlying checkpoint regressed between the two evaluations.

| Checkpoint | Exact board | Mean errors | Material F1 | Both kings correct |
| --- | ---: | ---: | ---: | ---: |

| Champion | 35.06% | 1.401 | 95.19% | 98.16% |
| Constant-board control | 34.84% | 1.420 | 95.10% | 98.19% |
| New board-input checkpoint | 35.16% | 1.391 | 95.15% | 98.19% |

Error-count distribution, percentages of post-move positions:

| Checkpoint | 0 | 1 | 2 | 3+ |
| --- | ---: | ---: | ---: | ---: |
| champion | 35.06% | 25.56% | 19.57% | 19.81% |
| control | 34.84% | 25.31% | 19.43% | 20.41% |
| oracle | 35.16% | 25.68% | 19.49% | 19.66% |

Exact-board accuracy by move band:

| Completed moves | Positions | Champion | Control | New checkpoint |
| --- | ---: | ---: | ---: | ---: |
| 1–10 | 4,000 | 85.85% | 85.32% | 85.45% |
| 11–30 | 7,054 | 22.94% | 22.92% | 23.15% |
| 31+ | 4,033 | 5.88% | 5.60% | 6.30% |

The initial board is correct for all 200 games and all models, reported separately.
The later move bands are much harder for all three readouts. Their game composition
and material differ, so the band comparison is descriptive, not a causal experiment.

New minus champion, paired 20,000 game bootstrap (seed 42):

- Exact board: +0.106 percentage points, 95% interval [-0.292, +0.512].
- Mean errors: -0.010 errors per position, 95% interval [-0.022, +0.002].
- Material F1: -0.039 percentage points, 95% interval [-0.090, +0.013].

New minus control: exact board +0.325 percentage points (interval [-0.046, +0.694]);
mean errors -0.029 per position (interval [-0.040, -0.019]); material F1 +0.044
percentage points (interval [-0.002, +0.090]). These intervals capture variation
across games, not across readout initialization seeds.

All three readouts reconstruct exactly 21/73 traced post-move boards. Before g5,
all recover 8/10 real piece entries, with two missing entries and one extra entry
(two errors under the replacement convention). Kings are correct. No policy
weights changed. This is evidence about this fitted linear readout, not proof
of the best information any possible decoder could extract.

Audit passed: identical initial heads and protocols, checkpoint hashes, frozen
state hashes equal to the previous experiment, all game/ply/label alignment,
independent histogram recomputation of aggregate and per-game/stage counts,
metric ratios and exported-file hashes. All 74 trace FENs and missing/extra counts
were checked independently. Fits including scoring took 133–137 seconds per H100,
excluding startup. Results and audit JSON are under
`data/probes/board-input-20260922/`, with prefix `full-board-readout-`.
