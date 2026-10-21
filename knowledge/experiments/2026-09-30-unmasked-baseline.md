# Baseline with immediate losses for invalid predictions

The existing d9-200k INT4 baseline scores **1303 ±16 internal Elo** and solves
**1941/8800 puzzles (22.06%)** when the highest-logit token is played without
legal-move masking. No training or weight changes.

| Measurement | Previous masked games | Unmasked reevaluation |
| --- | ---: | ---: |
| Internal Elo, separate 1,200-game fits | 1516 ±15 | 1303 ±16 |
| Puzzles solved | 2023/8800 (22.99%) | 1941/8800 (22.06%) |
| Game score | 54.83% | 32.63% |
| Invalid-prediction forfeits | 0 (masked) | 691/1200 (57.58%) |

The previous published 1517 was from a larger mixed-model rating pool. For the
comparison above, its 1,200 games against the same three opponents were extracted
and fitted separately, yielding 1516. The new pool contains only the 1,200 new
unmasked games. Both fits anchor sf1500 at 1500; these are not human/Lichess ratings.

Opponent results, each over 400 games:

| Opponent | Masked W/D/L | Unmasked W/D/L |
| --- | ---: | ---: |
| sf1000 | 311/8/81 | 208/3/189 |
| sf1500 | 191/33/176 | 111/3/286 |
| sf2000 | 121/29/250 | 66/7/327 |

The paired opening bootstrap puts the game-score difference at -22.21 percentage
points (95% interval -25.67 to -18.63). Time-based engine adjudication can vary,
so this is not a claim that every changed game outcome is solely due to masking.

All 82 changed puzzle outcomes were previous solves that now fail, with no gains.
The paired puzzle difference is -0.93 percentage points (95% interval -1.13 to
-0.73). There were 394 invalid puzzle predictions in total; many were on puzzles
that the masked model also failed later or by choosing another wrong move.

## What caused forfeits

Of the 691 game forfeits, 174 were `<eos>` predictions, 474 were illegal moves,
28 omitted required SAN disambiguation, and 15 had incorrect/noncanonical SAN
(such as claiming a capture on an empty square). Check and mate suffixes are
ignored, as required by the token contract. All other notation must match.
There were 21,751 model decisions during the games, with 3.18% causing a forfeit.
A modest per-move failure rate accumulates into a large full-game loss rate.

## Protocol and validation

- Current browser artifact: 9,454,892 bytes, SHA-256
  `3abfac3fd6d8f859918ec9f0366399d582b9996a1564d17e3ad1c014a91562b6`.
- Its graph and weights are byte-identical to the previous d9-200k ONNX
  `9d179f227cac980c06aa26a5f9824e975ee1c1ef4d21044e19e9b83955107ebe`;
  only the display-name metadata differs.
- Greedy temperature 0, entire vocabulary, no EOS suppression, no retries or
  fallback moves. Invalid predictions immediately lose games or fail puzzles.
- Same 200 fixed openings, both colors, sf1000/sf1500/sf2000 at depth 1,
  10 ms/1300 cp adjudication, 400-ply cap. Fairy-Stockfish binary SHA-256
  `41b8b4d539adfd9924929ee4a948d1a37dd1e9beaa535a811cb5e7fee9e4cb99`.
- All 1,200 game records replay legally, match the previous openings and colors,
  and have scored results. Each forfeit matches its logged invalid prediction
  and is awarded as a loss to the predicting side.
- All 8,800 puzzle identities and solution depths match the previous evaluation;
  none were skipped.
- H100, FP32 PyTorch reconstructed from the exact exported weights, TF32 disabled.
  Every illegal or near-tied prediction and an initial sample were checked with
  native ONNX Runtime CPU. Across 34,485 decisions, 1,426 native checks found
  zero top-token disagreements; maximum absolute logit error was 3.38e-5.
  This is validated artifact inference, not a fresh browser strength benchmark.

## Cost and artifacts

workspace2 Modal profile `second`, app `ap-pRg8KvkVwD1z06q8MmlCro`, completed and
verified stopped with zero tasks. The run used 303.56 function seconds.
Estimated function compute: **$0.36**, cumulative **$49.22** for this conversation.
Retaining all earlier reservations and overhead leaves the ceiling at $96.84,
under the $100 cap. Estimates use [Modal public rates](https://modal.com/pricing),
checked September 30; they are not an invoice.

Remote: `/data/experiments/unmasked-baseline-20260930/` on `nanodanya-data`.
Local evidence: `output/unmasked-baseline-20260930/`, including protocol, budget,
raw game/puzzle results, native-check audits, independent rating fits, report,
source hashes, and shutdown verification.
Scripts: `scratch/unmasked_baseline.py`, `scratch/report_unmasked_baseline.py`.

The local challenge baseline numbers are updated; v1 remains pending.
Nothing was committed, pushed, or deployed.
