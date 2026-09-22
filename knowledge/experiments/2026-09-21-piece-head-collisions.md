# Training the piece head against collisions

Completed on workspace2 H100, 2026-09-21. A collision penalty substantially reduces
overlapping predictions and improves all-living-piece accuracy. It does not
guarantee unique locations or correct boards. The champion remains unchanged.

## Matched experiment

- Both arms start from `head-warmup/final.pt`, the fitted head on the original
  frozen champion described in [the warmup experiment](2026-09-21-involved-piece-head-warmup.md).
- Same 100k training games, 5k validation games, seed 42, sampled batches,
  head LR `1e-3`, batch 32, context 512, 1,000 extra updates, and fresh optimizer.
- Control: next-move CE + `0.1 * involved-piece CE`.
- Penalty: control objective + `0.1 * collision loss`.
- For each valid position, collision loss is
  `sum_{i<j, both alive} sum_square p_i(square) * p_j(square) / living_piece_count`.
  Average over positions with at least two living pieces. This is expected
  colliding pairs per living piece, not a hard constraint on argmax outputs.
- Living masks are pre-move, derived from previously validated capture labels.
  Captured identities, BOS/EOS targets, and incomplete game prefixes are excluded.
  Ground-truth locations of uninvolved pieces are used for validation only.
- Both final policies verified bitwise identical to the original champion.
  Next-move loss remains exactly `1.4057233810424805`.
- App: https://modal.com/apps/workspace2/main/ap-QGexgCUh5rowidKEUeaNQR

## Held-out validation

Same fixed windows: 145,624 scored positions, 3,431,724 living-piece instances,
and 181,780 involved-piece targets. Positions can repeat across sampled windows.
Empty squares receive no accuracy credit. Comparisons use final step 1,000.

| Metric | Initial fitted head | Extra training, no penalty | Collision penalty |
|---|---:|---:|---:|
| Positions containing collisions | 95.46% | 95.60% | 28.66% |
| All-living-piece location accuracy | 77.48% | 78.33% | 86.08% |
| Move-involved location accuracy | 97.90% | 98.16% | 97.78% |
| Exactly correct living-piece placements | 735 | 1,152 | 7,061 |

The improvement is not merely softer probabilities: raw argmax collisions
decrease. There is a small tradeoff in involved-piece accuracy. Fully correct
placements still account for only 4.85% of validation positions after the penalty.

## Traced game, all 74 positions

These include the starting board, every White and Black move, and final mate.
They are broader than the earlier 36-position Black-only evaluation.

| Metric | Initial fitted head | Extra training, no penalty | Collision penalty |
|---|---:|---:|---:|
| Positions containing collisions | 73/74 | 71/74 | 16/74 |
| Total squares with overlapping predictions | 181 | 174 | 18 |
| Correct living-piece locations | 1,268/1,676 | 1,276/1,676 | 1,469/1,676 |
| All-living-piece accuracy | 75.66% | 76.13% | 87.65% |
| Move-involved targets correct | 96/96 | 96/96 | 96/96 |
| Exactly correct placements | 0/74 | 0/74 | 3/74 |

At the familiar position before `36...g5`, the penalty removes all collisions
but gets only 8/10 locations correct, versus 9/10 for the continued control:
the white king is predicted on g1 instead of f3, and the white rook on a1 instead
of c2. The black king and rook remain correct. Avoiding overlap alone cannot
tell an uninvolved piece which alternative square is correct.

The penalty improves the head across this game and held-out games; it does not
solve full-board reconstruction. No legal-board correction was applied, and
captured identities were filtered using ground truth throughout. Playing strength
is unchanged because the policy is unchanged.

## Artifacts and verification

- Volume `nanodanya-data`, root `/data/experiments/involved-pieces-20260921`.
- New head: `collision-penalty/final.pt`; matched control: `collision-control/final.pt`.
- Per arm: `baseline.json`, `metrics.jsonl`, `result.json`, `best.pt`, `latest.pt`.
- Local complete results and raw game predictions:
  `data/probes/involved-pieces-20260921/collision-result.json`.
- Remote raw game predictions: `collision-game.json`.
- Tests check pre-capture labels, incomplete-prefix masking, captured-piece
  exclusion, loss values and gradients, policy invariance, and recorded metrics.
  Fifteen involved-piece tests and five probe tests pass.
- Existing published viewer is unchanged and still shows the earlier heads.
