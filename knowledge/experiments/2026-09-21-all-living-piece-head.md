# Supervising every living piece

Completed on workspace2 H100, 2026-09-21. All-living location supervision greatly
improves full-board reconstruction compared with continuing the involved-only
objective. Collision frequency does not improve. The champion is unchanged.

## Matched protocol

- Both arms initialize from `collision-penalty/final.pt`, the frozen champion's
  head from [the collision experiment](2026-09-21-piece-head-collisions.md).
- Both run 1,000 more updates, head LR `1e-3`, batch 32, sequence 512, seed 42,
  fresh optimizer, identical sampled windows, original 100k training games.
- Control: `move CE + 0.1 * involved-piece CE + 0.1 * collision penalty`.
- Treatment: `move CE + 0.1 * all-living-piece CE + 0.1 * collision penalty`.
- All-living CE averages one correct-square target per living identity.
  Captured identities and incomplete prefixes are excluded. Empty squares
  provide no accuracy credit. No additional involved-only location term is used.
- Full training locations are expanded from validated mover/capture/castling
  identities and SAN destinations. Every mover and captured/rook source square
  is checked against the reconstructed state. Tests independently compare the
  results with board replay, including en passant, castling, and promotions.
- Both final policy state dictionaries are bitwise identical to the champion.
  Next-move loss stays exactly `1.4057233810424805`.
- App: https://modal.com/apps/workspace2/main/ap-v4sWmH0UXCTc9zede3qqGO
- Training function time: 150 seconds control, 108 seconds treatment, including
  validation/checkpoint work and any compilation. These are not throughput benchmarks.

## Held-out validation

Same fixed windows from the 5k validation games: 145,624 scored positions,
3,431,724 living-piece instances, 181,780 involved-piece targets. Sampled windows
can repeat positions. Both arms use their final step 1,000 checkpoint.

| Metric | Continued involved-only loss | All-living location loss |
|---|---:|---:|
| Living-piece location accuracy | 86.84% | 97.51% |
| Exactly correct living-piece placements | 9,457/145,624 (6.49%) | 95,321/145,624 (65.46%) |
| Positions containing collisions | 26.33% | 26.44% |
| Involved-piece location accuracy | 97.84% | 96.79% |
| All-living location CE | 0.87331 | 0.08716 |

Supervising all pieces strongly improves reconstruction, with a roughly one
percentage point tradeoff on the narrower involved-piece task. Collision rates
are essentially flat despite the large improvement in exact placements:
avoiding overlap and placing every piece correctly are different measurements.

## Traced game, all 74 positions

The trace includes the starting board, every White and Black move, and final mate.
It was not used for training, checkpoint selection, or hyperparameter selection.

| Metric | Continued involved-only loss | All-living location loss |
|---|---:|---:|
| Living-piece locations correct | 1,479/1,676 (88.25%) | 1,636/1,676 (97.61%) |
| Exactly correct placements | 3/74 | 47/74 |
| Positions containing collisions | 13/74 | 19/74 |
| Total squares with overlapping predictions | 16 | 22 |
| Involved-piece targets correct | 95/96 | 95/96 |
| Locations correct when away from starting square | 720/866 (83.14%) | 833/866 (96.19%) |

The last metric is a post-hoc diagnostic: it excludes pieces currently on their
identity's starting square, including pieces that moved and returned there.
It shows the accuracy gain is not confined to unchanged starting placements.

Before `36...g5`, the control gets 8/10 pieces right, with the white king on g1
instead of f3 and white rook on a1 instead of c2. All-living supervision fixes
both but moves the black rook from its actual a6 to c2, overlapping the white
rook. The new head gets 9/10 right with one collision. The black king remains
correct on h4. This is still an imperfect decoder, not a guaranteed valid board.

All scores select living identities using the game record. Captured-piece
detection is not measured, and no board-repair decoding is applied. The policy
is unchanged; this experiment demonstrates a better readout, not stronger play.

## Saved results

- Volume `nanodanya-data`, root `/data/experiments/involved-pieces-20260921`.
- New head: `all-pieces/final.pt`; control: `all-pieces-control/final.pt`.
- Per arm: `baseline.json`, `metrics.jsonl`, `result.json`, `best.pt`, `latest.pt`.
- Full local results and every identity prediction:
  `data/probes/involved-pieces-20260921/all-pieces-result.json`.
- Remote game predictions: `all-pieces-game.json`.
- Run with `MODAL_PROFILE=second modal run --detach modal_involved_pilot.py --action all-pieces`.
- 25 relevant tests passed; Python compilation and whitespace checks passed.
- Existing published viewer still shows the earlier heads; it was not changed
  by this training run. No commits or pushes were made.
