# Faster piece head with frozen-policy warmup

Completed on workspace2 H100, 2026-09-21. The fresh head was undertrained in the
first pilot. A faster head reads the previously missing black king and rook
from the unchanged champion. Joint training provides little evidence of better
underlying representations in this pilot.

## Protocol

Same champion, dataset, seed, batch size, validation windows, trace, and
independent readout procedure as [the first pilot](2026-09-21-involved-piece-pilot.md).

- Warmup: 1,000 updates, frozen policy, head LR `1e-3`.
- Joint: load both warmed weights, fresh optimizer, 1,000 updates,
  policy LR `3e-6`, head LR `1e-3`.
- Piece loss weight stays `0.1` throughout. No extra empty-square targets.
- Warmup policy weights verified bitwise identical to the original champion.
- Joint batches match the previous next-move-only control. Final checkpoints
  are compared, without choosing a checkpoint on the traced game.
- Training took 127 seconds for warmup and 237 seconds for joint updates.
- Run: https://modal.com/apps/workspace2/main/ap-u20DNRda9VvvwrnJFosJR7

## Held-out training metrics

| Configuration | Involved-piece location accuracy | Captured-piece accuracy | Next-move loss |
|---|---:|---:|---:|
| Previous head, LR 3e-6 | 44.57% | 24.58% | 1.40024 |
| Faster head, frozen champion | 97.90% | 94.74% | 1.40572 |
| Warmed head, joint training | 98.18% | 95.91% | 1.39310 |
| Previous next-move-only control | Untrained head | Untrained head | 1.39308 |

Accuracy counts only supervised pieces, with no credit for empty squares.
The frozen-to-joint head improvement also includes 1,000 additional head
updates. Without a continued-frozen arm, it cannot be attributed to backbone
updates alone. The next-move loss matches the control; playing strength was
not tested.

## Direct identity-head readout on GZrvqz3j

At the same 36 black-to-move positions, the frozen and joint heads both locate
all 49 move-involved piece instances correctly. Across all known living
identities, including pieces outside the supervised objective, the frozen head
gets 643/823 (78.13%) and joint gets 660/823 (80.19%).

Before `36...g5`, the direct head locates:

| Piece | Actual | Frozen champion + fitted head | Joint model + head |
|---|---|---|---|
| Black king | h4 | h4 | h4 |
| Black rook | a6 | a6 | a6 |
| White king | f3 | h4 | h4 |
| White rook | c2 | a6 | c2 |

The other six pieces are correct in both cases: 8/10 frozen and 9/10 joint.
Both heads can assign multiple identities to the same square. Ground truth
selects living identities for scoring; this does not measure captured-piece
detection or full-board validity. These scores are not interchangeable with
the independent square-based readout scores below.

The frozen model already supports a readout of the missing black king and
rook. Their absence from the old visualization does not establish that the
model lacks their locations. The white-king error also shows why high accuracy
on move-involved pieces is insufficient to establish complete board tracking.

## Independent representation comparison

Refit the same independent linear board probes on separate games. Score
correct colored pieces, without credit for empty squares.

| Fixed layer | Original champion | Next-move control | Warmed joint model |
|---|---:|---:|---:|
| 11 | 752/823 | 750/823 | 753/823 |
| 12 | 738/823 | 736/823 | 739/823 |
| 15 | 721/823 | 718/823 | 718/823 |

The joint model's validation-selected layer is 11: held-out piece recall
86.07%, traced-game recall 91.49%. At the critical position this probe still
recovers 8/10, omits the rook on a6, and decodes h4 as a white pawn. Differences
of one to three recovered pieces across the trace do not establish a useful
representation gain with one seed and one game.

Use the better-fitted identity head alongside the existing board probe for
inspection. Before scaling training, evaluate non-involved pieces and full-board
consistency on additional held-out games.

## Artifacts and checks

- Volume: `nanodanya-data`, `/data/experiments/involved-pieces-20260921`.
- Checkpoints: `head-warmup/final.pt`, `head-warmup-joint/final.pt`.
- Local results: `data/probes/involved-pieces-20260921/head-warmup-result.json`
  and `head-trace.json`.
- Direct-head evaluation app:
  https://modal.com/apps/workspace2/main/ap-azMVHhiipY2S4MEMGUyQd3
- Tests cover frozen-policy invariance, warmed-head initialization, distinct
  learning rates, causal trace prefixes, and pre-move identity/capture alignment.
  The existing 16 relevant tests passed; all five probe tests passed after
  adding the direct-head trace check.
- Original champion unchanged. Nothing deployed, committed, or pushed.
