# Why the policy prefers 36...g5

Frozen-checkpoint diagnostic of GZrvqz3j after 36.Kf3. The policy strongly
predicts the mating rook move once g5 is supplied, but usually calls it check
rather than mate and remains uncertain about termination. This localizes a
tactical consistency failure; it does not identify the internal circuit or prove
why the preceding g5 logit is high. No training or deployment was performed.

## Main measurements

All table probabilities use temperature 1. Move probabilities use the existing
legal mask with EOS suppressed and sum SAN suffix variants. Raw token probabilities
below use the entire vocabulary, without masking or EOS suppression. The two
board-input checkpoints receive correct board inputs throughout normal runs.

| Checkpoint | P(g5) before move | P(Rh2, any suffix) after g5 | Raw P(Rh2+) | Raw P(Rh2#) |
| --- | ---: | ---: | ---: | ---: |
| Original game checkpoint, own-games exact 10k | 47.82% | 89.28% | 71.94% | 17.15% |
| Pre-RL champion | 51.32% | 82.79% | 57.83% | 24.65% |
| Board-input checkpoint | 63.12% | 81.90% | 56.81% | 24.85% |
| Latest square-joint checkpoint | 64.98% | 81.23% | 60.23% | 20.76% |

The game checkpoint's g5 probability at production temperature 0.8 is 49.8037%,
reproducing the earlier T4 probe within 0.0004 percentage points. Latest joint
at temperature 0.2 assigns 98.6330% to g5; after g5 it assigns 99.9925% to Rh2.
Low temperature sharpens the wrong initial preference.

At the root, the latest model has 99.904% raw mass on the production legal set.
Raw g5 probability is 64.917%, versus 64.982% after masking. Thus the mask is not
manufacturing this preference by eliminating substantial probability elsewhere.

## Does it recognize the consequence?

Feed the model's preferred token Rh2+ after g5. It produces the correct resulting
piece placement in the board-input interface, although the token understates mate.
Raw next-token probabilities are:

| Checkpoint | EOS | Kg6 | Kg5 |
| --- | ---: | ---: | ---: |
| Original game checkpoint | 32.93% | 32.17% | 6.06% |
| Pre-RL champion | 38.70% | 35.99% | 8.43% |
| Board-input checkpoint | 51.62% | 25.31% | 6.08% |
| Latest square-joint checkpoint | 44.01% | 31.55% | 4.76% |

The game is checkmate. Kg6 moves the king two squares from h4; Kg5 lands on its
own pawn. These are diagnostic raw predictions after a terminal position, not
moves that the gameplay harness would execute. Because Rh2+ mislabels a mate,
this tests the model's own preferred continuation, not ordinary canonical SAN.
It is evidence of fragile consequence/termination recognition, not a general
measurement of king-movement knowledge.

As a nearby control, after g6 Rh2+ the latest model gives the genuinely legal
escape Kg5 99.918% raw probability. After f5 Rh2+ it gives Kg5 99.862%.
This rules out a blanket inability to predict a king escape from this rook check.

Supplying Rh2# explicitly makes EOS nearly certain for champion, served, and
oracle, but only 57.48% for joint, which puts 39.08% on the illegal Kh5. The '#'
token supplies the answer, so EOS after it is not independent evidence of mate
recognition. Removing the suffix gives other nonsensical continuations; suffix
removal is also atypical input and is not used for the headline interpretation.

## Board state and nearby positions

Existing fresh square probes, with checkpoint hashes verified, recover all ten
pieces exactly before and after g5 for both oracle and joint. After Rh2+, readout
quality worsens, including a wrong rook color. The same rook errors also appear
in the g6 Rh2+ control where move prediction is correct. These readout errors do
not isolate the cause of the mate failure or prove what board the policy believes.

Enumerated all 16 legal replacements for White's preceding move, replaying each
from the initial board. Among the seven king moves, only the actual Kf3 allows
mate in one after g5. Latest model P(g5) is 64.98% after Kf3 but 37.44% after Kd3
and 27.49% after Kf2. Stockfish depth 18 values g5 after the six alternate king
moves at +414 to +494 cp for Black; actual Kf3 g5 is mate in one against Black.
Thus g5 is useful in neighboring positions, while the policy fails on the exact
exception. Overgeneralization of a pawn-push pattern is a plausible hypothesis,
not a causal conclusion from these altered positions.

Five legal move-order transpositions reach exactly the same six-field FEN.
Latest joint still prefers g5 in all five, with probability 60.96% to 66.17%.
The preference survives these changes to recent history.

Blanking all board inputs lowers joint P(g5) from 64.98% to 51.96%. Changing only
the last board's white king or removing its rook changes the preference little.
These last-board interventions deliberately conflict with the retained history
and use the original mask, so they cannot establish that those pieces are ignored.

Stockfish 17.1, depth 20 and all 15 root moves, independently finds g5 uniquely
allows immediate mate. Kg5, Kh5 and g6 retain evaluations above +300 cp for Black.
This was not a choice among uniformly losing moves. Scores are search estimates.

## Interpretation and reproducibility

The strongest evidence is a mismatch between the high probability of g5 and the
strong conditional prediction of its mating reply. Finding Rh2 after g5 does not
establish that the root forward pass anticipates it, and probability of a move
is not a value estimate. The current inference path does one next-token prediction
and samples; it does not explicitly evaluate candidate replies and use their
consequences to revise the root move scores. The probes support unreliable mate
recognition and decision consistency, not a proven explanation of the internal
features producing the g5 logit or a general failure of piece-movement rules.

Commands:

```sh
MODAL_PROFILE=second modal run --detach scratch/modal_g5_diagnostic.py
MODAL_PROFILE=sparshithsampath modal run --detach scratch/modal_g5_diagnostic.py --arms served
.venv/bin/python scratch/audit_g5_diagnostic.py
```

Full local data: `data/probes/g5-diagnostic-20260923/`. Four checkpoint JSONs
contain all distributions, logits on legal support, raw top tokens, legal branches,
counterfactuals and histories. Separate engine JSONs contain engine identity, root
search settings and principal variations. `audit.json` records file/checkpoint
hashes and independent probability regrouping, position, mate and readout checks
for all 298 saved distributions. Compilation and whitespace checks passed.

Remote data: `/data/experiments/g5-diagnostic-20260923/` in the corresponding
account's nanodanya-data volume. Latest/readout run:
https://modal.com/apps/workspace2/main/ap-DHNtZlfVXSriuGyb8i83i5
Original game checkpoint run:
https://modal.com/apps/sparshithsampath/main/ap-JfbpbGp0OY7LEa3FyEO2Jc

Initial attempts hit an import packaging error and a checkpoint absent from the
workspace2 volume. Both were corrected before the reported successful runs; served
was loaded from the original account. All GPU jobs have completed.
