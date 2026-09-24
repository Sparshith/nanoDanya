# Why exact-policy RL did not eliminate the g5 failure

The strongest evidence is limited generalization from a sparsely covered, fixed
training distribution. The objective already assigns credit to the preceding bad
decision and strongly penalizes g5. A new frozen-policy audit finds substantial
training-set improvement and smaller held-out improvement. No weights were updated.

## Checkpoint lineage matters

The deployed own-games exact-10k model is the original 15M champion plus RL,
selected at step 5,500. The September 22 board-input model instead starts from
the original champion and receives next-move CE; square-joint continues that
board-input branch with next-move and board losses. The latest board-readable
checkpoint has not received the deployed RL treatment. Its roughly 65% g5
probability cannot be interpreted as regression after that RL.

For the actual RL comparison, g5 probability at temperature 0.8 is 55.2915%
before and 49.8037% after RL. It remains the top move. The original full-game
evaluation established an overall improvement: 59.8% score versus champion in
1,000 games. Failure to eliminate this tactic is not absence of all RL benefit.

## The objective already punishes the earlier move

`training/train_exact_policy.py` optimizes exact expected engine reward over all
representable legal actions, plus 0.05 times full-vocabulary KL to the original
champion. It samples positions, not actions. A bad move need not be sampled for
its reward to affect the gradient. There is no sampled-GRPO dead-group issue in
this trainer, and the outcome of the recorded game is not the reward.

Regraded g5 using the existing `prepare_own_game_rewards.grade` implementation,
50,000 total MultiPV nodes, cleared hash and terminal overrides. This diagnostic
uses local Stockfish 17.1; the archive used Fairy-Stockfish with a different hash.
It is a methodology-matched diagnostic, not reproduction of an archived engine
search. Exact mate-in-one is independently established by legal reply enumeration.

| Move | Diagnostic cp, Black perspective | Training reward |
| --- | ---: | ---: |
| g5 | -9999 (mate sentinel) | 0.024553 |
| Kg5 | +246 | 0.712138 |
| g6 | +203 | 0.678625 |
| Kh5 | +275 | 0.733523 |

At the deployed checkpoint, differentiate the real loss with respect to its
output logits, using the original champion as reference, temperature 0.8 and
beta 0.05. Sum gradients over all g5 suffix variants, equivalent to shifting
their logits together:

- Reward-loss gradient: **+0.213402**.
- Weighted KL gradient: **-0.003627**.
- Combined gradient: **+0.209776**.

Gradient descent would lower g5 logits. KL offsets only **1.70%** of this local
reward signal. The reward clamp does not remove the g5-versus-safe-move distinction.
This is a local output-space diagnostic on a position absent from the archive;
it does not establish how a full-model optimizer step would generalize, nor
exclude KL effects across the entire training distribution.

The stored exact threat labels were also checked against original archive
scores: just **3 fatal moves** in the training archive were above the -1000cp
reward floor. A widespread failure to label immediate mate is unsupported.

## Reconstructed training exposure

Verified hashes for all three original dataset splits and training threat labels.
Replayed the trainer's actual half-uniform/half-priority sampler, seed 20260919,
in unchanged training-file order through 7,500 steps. The selected checkpoint is
step 5,500, with **352,000 position presentations** over 231,377 available positions.
It encountered 171,481 distinct positions, averaging 1.52 presentations per
available position. The exact g5 board is absent from train, validation and test.

| Training category | Available positions | Distinct encountered by step 5,500 | Presentations |
| --- | ---: | ---: | ---: |
| Any move allows immediate mate | 12,645 | 7,845 | 13,563 |
| Mate blunder with a nonfatal alternative scored at least -200cp | 4,290 | 3,313 | 7,069 |
| Pawn move allows rook mate, at most ten pieces | 196 | 100 | 146 |
| At most ten pieces, only kings/pawns/rooks, at least one rook, and playable alternative | 24 | 20 | 42 |

The last row also requires an immediate-mate blunder. These descriptive material
categories were chosen after seeing g5; they are not a predeclared generalization
test or proof that matching examples are necessary. They show how sparse relevant
endgame coverage is despite the headline 10,000 source games. Playable mate threats
account for only **2.01%** of sampled position presentations.

**12.26%** of sampled presentations have identical rewards for every legal move,
mostly from the +/-1000cp clamp. They offer no reward-based preference gradient,
though KL can still act. This is training inefficiency, not the direct cause of
the g5 error. All saved checkpoints passed the selection gates; step-5,500
validation KL was 0.05676 versus a 0.2 limit, so selection was not halted by that
gate. Validation reward largely plateaued after step 2,500.

The archive contains games of the original champion against SF1500. It is fixed
during training. There is no repeated collection of the updated model's own
mistakes, no on-policy refresh, and no explicit value or terminal-state objective.
SAN '+' and '#' variants receive the same action reward, and terminal boards are
excluded from position sampling. The wrong check/mate suffix and EOS behavior
found in the earlier probe therefore are not directly supervised by this loss.

## New matched transfer evaluation

Both original and deployed RL checkpoints evaluated together on H100, float32,
temperature 0.8, production legal support and EOS suppressed. No fitting or
checkpoint selection. Use the 1,000 previously fixed training mate targets and
the 500 targets collected in the September 19 coverage follow-up. Each set has
one position per game. The latter is disjoint from RL training, but has already
been inspected in earlier experiments and is development evidence, not a new
untouched final test. Source file hashes and game/ply pairing were verified.

| Metric | Original | Deployed RL |
| --- | ---: | ---: |
| Mean mate-blunder probability, training targets | 6.184% | 2.349% |
| Greedy mate blunders, 1,000 training targets | 52 | 17 |
| Mean mate-blunder probability, 500 held-out threats | 6.840% | 5.315% |
| Greedy mate blunders, 500 held-out threats | 25 | 21 |

Held-out probability change: **-1.525 percentage points**, paired game-bootstrap
95% **[-2.086, -0.997]**, 10,000 resamples, seed 20260923. Greedy change is -0.8pp,
interval [-2.0, +0.4], so greedy improvement is not established. Of the 25 original
greedy mistakes, RL fixes seven, retains eighteen, and introduces three elsewhere.
Training mate probability falls 62.0%, held-out probability 22.3%. These are
different position sets, so the magnitude gap is descriptive rather than an
isolated causal estimate of memorization. Intervals omit training-seed uncertainty.

Existing interventions strengthen the generalization diagnosis. Half-targeted RL
reduced training mate probability 6.193% to 0.626%, but old held-out targets only
4.617% to 3.749%. Increasing distinct targets from 1,000 to 2,208 at matched
budget gave only another 0.265pp on the fresh test and no demonstrated H2H gain.
Therefore, simply repeating examples or modestly widening the same archive has
already been tried and did not solve the class of failure.

## Decision

The evidence supports correct local reward direction plus incomplete transfer,
with sparse relevant coverage in a fixed archive. It does not isolate architecture,
optimization, regularization, or data coverage as the sole cause. The earlier
suggestion to assign feedback to the preceding bad move is already implemented.

The next useful controlled experiment would refresh examples from the model
being improved and compare that with continued training on the old archive at
matched budget, from the same checkpoint. Evaluate broad decisive errors on
held-out source games and preserve full-game strength. This is a proposal, not
a claim that on-policy refresh is proven to solve the problem. The board-input
branch would require a deliberate matched RL experiment before attributing any
benefit to combining its representation changes with outcome training.

Reproduction: `MODAL_PROFILE=sparshithsampath modal run --detach
scratch/modal_rl_failure_audit.py` after generating the local reward table.
Data and verification: `data/probes/rl-failure-audit-20260923/`, including
`g5-reward-table.json`, `archive.json`, `signal-and-transfer.json`, `summary.json`.
Summary checks independently recompute grouped analytical policy gradients,
metric means, game pairing, sampling totals and bootstrap intervals. No optimizer
step was taken. Compilation and whitespace checks passed. Remote run completed:
https://modal.com/apps/sparshithsampath/main/ap-0tdlCpUEzfLbdxzF8f0ah5
