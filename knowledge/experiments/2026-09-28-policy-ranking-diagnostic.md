# Does exact-policy RL improve move ranking or mostly change probabilities?

Status: complete. Frozen original champion and own-game exact-10k step 5,500.
User requested this diagnostic after the temperature-0.2 H2H confirmation.
No training, checkpoint selection or deployment.

## Result

RL modestly improves greedy move quality, but most of the expected-reward gain
at temperature 0.8 occurs without changing the preferred chess move. This supports
probability redistribution as a large part of the observed gain, not the stronger
claim that RL failed to learn better rankings at all.

| >=300cp blunder measure | Original | Exact-10k | Change, percentage points | Paired game-bootstrap 95% |
| --- | ---: | ---: | ---: | --- |
| Greedy legal top move | 13.369% | 12.814% | -0.555 | [-0.742, -0.374] |
| Expected blunder probability, temperature 0.8 | 16.175% | 14.676% | -1.500 | [-1.576, -1.426] |
| Expected blunder probability, temperature 0.2 | 13.558% | 12.872% | -0.686 | [-0.803, -0.571] |

Greedy blunders decrease from 3,903 to 3,741: **421 fixed, 259 introduced,
3,482 retained**. Thus 89.2% of original greedy blunders remain, and introduced
errors offset 61.5% of fixes. This is a real but small net improvement on these
archived positions, not an estimate of full-game Elo.

The preferred chess move changes on **2,924/29,194 positions (10.02%)**.
Among changed choices, 1,196 improve by more than 30cp, 962 worsen by more than
30cp, and 766 stay within 30cp. There are 31 additional annotation-only token
changes. Reward-based grading finds 1,240 improved and 986 worsened changed
choices; reward saturation creates more ties than raw CP grading.

Greedy near-best rate (within 30cp) increases 52.028% to 52.326%, +0.298pp,
95% [+0.048, +0.554]. Best representable engine move is ranked first at
11,097 positions before and 11,129 after; mean rank improves 4.332 to 4.292.

### Where the probability improvement occurs

| Partition | Positions | Contribution to total reward gain at 0.8 | Contribution to total blunder-mass reduction at 0.8 |
| --- | ---: | ---: | ---: |
| Preferred chess move unchanged | 26,270 (89.98%) | 81.74% | 83.22% |
| Preferred chess move changed | 2,924 (10.02%) | 18.26% | 16.78% |

These are signed contributions over disjoint sets of positions, not causal
attribution. An unchanged top move can still have changes in lower-ranked moves.
At temperature 0.2 the unchanged-top partition contributes 40.85% of the reward
gain and 44.13% of the blunder-mass reduction; ranking changes matter more once
sampling is already concentrated.

At 0.8 mean probability on each model's greedy chess move rises 61.96% to 67.37%,
while mean move entropy falls 1.136 to 0.960 nats. At 0.2 the corresponding
top-move masses are 89.78% to 91.44%. This supports policy concentration alongside
some reranking, without establishing that RL is equivalent to temperature scaling.

Descriptive slices retain the same direction: greedy blunders in positions whose
best move is >+200cp decrease 15.77% to 15.21%; after ply 80, 13.59% to 13.32%.
Those weaknesses remain substantial. Slice intervals were not calculated.
Raw top-token illegality including EOS rises 2.548% to 2.812%, +0.264pp,
95% [+0.157, +0.368]; all quality metrics above condition on legal moves.

## Interpretation and next decision

The original hypothesis is partly supported. RL improves average move ranking,
but modestly, and its larger high-temperature benefit is dominated by safer
probabilities on unchanged-top positions. The majority of old decisive errors
persist, and new errors erase much of the correction benefit. Lower-temperature
play already avoids many alternatives that the RL objective learned to suppress.

This helps explain why large temperature-0.8 game gains need not carry over to
0.2. It does not prove the mechanism of the 52.4% H2H result: these positions
come from the old champion-vs-SF1500 source, not the fresh low-temperature H2H
state distribution, and engine labels approximate consequences under engine play.

Next diagnostic training proposal: compare the existing expected-reward loss
against explicit engine-preference/ranking supervision, holding initialization,
data, update budget and evaluation temperature fixed. Track fixed AND introduced
greedy blunders and fresh 0.2 full-game strength. The diagnosis motivates this
control but does not establish that changing the objective will work. No such
training run was launched.

## Verification

All 29,194 legal engine tables, reward transformations, normalized token
histories, unique identities and test-game membership passed. 128 histories
fully replayed; production support checks and 256 greedy comparisons passed.
There were two unsupported legal-action occurrences, retained at zero action
probability while preserving the full legal best score for blunder labels.
Optimized/native float32 forward maximum logit difference was 1.526e-5, within
the fixed 1e-4 tolerance. Checkpoint hashes match prior audits and remained
unchanged. No gradients or optimizer steps. Three focused metric tests passed.
Local analysis independently reconstructed classifications and aggregate
decompositions and bootstrapped complete source games. Remote elapsed time:
356.8 seconds. Differences from old BF16 training-evaluation counts should not
be interpreted as checkpoint changes; this comparison scores both models in FP32.

## Fixed diagnostic

Use all 29,194 positions from 999 games in the original exact-10k test split.
This split was held out from RL training but its metrics were inspected in prior
experiments. It is development evidence, not a newly collected blind test.
Use the original complete 50k-node engine action tables without relabeling.

- Compare production greedy decisions: highest legal token logit, then map the
  token to a chess move. Count check/mate suffix-only changes separately.
- Grade changed choices as materially better/worse when their engine scores
  differ by more than 30cp; also report reward changes and near-equal choices.
- Track greedy >=300cp blunders fixed, retained and introduced, near-best moves
  (within 30cp), and rank of the best representable engine move.
- Measure expected reward, >=300cp blunder mass, move entropy and raw legality
  at both 0.8 and 0.2. Positive temperature scaling leaves greedy token rank
  unchanged. Sum suffix variants for stochastic move probabilities.
- Partition aggregate changes into positions whose top chess move changes and
  positions whose top move stays unchanged. Contributions sum to the total;
  this decomposition is descriptive, not causal mediation.
- Primary uncertainty: paired source-game bootstrap, 10,000 samples, seed
  2026092802, position-weighted ratios. Intervals exclude training-seed variation.
- Verify checkpoint and test-file hashes, source-game split membership, all
  engine legal-action sets, all normalized token histories and reward transforms.
  Fully replay 128 evenly spaced histories and compare production action support
  and greedy choices. Check optimized final-position logits against native forward.

Inputs are archived original-champion/SF1500 positions collected at temperature
0.8. Findings do not directly measure the new state distribution from 0.2 H2H.
CP labels and sigmoid saturation retain their original limitations. Best-engine
rank uses the best representable action; blunders use the best full legal score.

Local reports and source snapshot: `data/probes/policy-ranking-20260928/`.
Remote report: `/data/experiments/policy-ranking-20260928/result.json`.
Runner: `scratch/modal_policy_ranking.py`.
Metrics/analysis: `scratch/policy_ranking_metrics.py`, `scratch/analyze_policy_ranking.py`.
Cloud run: https://modal.com/apps/sparshithsampath/main/ap-vzuaEs8HbYcgAnbu7wCUgk

Three focused tests cover probability improvements without rank changes,
notation-only changes, and production top-token versus aggregated-move choices.
An initial launch was stopped to correct history validation: archived histories
omit SAN suffixes while stored training prefixes use canonical annotated SAN.
The stored prefixes remain the model input; no dataset or recipe was changed.
