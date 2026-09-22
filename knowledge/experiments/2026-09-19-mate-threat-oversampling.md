# Mate-threat oversampling versus uniform training

Status: complete. Modest held-out mate-safety transfer; no established Elo gain. No production change.

Question: does oversampling diverse positions with an avoidable mate-in-one
blunder improve unseen source games, beyond equal-budget ordinary training?

- Use the existing exact-10k archive (231,377 train / 28,901 validation / 29,194
  test positions). Preserve whole-game splits and exclusion of shared boards.
  Start both arms from original `plain/games-15m`, which has not been fine-tuned
  on this archive. Pretraining overlap is unknown. These are previously used
  aggregate evaluation splits, not a newly collected independent test set.
- Exhaustively enumerate legal moves and immediate opponent checkmates. Fatal
  moves get exact cp=-10000 overrides in both arms; otherwise retain existing
  engine winprob rewards. Require a nonfatal alternative with engine cp >= -200
  for the target pool. Avoiding immediate mate alone need not save the game.
- Before model inference, select by seeded hash, one position per source game:
  at most 1,000 training / 250 validation / 500 test targets. Minimum viable counts
  100 / 30 / 30. Reserve all descendants of a source game together; no synthetic
  positions in this first test. Ordinary validation/test probes: up to 1,000 each,
  also one position per game. No selection by model probability.
- Both arms: 300 full-model AdamW updates, LR 3e-6, beta 0.05, temperature 0.8,
  batch 64, same seed and frozen reference, exact legal-conditional engine reward.
  Control samples 64 archive positions uniformly. Targeted samples 32 uniformly
  plus 32 from the fixed target pool. Sampling is the sole arm difference.
- Fixed step 300 for evaluation, no checkpoint selection. Curves at 0/50/150/300
  on training targets, validation targets, and ordinary validation. Evaluate test
  only after training. Report mate-blunder probability, greedy mate-blunder rate,
  ordinary reward / cp-300 blunder probability / legality / KL. Paired source-game
  bootstrap compares target and control test changes. One seed is a pilot limit.
- Finish with 400 fresh balanced H2H games of targeted versus control, temperature
  0.8 and 200-ply cap. Report caps separately; this screens relative strength,
  not a precise Elo estimate or preservation versus the deployed 7,500-step model.

Scripts: `scratch/mine_mate_threats.py`, `scratch/train_mate_threats.py`.
Volume dataset: `/data/datasets/own-games/mate-threats-20260919`.
Checkpoints: `/data/checkpoints/grpo/mate-threats-20260919/{control,targeted}`.

## Mining and implementation checks

Completed app `ap-kCbeZgEnXKL6iABdG16XJs`. Exact scans found 12,645 / 1,585 /
1,753 positions with at least one mate-in-one blunder in train/validation/test.
Of these, 4,290 / 503 / 573 have the required playable alternative. The fixed
one-per-game selections contain 1,000 / 250 / 299 positions, so test size is 299,
not the cap of 500. All game and cross-split board-disjointness assertions pass.
Only three fatal moves in training had old engine scores above -1000 cp; none
in validation or test did. Missing immediate-mate punishment is therefore rare
in this archive. Full counts/hashes are saved in the dataset manifest.

Sixteen focused tests pass, including exact-label comparison to exhaustive reply
search, color symmetry, matched sampler RNG/budget, and a two-step training,
checkpoint/reload/resume/evaluation smoke test. The two first GPU launches failed
before initialization because the runner's default PYTHONPATH omitted the project
root; relaunched with explicit project and nanochat paths. No updates or candidate
outputs were made by the failed launches. Active runs: control
`ap-BriQYse8LI08S3SFjuVO7k`, targeted `ap-rvGqFvIuvpjHWghmizWDvr`.
Fresh H2H seed fixed before results: 27319000, 400 games, eight shards.

## Fixed-budget position results

Both 300-step arms completed. Identical baseline distributions verified across
arms. Test contains 299 selected threats and 999 ordinary positions, all
one-per-game. Paired game bootstrap: 10,000 draws, seed 2026091902.

| Mate-in-one blunder probability | Original | Uniform control | Half-targeted |
| --- | ---: | ---: | ---: |
| Training targets (1,000) | 6.1931% | 5.6176% | 0.6257% |
| Validation targets (250) | 5.7458% | 5.4643% | 4.7948% |
| Test targets (299) | 4.6170% | 4.2266% | 3.7485% |

Targeted minus control test difference: -0.4781 percentage points, paired 95%
[-0.8308, -0.1433]. Targeted versus original: -0.8684 pp, 95%
[-1.3100, -0.4645]. This supports modest transfer on this selected position
population; uncertainty excludes training-seed variation. Greedy mate choices
are 14/299 original, 13/299 control, 12/299 targeted. The greedy difference is
not established (target minus control CI includes zero).

Ordinary test expected reward: 0.621644 original, 0.624604 control, 0.624931
targeted. cp-300 mass: 16.8501% -> 16.0300% / 15.9061%. Target-control ordinary
reward/blunder differences have intervals including zero. Raw illegal probability
mass rises with targeting (3.3404% control vs 3.6992% targeted, +0.3589 pp,
95% [+0.2477, +0.4763]); raw top-choice illegality including EOS is 19/999
original, 18/999 control, 25/999 targeted. Existing legal filtering remains on
in game evaluations. No inference-time mate veto is added.

The selected examples cover both colors (test 149 Black / 150 White), all six
blundering piece types, median ply 50 and 21 pieces on test boards. Queen/rook
mating replies dominate. This is source-game transfer, not a held-out tactical
motif test. Frozen known example GZrvqz3j is checked separately after training,
for illustration only; it does not select checkpoints or change the protocol.

Artifacts: data/traces/grpo/mate-threats-20260919/{manifest,selection_audit,
position_summary,control_test,targeted_test,control_curves,targeted_curves}.json.
H2H app: ap-CmCzPCAssI6K1yVVsqFUdz.

## Fresh games and known-position diagnostic

All 400 fresh H2H games replay-audited, 200 games per color, no duplicate IDs or
missing game numbers. Targeted versus equal-budget uniform control:
**168W / 81D / 151L**, score **52.125%**. Approximate independent-game 95% score
interval [47.749%, 56.501%], Elo difference +14.8 [-15.7, +45.4]. No demonstrated
strength gain or proof of equivalence. Six 200-ply caps counted as draws; scoring
all six as losses/wins bounds score at 51.375% / 52.875%. Benchmark file:
benchmark/results/h2h_mate_threats_targeted_control_20260919_400.jsonl.

Exploratory reconstruction of GZrvqz3j, using the production T4 inference path at
temperature 0.8 (outside the selection metrics): original pre-RL g5 55.291% /
Kg5 41.146%; control g5 61.254% / Kg5 35.836%; targeted g5 44.920% / Kg5 51.843%.
The targeted checkpoint now picks Kg5 greedily but still assigns almost 45% to
the mating blunder. This diagnostic was known before the study and is not an
independent test or a basis for checkpoint selection. Probe apps
ap-Q39bYn9a9rYPikr3kzxGsL and ap-8tCYZ7rYRW9d6RuKnjAg77 completed.

## Interpretation

The optimizer can strongly reduce the targeted error on examples it repeatedly
sees: training mate probability falls about 90%. There is also measured source-
game transfer, with an extra 0.478 percentage-point reduction over uniform
training on the 299 threats. However, held-out improvement is much smaller,
most greedy mating mistakes persist, and full-game strength did not separate
clearly. This supports limited generalization as a bottleneck, not a claim that
targeted training is ineffective or that the architecture cannot learn tactics.
Raw-legality regression remains a tradeoff. One seed, selected threat distribution,
and reused prior evaluation splits limit broader inference.

Next useful test: expand distinct threat-position coverage at a matched update
budget before increasing repetition of the same 1,000 examples. Keep a fresh
held-out set for that follow-up because this test set has now been inspected.
No promotion, live-bot change, commit, or push. Both fixed final checkpoints and
optimizer-resume states remain on the Modal volume. Small reports, distributions,
and audit: data/traces/grpo/mate-threats-20260919/. Scratch scripts are disposable;
all substantive findings are recorded here.
