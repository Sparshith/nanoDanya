# nanoDanya State

Last updated: 2026-09-28

The project asks how much chess (legal continuation, tactics, termination) a pure
next-token model internalizes from data and objectives. Runtime scaffolds (legality
masks, EOS suppression, resign heuristics) keep games running but are never evidence
of learned capability; every result should say whether it measures the raw model or
model plus harness.

Model aliases come from `model_registry.py`. Public snapshot of this story:
https://www.sparshith.com/posts/nanodanya.html (numbers verified against
`benchmark/results/` on 2026-07-04).

## Current story

- **Iterative SF2000 RL complete, no established strength gain (2026-09-28).**
  Ten rounds, 10,000 fresh games, 3,000 updates per arm. Refresh scores 48.8%
  versus initial (1,000 games, opening-cluster 95% 46.1% to 51.6%) and 49.2%
  versus fixed archive (1,000 games, 46.1% to 52.2%). SF2000 scores are 17.0%
  initial, 16.7% refresh, 18.9% fixed (500 each). Held-out greedy blunders improve
  11.54% to 10.53%, but this does not translate into demonstrated game strength.
  Final games replay-audited, all training on workspace2. No promotion
  ([protocol](experiments/2026-09-28-iterative-sf2000-rl.md)).

Scale under plain CE is the lever; data source and architecture are second-order
([scale matrix](experiments/2026-05-scale-matrix-h2h-and-stockfish.md)):

- **Source question settled.** At matched 5M scale, actual games vs puzzle-linked
  games is a tie (47.25 / 52.75 H2H). The old "puzzle data is better" story is dead;
  earlier puzzle-model wins were scale in disguise.
- **Registry baseline: the 15M-game model.** The registry default is `plain/games-15m`
  (L16_H8_E1024 best, saved step 148,400, `checkpoints/plain/games-15m/l16_best.pt`), which is
  also `modal_benchmark.py`'s default; the size-matched L12 is `plain/games-15m/l12`.
  The H2H matrix, Stockfish ladder, and blog numbers are from the L12: ~68-70% vs
  both 5M models, and 97% / 78.25% / 42.5% vs sf500/sf1000/sf1500. Architecture is a
  small effect on top: the L16 beats the L12 only 54.45% over 1000 games. The lichess
  bot serves the 10k own-game exact-policy model (deployed 2026-09-19;
  `inference/serve.py`) and
  matchmakes: rated rapid 10+0 challenges to online bots, capped at 5 games/day with
  a 2h gap (`lichess_bot/app.py`); bullet/blitz declined (T4 cold starts would flag).
  On **2026-09-29** a 100-game rated rapid trial (10 concurrent, temperature 0.2)
  started from 1525 rapid; when it completes it archives itself and the 5/day
  schedule resumes. The earlier 0.2 trial reached 1412 -> 1525 over 78 games.
  On **2026-09-19 only**, the user requested a 50-game daily cap with a one-minute
  matchmaking gap/tick. It expires at midnight IST. Counts now persist across
  restarts, reconcile with game history, and include incoming games; one listener
  and one active game at a time. Four games were already played when enabled.
  **Pre-switch public rating: 1418 rapid**, 514 rapid games, rd 45, measured
  2026-09-19. Today's 46 post-switch starts produced 43 completed games
  (11W/7D/25L) and three aborts, finishing at **1409 rapid (-9)**. Four losses
  were on time; logs confirm one failed move submission was swallowed, leaving
  the bot waiting while its clock ran. The delivery fix refreshes server state
  before retrying; live recovery remains unverified until another game occurs.
  On loss GZrvqz3j, the sole move allowing mate in one (36...g5) remains
  top-ranked: 55.29% pre-RL -> 49.80% post-RL at temperature 0.8. The weakness
  predates RL; this position improved insufficiently to change the top choice.
  No live rating improvement established. The 1725 provisional from day one
  was inflation.
- Raw legality scales the same way: 3.9% real illegal (500k) to 0.44% (5M) to 0.24%
  (15M) ([raw legality](experiments/2026-05-raw-legality-by-scale.md)). The 15M model
  is at 99.71% raw-legal, better than every frontier API model tested on the same
  positions; best API model was gemini-3.1-flash-lite at 98.95%
  ([API comparison](experiments/2026-05-27-api-model-legality.md)).
- Errors, both nanoDanya's and the API models', concentrate in middlegame and
  endgame; openings are near-perfect for everyone.

- **Under-10 MB browser baseline works, but loses strength (2026-09-24).** A
  16.3M-parameter student distilled from the L12 exports to a 9.42 MB INT4 ZIP and
  passes browser checker/play tests. In 100-game temperature-0.2 SF1000 tests,
  teacher/float-student/q4-student scored 89%/23.5%/28.5%; q4 scored 9% H2H
  against the teacher. Most loss predates quantization. These are masked game
  scores, not tournament Elo. A separate 1,200-game tournament measured the INT4
  student at **1260 BayesElo (+17/-18)** versus **1768** for the existing INT8
  baseline in the same fit (SF1500 anchored at 1500). No champion replacement
  ([experiment](experiments/2026-09-24-browser-student-distillation.md)).

Settled negative results:

- `weighted/puzzles-5m`: tactical upweighting hurt raw legality, didn't explain gains.
  Checkpoint, registry entry, and `training/train_weighted.py` deleted 2026-07-04
  (plain puzzle controls retrain via `training/train.py`, which ignores weight bins).
- King-safety SFT on `plain/games-5m`: improved the targeted in-check/late slices but
  slightly regressed overall legality; superseded by 15M pretrain
  ([king-safety SFT](experiments/2026-05-19-king-safety-sft.md)). Code and artifacts
  (`sft/`, `benchmark/king_safety_legality.py`, the modal_benchmark mode, volume
  `king_safety_sft/` data + checkpoints) deleted 2026-07-04; findings stand.
- 1-ply value-guided move selection: worse median cp loss and blunder rate than
  greedy ([eval head](experiments/2026-06-16-eval-head-boxed-classification.md)).
- Eval-supervised lines are not strength-competitive (`eval-aware/weighted-ft` lost
  to `plain/puzzles-5m` at 36%). The whole eval-aware line (scripts and checkpoints:
  v1, v2, weighted-ft, eval_boxes_test, ab_test) was deleted 2026-07-04 for a
  rewrite; the findings stand, the artifacts are gone.

What does work besides scale:

- **Position judging.** Boxed win-prob classification (64 bins, cross-entropy) fixed
  the eval-head mean collapse: corr 0.83 with Stockfish win%. The model judges well;
  using that judgment for move selection is the unsolved part.
- **Puzzle-solve eval.** Rating-binned solve rates decline monotonically (full-solve
  77% at 600-700 down to 2.5% at 2700-2800), so the metric discriminates; hard
  puzzles fail on the deep line, not move one
  ([puzzle eval](experiments/2026-06-16-puzzle-solve-eval.md)). GO for the
  tool-assisted phase.

Linear probes support the story from inside: board occupancy and legal-move sets are
linearly readable from hidden states and improve with data
([probes](experiments/2026-05-01-linear-probes.md)).

## Measurement conventions

- **Raw top-1 legality**: is the highest-logit token legal before any mask. Report
  pre-`<eos>` and excluding `<bos>`. `raw_top1_illegal_rate` is strict SAN-token
  illegality; `raw_top1_real_illegal_rate` excludes under-disambiguated SAN (e.g.
  `Ng4` when SAN requires `Nfg4`), a notation failure, not a board-state one.
- **Masked/game results** measure model plus harness, never raw legality.
- H2H at temperature 0 is degenerate (deterministic games collapse); use temp 0.8.
- sf500 is saturated for 5M+ models; discriminate at sf1000/sf1500.
- **Tournament Elo** (challenge leaderboard): BayesElo over games from 200 fixed
  openings with greedy play and 1300 cp adjudication, sf1500 depth 1 pinned at 1500.
  Temperature 0 is fine here because the openings supply variety. The 15M L12 INT8
  CPU export = 1766 +18/-17
  ([tournament Elo](experiments/2026-09-24-tournament-elo.md)). Not comparable to
  temp-0.8 sf-score runs.
- Benchmark files are only comparable when protocol matches. Checkpoints with
  identical arch can be incompatible if their `meta.pkl` tokenizer contracts differ.
  Checkpoint-selection metric matters: `*_best_move.pt` is often more honest than
  best-combined-objective for legality questions.

## Open questions

- Game endings: the model still relies on the max-move guardrail; `<eos>` behavior
  should be learned, not suppressed around.
- Middlegame tactics and conversions are now the real gap (blog framing: the
  problems left are chess problems, not legality problems). sf1500 score 42.5% is
  the number to move.
- Tool-assisted / action-value direction (FEN-snapshot rebuild): superseded
  2026-08-10 before starting; a play-time Stockfish toolcall is a harness under the
  end-to-end rule. Replaced by GRPO on the model itself.
- GRPO v0 (binary puzzle first-move reward) ran end to end 2026-08-16: machinery
  validated, reward misspecified. It generalized on its objective but lost real
  strength (sf1500 53.25 -> 28.75%, H2H 35.5% vs its own init, legality 0.20 ->
  0.59%) ([GRPO v0](experiments/2026-08-16-grpo-puzzle-first-move-v0.md)). Fresh
  same-protocol baseline: the L16 champion is 53.25%/400 games vs sf1500 (the
  blog's 42.5% is the L12 over 200 games, older protocol).
- GRPO v1 (winprob reward over every legal move, 50/50 mixed game/puzzle
  positions, `training/train_grpo.py` + `datasets/mixed/grpo-eval`) fixed the v0
  failure on 2026-08-26 and was run to its ceiling on 2026-08-29 (resumed to
  early stop at 875, best step 625). Final line (`grpo_v1_long_best.pt`):
  puzzle first-move 63.3% / full-solve 45.8% (champion 54.9/38.1), legality
  0.20% (= champion), sf1500 52.1%, H2H 50.1% (both within noise of champion).
  Net: champion strength + ~8 points puzzle tactics, no promotion (bar: sf1500
  clearly above 53.25%). The recipe is harvested: past step ~625 KL creeps
  (0.20 -> 0.27) and dead_groups rises (37% -> 47%, sharp policy = identical
  draws = no signal) with no benchmark movement. Next lever is position
  distribution, not reward: RL on self-play positions with winning and
  late-middlegame plies oversampled, targeting the conversion cliff
  ([GRPO v1](experiments/2026-08-26-grpo-eval-v1.md)).
- Own-game exact policy pilot (2026-09-18) is the first positive strength result
  in this line: 9,449 positions from 320 archived champion-vs-SF1500 games,
  exact legal-conditional reward, 300 full-model updates. Fresh H2H 56.75%/400
  vs champion, estimated +47 Elo (approximate 95% +16 to +79); fresh SF1500
  62.25%/400 vs same-day champion 50.125%/400. Untouched 40-game position test
  blunder probability 14.96 -> 14.21%, with a small raw top-choice legality
  regression. Positive pilot, not a confirmed >=50 Elo or public rating gain.
  No promotion; larger follow-up below. This broader
  recipe succeeds where the tiny stalemate corrections did not, but the causal
  contribution of data, exact feedback, and update budget is not isolated
  ([own-game exact pilot](experiments/2026-09-18-own-games-exact-policy.md)).
- The 10k-source-game follow-up (2026-09-18) improves further: 8,000 training
  games / 231,377 positions, 7,500 updates from the original champion, step 5,500
  selected. Fresh H2H 59.8%/1,000 vs champion, estimated +69 Elo (approximate
  95% +49.6 to +88.9), and 56.2%/1,000 vs the pilot, +43 Elo (+24 to +63).
  SF1500 67.25%/400 vs fresh champion 50.5%/400. Held-out blunder probability
  16.17 -> 14.68%; standard raw legality 9 vs 8 illegal top choices / 4,096,
  but own-game raw top-choice illegality rises 2.60 -> 2.84%. All 2,800 fresh
  matches replay-audited; game tests retain legal masking and score caps as
  draws. Data and update budget both increased, so data-only causality remains
  unresolved. Deployed to the Lichess serving endpoint on 2026-09-19 at the
  user's request; registry baseline remains unchanged. Public-rating gain pending
  ([10k scale run](experiments/2026-09-18-own-games-exact-10k.md)).
- **Exact-10k gain is not established at temperature 0.2 (2026-09-28).**
  The same frozen candidate scored 364W/320D/316L, **52.4%/1,000** against the
  original champion, color-stratified bootstrap 95% **49.90–54.95%**; implied
  H2H Elo +16.7 [-0.7, +34.5]. A small gain remains plausible, but the earlier
  +69 Elo at 0.8 does not carry over as an established gain at 0.2. All games
  replay-audited; 500 per color, three unresolved caps, 949 distinct sequences.
  Checkpoint hashes match the September 23 audit before and after collection.
  Champion metadata confirms saved step 148,400, correcting the old 200k note.
  No retraining or deployment
  ([confirmation](experiments/2026-09-28-exact-policy-temperature-02.md)).
- **Ranking diagnostic (2026-09-28): real but modest greedy improvement.** On
  29,194 previously held-out positions from 999 games, exact-10k changes the
  preferred chess move 10.02% of the time. Greedy >=300cp blunders fall
  13.369% -> 12.814%, -0.555pp (paired game-bootstrap 95% [-0.742, -0.374]):
  421 fixed, 259 introduced, 3,482 retained (89.2% of original errors remain).
  At temperature 0.8, 81.74% of expected-reward gain and 83.22% of blunder-mass
  reduction occur on unchanged-top positions. At 0.2 those shares are 40.85%
  and 44.13%. Thus RL does rerank usefully, but much of its larger 0.8 benefit
  is probability redistribution; this is not proof that sharpening alone causes
  the H2H difference. Source positions were collected at 0.8 and already inspected.
  Raw legality regresses slightly. No training or deployment
  ([diagnostic](experiments/2026-09-28-policy-ranking-diagnostic.md)).
- Mate-threat oversampling pilot completed (2026-09-19): two matched 300-update
  arms from the original pre-RL model, 1,000 training targets from different games.
  Targeted training lowers training mate-blunder probability 6.19% -> 0.63%; on
  299 held-out source games, 4.62% -> 3.75%, versus 4.23% with uniform training.
  Extra reduction -0.478 pp (paired 95% [-0.831, -0.143]); greedy errors 14 -> 12
  versus control 13. H2H 168W/81D/151L, 52.125%/400, Elo +14.8 [-15.7, +45.4]:
  no established strength gain. Strong fitting, modest transfer, some raw-legality
  regression. The g5 diagnostic switches its top move to Kg5 but retains 44.9%
  on g5. No deployment; generalization/coverage is the next diagnostic lever
  ([experiment](experiments/2026-09-19-mate-threat-oversampling.md)).
- Coverage follow-up complete: 2,208 independent training-game threats versus
  1,000, still 300 updates. On 500 fresh threats from a new 2,000-game source,
  mate probability falls 5.176% -> 4.911%, difference -0.265 pp (paired 95%
  [-0.478, -0.064]); greedy errors 16 -> 18, no established change. H2H
  151W/93D/156L, 49.375%/400, Elo -4.3 [-34.3, +25.6]. Small probability
  improvement, no established strength gain. No deployment
  ([experiment](experiments/2026-09-19-mate-threat-coverage.md)). DeepMind's
  Searchless Chess used supervised Stockfish labels and also reported tactical
  blunders/conversion failures; it is not evidence that more RL alone fixes
  these errors ([comparison](experiments/2026-09-19-searchless-chess-comparison.md)).
- Runtime vetoes (blunder-check, stalemate-check) are REJECTED as a direction
  (2026-07-06): the model must be end to end; no rule-based or judge-based harness
  interventions beyond the existing legality mask. The conversion-cliff evidence
  from public games instead motivates training-time fixes: joint policy+boxed-value
  training, eval-as-tokens, or outcome-signal fine-tuning. Metrics to move:
  blunder rate (move-quality cp-300) and stalemate-from-winning rate. Human
  baseline for the blunder metric (2026-08-26, from the stored puzzle-game
  evals, 249M moves): 8.07% overall, 10.6% at 1400-1600 down to 5.5% at
  2800-3000 (log entry 2026-08-26). Champion measured directly the same day:
  7.59% over 8,192 positions (median cp loss 15, best-move match 46.4%), i.e.
  per-move safety like a ~2000 human; the old "11.8%" was eval-aware/v2 on 600
  positions. Confirmed 2026-08-28 on the bot's own 381 lichess games: 10.31%
  there, exactly the 1400-1600 human bin, matching its 1446 rating. The gap is
  compounding: blunder rate is 3.96% in equal positions but 12.47% when winning
  and 15.01% when losing, and rises from 2.3% (ply 1-20) to ~15% (ply 41-80).
  Not blunder frequency in normal positions but conversion is the lever (log
  entry 2026-08-28).
- Eval-as-tokens is untried (from the Feb planning notes): inject discretized eval
  tokens into the move sequence (`<+0.5> e4 <-0.3> e5`) so the model predicts evals
  inline. Candidate objective for the eval-aware rewrite alongside the boxed head.
- Eval-aware training is being rewritten from scratch (old scripts and checkpoints
  deleted 2026-07-04). Building blocks that survive in `training/common.py`:
  `forward_hidden`/`move_logits` for backbone access; the boxed win-prob recipe
  (64 bins, win_k=0.00368208) is in the experiment note and git history.
