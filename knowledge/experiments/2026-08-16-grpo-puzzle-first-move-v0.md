# GRPO v0: Puzzle First-Move Reward Makes a Puzzle Specialist, Not a Stronger Player

Date: 2026-08-16.

## Setup

Direction: RL the champion itself instead of the abandoned tool-assisted phase
(play-time Stockfish calls are a harness under the end-to-end rule). v0 recipe:
GRPO on puzzle positions with binary first-move reward.

- Data: `datasets/puzzles/grpo`, 490k train / 10k val positions from puzzles_raw,
  first 100k games in file order excluded (benchmark eval zone). Prep:
  `data_prep/prepare_grpo_positions.py`.
- Trainer: `training/train_grpo.py`. K=8 draws at temp 0.8 from the puzzle-position
  distribution, reward 1/0 vs the correct token set (solution move + mating
  alternates, expanded to all +/# annotation variants; the champion vocab is
  annotated, 14,750 tokens), advantage = reward minus group mean, exact
  full-distribution KL to the frozen champion, kl_beta 0.05, lr 1e-5, batch 64.
- Init and comparison model: `plain/games-15m` (L16 champion).
- Checkpoint benchmarked: `checkpoints/grpo/pilot/grpo_pilot2_best.pt` (step 75 of a
  300-step pilot; val reward plateaued there at 0.70, up from 0.50).

Training dials all passed: reward up 40%, KL plateaued ~0.5-0.65, illegal mass flat
~2.6%, entropy 1.03 -> 0.55 without collapse. The dials measure puzzle positions at
temp 0.8; they did not predict what follows.

## Results (all 2026-08-16, A10G, same-day protocol, benchmark/results/)

| metric | champion | GRPO v0 |
|---|---|---|
| puzzle first-move acc (8,800 held-out) | 54.9% | 66.4% |
| puzzle full-solve | 38.1% | 47.9% |
| full-solve, 600-700 bin | 76.7% | 95.3% |
| full-solve, 2700-2800 bin | 2.5% | 2.0% |
| raw top-1 real-illegal (4,096 pos) | 0.20% | 0.59% |
| vs SF1500, 400 games | 53.25% | 28.75% |
| H2H vs champion, 400 games | — | 35.5% |

Files: `puzzles_grpo_pilot2_400`, `legality_{grpo_pilot2,plain_games15m_l16}_4096`,
`sf1500_{grpo_pilot2,plain_games15m_l16}_400`, `h2h_grpo_pilot2_vs_champion_400`.

## Reading

- The gains are real and generalize: +11.5 first-move / +9.8 full-solve on held-out
  puzzles, concentrated in easy/mid bins; the hard tail is unmoved. Full-solve
  improving while only move one was rewarded means the first-move distribution
  sharpening carries through short lines.
- The costs are larger: -24.5 vs SF1500, 35.5% H2H against its own init, legality
  3x worse (middlegame/endgame concentrated). Puzzle positions are a biased slice
  (someone always has a forcing win); optimizing there pulls probability mass
  toward forcing moves in quiet positions where they are wrong.
- Champion baseline correction: L16 vs sf1500 is 53.25% over 400 games under
  today's protocol, not the 41% from the old 50-game file; the blog's 42.5% was
  the L12 over 200 games. Cross-day protocol comparisons stay unreliable; the
  GRPO comparison here is same-day, same-code.

## Verdict

The GRPO machinery works (stable, cheap, generalizes on its objective). The v0
reward is misspecified for strength: binary puzzle first-move only. Champion stays
champion; the bot is untouched. Next levers, in order: mix ordinary game positions
with eval-delta rewards (the ndjson has per-move Stockfish evals) so quiet-position
play is rewarded too; stronger KL / earlier stop to map the puzzle-gain vs
strength-cost frontier; keep the specialist checkpoint as a story artifact (best
puzzle solver trained here).
