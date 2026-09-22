# Exact policy improvement on the champion's own positions

Status: complete. Positive pilot strength result; no production promotion.

This tests a broader practical recipe after the small stalemate corrections
failed to establish transfer. It does not isolate data scale from update budget,
parameter scope, or learning rate.

- Source: 400 archived champion-versus-SF1500 games,
  `benchmark/results/sf1500_plain_games15m_l16_400.jsonl`. These games now supply
  training data; the old 53.25% score is not a fresh evaluation of this candidate.
- Seed 20260918. Whole-game split: 320 training / 40 validation / 40 test games.
  At most 32 random own-turn positions per game, starting at ply 16. Board states
  appearing across splits are excluded. Counts: 9,449 / 1,067 / 1,160 positions.
- Every legal move graded by full-strength fairy-stockfish, 50k total nodes per
  position with MultiPV, cleared hash, terminal overrides. Store complete tables,
  search depth, source/engine hashes, and reject incomplete move coverage.
- Reward retains v1's sigmoid and +/-1000cp clamp. Objective is exact expected
  reward conditional on legal tokens, plus raw-policy KL to the frozen champion.
  All SAN check/mate annotation variants receive the move's reward.
- Train from original champion, all parameters, 300 updates, AdamW LR 3e-6,
  batch 64 (8 x 8 accumulation), temperature 0.8, KL coefficient 0.05. Half the
  sampled positions are uniform; half come from winning positions or playable
  positions where the recorded move lost at least 300cp. No action sampling.
- Evaluate the entire fixed validation set initially, at step 1, every 25 steps,
  and at the end. Select highest conditional reward subject to raw KL <= 0.2 and
  illegal probability no more than 0.005 above baseline. Keep champion if no
  eligible improvement. Report legality, all-best-token mass, and cp-300 mass.
- Read the test split only after selection. Final strength evaluation: 400 fresh
  H2H games vs champion, seed 20260919; 400 fresh SF1500 games each for candidate
  and champion, seed 20260920. Temperature 0.8, 200-ply cap, 8 shards, SF 20ms,
  existing legality mask and EOS suppression. Report unresolved caps separately.

Dataset: `/data/datasets/own-games/exact-v1-20260918`.
Training output: `/data/checkpoints/grpo/own-games-exact-v1-20260918`.
Code: `data_prep/prepare_own_game_rewards.py`, `training/train_exact_policy.py`.

Validation before launch: 19 focused tests passed, including engine reward
perspective for both colors, full legal coverage, complete-game splitting,
annotation aggregation, and batched exact-gradient parity. A tiny CPU integration
run completed training, selection, checkpoint reload, and test evaluation without
changing its source checkpoint.

## Position results

Grading completed with no missing legal moves; mean minimum MultiPV depth 12.31.
The fixed 300-update budget completed successfully. Every evaluated checkpoint
passed the predeclared KL/illegal-mass eligibility bounds; step 300 had the highest
validation reward and was selected. Validation reward improved 0.729705 to
0.732027 and legal-conditional cp-300 blunder probability 20.20% to 19.39%.

Untouched test split, 1,160 positions from 40 separate games:

| Metric | Champion | Selected step 300 |
| --- | ---: | ---: |
| Conditional expected reward | 0.605658 | 0.608512 |
| Conditional probability on best-reward moves | 53.85% | 54.78% |
| Conditional cp-300 blunder probability | 14.96% | 14.21% |
| Raw illegal-mapping probability | 3.772% | 3.799% |
| Raw top choice outside legal mapping | 2.155% | 2.672% |
| Raw KL to champion | 0 | 0.02012 |

The raw illegality measures here include EOS and are not the project's usual
real-illegal metric excluding notation errors/EOS. Best-reward mass includes all
token variants and reward ties, including ties created by reward saturation.

Paired bootstrap resampling complete test games (10,000 draws, seed 20260918)
gives a 95% interval of -0.918 to -0.606 percentage points for the blunder-mass
change, and +0.00210 to +0.00380 for expected reward. This is evidence of transfer
on the engine-scored objective in this archived-game distribution, not an Elo
interval. Raw top-choice illegality worsens by six positions (25 to 31).

Small reports are in `data/traces/grpo/own-games-exact-v1-20260918/`:
manifest, protocol, metrics, test, and test_uncertainty JSON. Model weights and
optimizer remain on the Modal volume. Apps: preparation `ap-G4saJ6KYTpDpEphQBEdRWp`,
training `ap-KLpCILuOzYtqPTG3BlUOmD`.

## Fresh game results

| Match | Wins / draws / losses | Score |
| --- | ---: | ---: |
| Candidate vs champion | 194 / 66 / 140 | 56.75% |
| Candidate vs SF1500 | 230 / 38 / 132 | 62.25% |
| Fresh champion vs SF1500 | 185 / 31 / 184 | 50.125% |

H2H implies +47.2 Elo against the champion under this protocol. Approximate
independent-game 95% interval: +16.1 to +79.1 Elo (score 52.32–61.18%). This is a
positive pilot result, not confirmation that the true improvement is at least
50 Elo, a public Lichess rating gain, or replication across seeds/training runs.
The SF1500 score difference is +12.125 percentage points, approximate
independent-game 95% interval +5.60 to +18.65 points. Timed Stockfish searches can
vary; H2H provides the separate strength check without that opponent.

The existing game harness scores 200-ply caps as draws: 4 H2H caps, 12 candidate
SF caps, and 7 champion SF caps. These are unresolved games, not verified chess
draws. Assigning all H2H caps as candidate losses or wins gives score bounds
56.25–57.25%, so they do not explain the observed advantage. All results use the
existing legality mask and EOS suppression. Raw top-choice legality regressed
slightly on the position test and remains a separate concern.

All 1,200 fresh games were replayed locally. Legal histories, unique game
numbers, terminal outcomes/winners, caps, and aggregate scores verified. Evidence:
`benchmark/results/h2h_own_games_exact_v1_20260918_400.jsonl`,
`sf1500_own_games_exact_v1_20260918_400.jsonl`, and
`sf1500_champion_exact_control_20260918_400.jsonl`. Reconstructed scores and
uncertainty are in the trace directory's `game_audit.json`.

Interpretation: broader own-game feedback plus exact policy improvement produced
both engine-objective transfer and a full-game improvement in this pilot. The
run changes several factors relative to the old experiments, so it does not
identify which change caused the gain. The budget ended while validation still
improved slowly; no ceiling is established. Next useful step is independent
replication before promotion. Champion serving, registry, and bot are unchanged.
