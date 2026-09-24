# RL checkpoint confirmation at temperature 0.2

Status: complete. All 3,000 predeclared games and position scoring finished.
Checkpoint, data and replay audits passed. No retraining or deployment.

## Result

**A playing-strength gain from these RL checkpoints is not established at 0.2.**
Both primary confidence intervals include 50%. Small positive gains remain
possible, but the earlier roughly 56% match scores at 0.8 cannot be assumed to
hold under the new evaluation standard.

| Comparison, left model's score | W/D/L | Score | 95% CI | Caps |
| --- | --- | ---: | --- | ---: |
| Joint after RL vs own before | 393/229/378 | 50.75% | 48.25–53.25% | 0 |
| Control after RL vs own before | 407/221/372 | 51.75% | 49.20–54.35% | 2 |
| Joint after RL vs control after RL | 416/192/392 | 51.20% | 48.55–53.85% | 2 |

Every matchup has 1,000 games, 500 per color. Score counts draws as half a point.
Caps were counted as draws; all-loss/all-win cap assignments give point-score
bounds of 51.65–51.85% for control versus its initialization and 51.10–51.30% for
the final joint/control matchup. Caps do not explain the inconclusive outcome.

There is a large color effect in this low-temperature sample. Joint-after scored
34.4% as White and 67.1% as Black against its initialization. Control-after scored
39.0%/64.5%, and joint-after versus control-after scored 35.5%/66.9%. Colors were
balanced and uncertainty stratified by color. The cause of this pattern was not
investigated in this confirmation run. Matchups contained 782, 823 and 719 unique
complete move sequences respectively; repeated sequences were retained as
independent sampled games, not selectively removed.

| Position measure | Joint before → after | Control before → after |
| --- | --- | --- |
| Mate-in-one blunder probability, 458 held-out traps | 3.82% → 3.21% | 3.81% → 3.21% |
| ≥300cp blunder probability, 1,000 ordinary positions | 5.70% → 5.40% | 5.72% → 5.40% |
| Raw illegal-token probability, ordinary positions | 2.36% → 2.70% | 2.34% → 2.68% |
| Original g5 probability | 98.63% → 99.73% | 98.58% → 99.73% |

Joint's held-out trap probability changed by −0.611pp (95% paired game-bootstrap
CI −1.280 to −0.046), and control's by −0.600pp (−1.256 to −0.046).
The difference in changes was −0.0107pp (−0.0344 to +0.0125), giving no clear
extra benefit from joint. Ordinary blunder probability fell by −0.304pp
(−0.546 to −0.088) for joint and −0.315pp (−0.558 to −0.096) for control.
These are modest position-level improvements on the reused held-out set,
originally sampled from games played at 0.8. They did not establish an improvement
in fresh full games played at 0.2. Ordinary-position mate-blunder changes remain
inconclusive. Raw legality deteriorated slightly in both arms.

Greedy mate errors remain 17/458 before and 14/458 after for both arms, exactly as
at 0.8. Positive temperature scaling does not change the favorite token.
The g5 move remains the greedy choice throughout and became more probable after
RL. This run supports treating the trained models as experimental candidates,
without claiming a demonstrated strength upgrade at temperature 0.2.

## Verification and artifacts

- Independently replayed all 3,000 fresh games; verified results, terminations,
  balanced colors, per-shard seeds, exact counts, model identities and game IDs
  disjoint from the previous 5,800 recorded games.
- Verified all four checkpoint hashes against the original training receipts
  and the original dataset manifest hash. No weights, labels or training recipe
  changed. All paired test row identities matched the original held-out set.
- Verified greedy mate and raw top-token legality decisions were invariant to
  temperature for every test row. Recomputed g5 probabilities agree with the
  original trace's cached 0.2 calculations within 1e-5.
- Six local checks passed, including the new check comparing temperature-scaled
  evaluation probabilities with the normal inference loader. Existing gradient,
  save/load, microbatch and resume checks also passed.
- Local `result.json`, `positions.json`, `summary.json`, `audit.json` and
  `code-manifest.json` are under `data/probes/board-rl-t02-20260924`.
  Analysis: `scratch/analyze_board_rl_t02.py`.
- [Cloud run](https://modal.com/apps/workspace2/main/ap-TQ7DHtPpQ1eFWV65GVumF8).


## Protocol

- Reuse the four frozen September 23 board-RL checkpoints: joint/control before
  and after RL. No retraining, checkpoint selection, or changed reward labels.
  These checkpoints were trained at 0.8; this experiment changes evaluation only.
- Primary comparisons: each post-RL arm versus its own pre-RL initialization,
  **1,000 fresh games each**, 500 per color. Secondary: post-RL joint versus
  post-RL control, another **1,000 games**, 500 per color. Stop at these counts
  regardless of results. All players use temperature 0.2, legal masking, EOS
  disabled, KV caching, 200-ply cap. Seeds: 2026092400 + matchup index * 10000
  + 50-game shard offset. No new comparison to historical 0.8 match scores.
- Report score (win=1, draw=0.5), W/D/L, per-color scores, 95% bootstrap intervals
  stratified by color (10,000 draws), and unresolved caps with worst/best bounds.
  The two primary arms' intervals are per-comparison, not simultaneous intervals.
- Rescore the same 458 training-held-out mate traps and 1,000 ordinary positions
  at 0.2, FP32, for all four checkpoints. Use paired source-game bootstraps.
  This is reuse of an already inspected held-out set, not a new tactical test.
  Greedy choices should be invariant to positive temperature scaling.
- Recompute the known g5 diagnostic at 0.2. It remains excluded from training.
  Board readouts do not depend on sampling temperature and are not refitted.
- Verify checkpoint/data hashes, replay every fresh game, validate seeds, colors,
  results and unique game IDs, and check paired held-out row identities.

Remote root: `/data/experiments/board-rl-t02-20260924` on workspace2.
Local root: `data/probes/board-rl-t02-20260924`.
Runner: `scratch/modal_board_rl_t02.py`. Shared evaluation helper now accepts an
explicit temperature; the original trainer's default and recipe remain 0.8.
