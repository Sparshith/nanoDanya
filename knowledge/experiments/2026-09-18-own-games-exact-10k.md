# Own-game exact policy improvement: 10,000-game scale run

Status: complete. Estimated +69 Elo vs the original champion and +43 vs the
400-game pilot, each over 1,000 fresh games. No promotion.

User requested scaling the successful 400-game pilot to 10,000 source games.
This scales both position diversity and training budget, not an isolated data-only
ablation. Start from the original champion, not the pilot's improved checkpoint.

## Frozen recipe

- Generate 10,000 fresh champion-vs-SF1500 games, temperature 0.8, 200-ply cap,
  20ms engine searches, seed 27100000, 100 shards of 100 games. Eight A10G workers.
  The existing legal mask/EOS suppression remain. Save each completed shard.
- Split whole games 80/10/10 using seed 20260919: 8,000 train, 1,000 validation,
  1,000 test. Up to 32 own-turn positions per game starting at ply 16. Exclude
  board states shared across splits. Position counts depend on generated games.
- Preserve the pilot's complete legal-move reward tables, 50k MultiPV nodes,
  terminal overrides, sigmoid/clamp, and half-uniform/half-priority sampling.
  Grade on CPU with 16 workers; grading can resume completed records.
- Original champion initialization and reference, all parameters, AdamW 3e-6,
  exact legal-conditional expected reward plus raw KL 0.05, temperature 0.8,
  effective and physical batch 64 (validated by a memory-only smoke test). Budget
  7,500 updates, 25 times the
  pilot budget, approximately the same number of position presentations per
  available training position. No learning-rate or reward sweep.
- Fixed validation subset: up to four random positions from every validation
  game, selected by history identifiers before model inference. Evaluate initially,
  at step 1, every 250 steps, and finally. Select best reward subject to the same
  KL <= 0.2 / raw illegal-mass <= baseline + 0.005 guards. Persist optimizer,
  sampling state, metrics, and model for resumption without resetting the reference.
- Evaluate the complete untouched position test after selection. Fresh matches:
  1,000 H2H vs champion (seed 27200000), 1,000 vs the 400-game pilot (27210000),
  and 400 vs SF1500 each for candidate and champion (27220000). Eight shards,
  same temperature/cap/engine search budget. Report unresolved caps separately.
  Evaluate standard raw legality on 4,096 positions as a separate regression check.
- No promotion or bot change is part of this run.

Source: `benchmark/results/own_games_source_10k_20260918.jsonl` plus its `_shards/`
directory. Volume source: `datasets/own-games/source-10k-20260918.jsonl`.
Dataset: `/data/datasets/own-games/exact-10k-20260918`.
Training: `/data/checkpoints/grpo/own-games-exact-10k-20260918`.

## Infrastructure checks

The first collection attempt (`ap-dxuEIKjqRDlY6oDAnNE4iu`) encountered a worker
preemption and two engine-response timeouts during games. Its local aggregate
failed before saving game records, and the app was stopped. Those partial games
are not counted in the 10,000-game dataset. The replacement preserves completed
shards, retries failed remote calls at most twice, reserves two CPU cores per
GPU worker, and increases the response watchdog from 10s to 60s. The engine's
search budget remains 20ms. This changes infrastructure, not the reward recipe.

Twenty-five focused tests passed. An interrupted/resumed two-step integration
run reproduced continuous training weights and metrics bit-for-bit on CPU.
Tests also check missing-shard recovery, rejection of mismatched seeds, complete
game numbering, partial grading recovery, and fixed validation sampling.

The replacement collection is `ap-Rm9LZFw9Zmp61vqerWrMbW`. A memory-only A10G
check (`ap-ZwlO4YWBT9WmodDvkuxatJ`) successfully executed a full AdamW update
with physical batch 32, effective batch 64, and all histories at the maximum
200-token length. Peak allocated memory was 10.15 GiB, reserved 10.53 GiB.
No weights were saved. A second, two-update check at physical batch 64
(`ap-pACzl9VsAvOCMIoofy86fZ`) passed with persistent AdamW state: allocated
18.05 GiB, reserved 18.48 GiB. The scale run therefore uses physical batch 64
without accumulation; reward, effective batch, learning rate, and update budget
stay fixed. The disposable check script is `scratch/check_exact_batch_memory.py`.

## Source audit

All 10,000 games replayed legally, with matching outcomes and terminations,
unique identifiers, complete numbering, and 5,000 games per color. Champion
results: 4,507 wins, 1,089 draws, 4,404 losses (50.515% score). Of the draws,
338 hit the 200-ply cap. Source SHA-256:
`b9ca89668a153cc34a874699c24a846eeb51de6ff7e7524a8bda53c024e85fbb`.
Audit: `data/traces/grpo/own-games-exact-10k-20260918/source_audit.json`.
CPU grading app: `ap-iCzrku2mmeq7wtp66XqA9t`.

After cross-split board exclusion: 231,377 train / 28,901 validation / 29,194
test positions. Fresh champion controls finished: SF1500 181W/42D/177L,
50.5% score, 12 unresolved caps; all 400 games replay-audited. Standard raw
legality: 8 real-illegal top choices / 4,096 positions (0.195%).

Grading completed without missing legal tables: mean minimum MultiPV depth 12.295.
Priority positions: 148,803 train / 18,765 validation / 18,341 test. Training app:
`ap-m3KSwjOSY5ylzqYC8NWOYp`.

### Vocabulary support correction before training

The first training launch stopped before any update: 27 training positions
contained legal SAN spellings absent from the fixed vocabulary, primarily
promoted-piece disambiguation. The same issue affects 3 validation positions and
1 test position. No position has zero playable tokens. CPU structural audit
`ap-aFSqmdpWjEXkjzyXESS6Gd` inspected vocabulary coverage only, not model scores.

Removed the overly strict all-SAN-in-vocabulary assertion. Full engine tables
and all positions remain; expected reward sums over representable legal tokens,
exactly matching the game harness. Unrepresentable moves have zero probability.
The diagnostic best-mass metric means best representable move; blunder labels
still compare against the best engine score over all legal moves. No vocabulary,
reward, or hyperparameter change. The pilot had no such affected positions.
A test checks exact agreement with inference action support and rejects an
empty support; all 26 focused tests pass. The empty failed output directory was
removed after confirming it contained no artifacts. Replacement training app:
`ap-F3QCjTQcsV0c6wvGi249Ao`. Coverage report:
`data/traces/grpo/own-games-exact-10k-20260918/vocabulary_audit.json`.

## Training selection

All 7,500 updates finished. Step 5,500 selected on 3,988 fixed validation
positions: expected reward 0.624304 -> 0.630463, blunder probability 16.958% ->
15.257%, KL 0.05676, illegal probability mass 3.821% -> 3.894%, raw top-choice
illegality 2.457% -> 2.859%. All checkpoints passed the predefined eligibility
limits. Improvements were small after step 2,500; later checkpoints never beat
step 5,500. The test and fresh matches use that selected checkpoint.

## Results

| Fresh match | Games | W / D / L | Score | Estimated Elo difference (approximate 95%) |
|---|---:|---|---:|---|
| Candidate vs original champion | 1,000 | 501 / 194 / 305 | 59.8% | +69.0 (+49.6 to +88.9) |
| Candidate vs 400-game pilot | 1,000 | 465 / 194 / 341 | 56.2% | +43.3 (+24.0 to +62.9) |
| Candidate vs SF1500 | 400 | 250 / 38 / 112 | 67.25% | |
| Fresh champion vs SF1500 | 400 | 181 / 42 / 177 | 50.5% | |

All 2,800 games replayed legally with matching outcomes, balanced colors,
unique IDs, and complete numbering. Legal masking/EOS suppression are active.
Unresolved 200-ply caps are scored as draws: 29 vs champion, 9 vs pilot, 9
candidate-SF, 12 champion-SF. Treating all candidate H2H caps as losses or wins
bounds observed score at 58.35–61.25% vs champion and 55.75–56.65% vs pilot.
Intervals use independent-game standard errors, not public-rating uncertainty.
One champion H2H worker hit a checkpoint-read OSError before playing; automatic
retry succeeded. No failed attempt or incomplete game was counted.

Untouched test: 29,194 positions from 999 of the 1,000 held-out games (one had
no retained position). Expected reward 0.614319 -> 0.619790; legal blunder
probability 16.174% -> 14.676%, change -1.498 percentage points, game-bootstrap
95% [-1.573, -1.425]. Best representable-move mass 50.686% -> 52.082%, KL 0.05485.
Raw top-choice illegality including EOS worsens 760 -> 828 / 29,194 (2.603% ->
2.836%); illegal mass 3.834% -> 3.884%. Standard legality on the same 4,096
human-game positions is nearly unchanged: 8 -> 9 real-illegal top choices,
legal probability mass 98.340% -> 98.438%. These are distinct distributions
and legality definitions.

This supports a strength gain from scaling this recipe. Both data diversity and
update budget increased; it does not isolate the effect of more games alone.
The point estimate exceeds +50 Elo over the champion, but the approximate lower
95% bound is +49.6, so a strict >=50 Elo claim is not established. No vocabulary
expansion, bot deployment, registry change, commit, or push was performed.

Artifacts: `data/traces/grpo/own-games-exact-10k-20260918/` holds manifest,
protocol, metrics, test, test_uncertainty, vocabulary_audit, source_audit, and
the four match audits. Full games are in `benchmark/results/` under
`h2h_own_games_exact_10k_{champion,pilot}_20260918_1000.jsonl`,
`sf1500_own_games_exact_10k_20260918_400.jsonl`, and
`sf1500_champion_exact_10k_control_20260918_400.jsonl`. The selected checkpoint
is `/data/checkpoints/grpo/own-games-exact-10k-20260918/best.pt` on Modal volume
`nanodanya-data`. The live champion is unchanged.
