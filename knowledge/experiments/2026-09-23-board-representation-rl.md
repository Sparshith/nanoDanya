# Does better board representation make outcome-based RL more effective?

Status: complete. Protocol below was fixed before collection or training. Both
arms finished all 1,500 updates and the full evaluation. No deployment, commit,
or push was performed.

Follow-up: [evaluation at temperature 0.2](2026-09-24-board-rl-temperature-02.md)
did not establish a full-game strength gain for either arm. The match results
below apply to temperature 0.8.

## Result

RL improved average tactics and playing strength in both arms, but the model
with better board reconstruction did not benefit more. The original g5 mistake
became more likely. This pilot does not support scaling this recipe on the
assumption that a better board representation will make move reasoning click.

| Measure | Joint, before → after RL | Control, before → after RL |
| --- | --- | --- |
| Immediate-mate blunder probability, 458 held-out traps | 4.64% → 3.95% | 4.64% → 3.95% |
| Greedy mate blunders on those traps | 17/458 → 14/458 | 17/458 → 14/458 |
| ≥300cp blunder probability, 1,000 ordinary positions | 6.97% → 6.21% | 6.97% → 6.22% |
| Raw illegal-token probability, ordinary positions | 3.16% → 3.45% | 3.15% → 3.44% |
| Fresh-probe exact board reconstruction, development set | 73.57% → 71.08% | 71.69% → 69.56% |
| Original g5 probability | 72.13% → 79.31% | 71.93% → 79.24% |

Move probabilities use temperature 0.8 and legal conditioning, except the
explicitly raw legality metric. g5 remains the greedy move in every checkpoint.
The known trace was excluded from all RL datasets. Board-probe reconstruction
is position-weighted and uses the same 200 development games and freshly fitted
3,000-step linear heads before and after RL.

The primary paired change was −0.691 percentage points for joint (95% game
bootstrap CI −1.028 to −0.379), versus −0.687 for control (−1.025 to −0.375).
Joint minus control change was **−0.0046pp, CI −0.0217 to +0.0143pp**. Both
improved; there is no measurable additional tactical benefit from joint in this
run. Greedy-error reduction alone is inconclusive, with its change CI spanning
zero. Ordinary-position immediate-mate probability barely changed, and its CI
also spans zero. Raw legality deteriorated slightly in both arms; full games
still enforce legal moves at runtime.

| Full-game comparison, left model's score | Games | W/D/L | Score | 95% CI | Unresolved caps |
| --- | ---: | --- | ---: | --- | ---: |
| Joint before vs control before, held-out source games | 1,200 | 480/213/507 | 48.88% | 46.33–51.42% | 11 |
| Joint after vs control after | 1,000 | 400/189/411 | 49.45% | 46.65–52.20% | 18 |
| Joint after vs own before | 400 | 189/69/142 | 55.88% | 51.38–60.25% | 8 |
| Control after vs own before | 400 | 189/73/138 | 56.38% | 52.00–60.75% | 4 |

Score counts wins as 1 and draws as 0.5. Caps are scored as draws but remain
unresolved; treating all caps as losses or wins gives score bounds of
48.55–50.35% for the final joint/control matchup, 54.88–56.88% for joint versus
its initialization, and 55.88–56.88% for control versus its initialization.
The joint/control score change was +0.58pp (CI −3.16 to +4.33pp), also
inconclusive. Confidence intervals condition on these fixed checkpoints and
sampled games; they do not measure variation across training seeds.

Joint retained a 1.52pp exact-board reconstruction advantage after RL, so the
negative result cannot simply be attributed to complete loss of that measured
advantage. However, the initial advantage was modest, the two policies were
already extremely similar, and this is one shared training seed on a fixed
dataset. It does not establish that representations can never help RL, nor does
it isolate representation quality from all other differences between checkpoints.

## Run and verification

- Collected 4,000 games and graded 115,490 positions. Removed 659 candidate
  occurrences shared across splits. Training used 69,244 positions, including
  1,878 eligible mate traps; source-arm counts were 34,458 joint and 34,786 control.
  Validation contained 400 ordinary positions and 161 distinct-game traps.
  Test contained 1,000 ordinary positions and 458 distinct-game traps.
- Both runs used identical batch sequences, final sampling hash
  `6486e1f3dd47eeeeab06db6fce31f595a676344cb514c5c0a44b88d70d98c1a7`.
  Dataset manifest SHA256:
  `2a549ed80134a043e8fb2c9a3fa7cc5d653c6e4e2443fce19f6e7de5f3d2ae5f`.
- Independently replayed all 5,800 games; verified unique game IDs, outcomes,
  termination reasons, colors, checkpoint lineage and evaluation seeds.
  Checked 2,019 positions and 117,177 board-prefix states against chess replay,
  including all held-out mate targets. Cross-split board/game disjointness,
  trace exclusion and paired test identities passed.
- All test legal moves are representable in the fixed tokenizer. Seven
  non-test legal-action occurrences use unsupported rare queen-disambiguation
  strings (`Qd5e6`, `Qf2e1`, `Qe2f1`); as in the existing policy trainer they have
  zero action probability. Both arms share this limitation.
- Frozen references remained unchanged; both board encoders updated. Saved
  models passed parity checks through the normal inference loader. Fresh
  readouts verified unchanged backbones and identical head initialization,
  data, seed and fitting budget across comparisons.
- 32 local checks passed, including gradient flow, microbatch equivalence,
  save/load and exact interrupted/resumed training equivalence. Cloud
  preemptions during preparation were recovered from saved grading results;
  neither RL recipe nor training budget changed.
- Final checkpoint hashes: joint
  `d05f014512b3e7954d04919b30bbbbcdd2e3afbdddadfa40553046ac8df7a3bb`,
  control `7b0116a250064fd6e37717d9192cd8fe61cc1fb730edfbe933f81b653ad76eb0`.

Local results include `dataset.json`, `train.json`, `evaluate.json`, `summary.json`,
`audit.json`, `dataset-audit.json`, held-out reward tables and `code-manifest.json`
under `data/probes/board-rl-20260923`. Analysis: `scratch/analyze_board_rl.py`.
Cloud runs: [collection](https://modal.com/apps/workspace2/main/ap-Hnn5TAkaxWK42nTICkUNem),
[data](https://modal.com/apps/workspace2/main/ap-F4RDkWAdLL6GdOYufdTyoG),
[training](https://modal.com/apps/workspace2/main/ap-UAJHPuwqPdKNPXqjBgOl55),
[evaluation](https://modal.com/apps/workspace2/main/ap-al4WtL15A5zMG0rYN7LbFB).


## Fixed protocol

- Initial checkpoints are the matched September 22 square-joint `joint` and
  `control` arms. Both receive their normal correct board inputs. Joint previously
  had modestly better fresh-probe reconstruction; their playing strength did not
  clearly separate. Hashes: joint `038d7df248493977dbc21eaf5c88aed2a3c5042b8e02ecfb09d3a5dc2e49010c`,
  control `30ed280ab9c8eb9d110bb4efbe0bb34ff745c59c4bc3e7082672c606dd26214e`.
- Collect **4,000 fresh H2H games** between these fixed starting models,
  temperature 0.8, legal masking, EOS suppressed, KV cache, 200-ply cap.
  For 2,000 games sample joint's own positions and for 2,000 sample control's;
  each source arm has 1,000 games per color. Seed family 2026092300 + shard offset.
  This pooled source avoids choosing all training states from one arm.
- Split within source-arm/color strata by seeded hash: **60% train / 10%
  validation / 30% test**, 2,400/400/1,200 whole games. Up to 32 random own-turn
  positions per game from ply 16. Remove positions with the same first four FEN
  fields across splits. Exclude pre-g5, post-g5 and post-Rh2 trace boards from all
  training/evaluation datasets; inspect the known trace only after evaluation.
- Grade every retained position's full legal action set with Stockfish, 50,000
  total MultiPV nodes, one thread, cleared hash, engine binary hash recorded.
  Keep existing sigmoid/clamp rewards and terminal overrides. Exhaustively label
  moves allowing mate in one and override their score to -10000. Both arms use
  identical tables. Fail incomplete legal tables rather than filling labels.
- Select at most one eligible threat per held-out game, up to 500 per split,
  by fixed hash before model inference. Eligibility requires a fatal move and a
  nonfatal alternative valued at least -200cp. Also select one ordinary position
  per held-out game, up to 1,000. Report actual counts without changing collection
  size based on outcomes. These are genuinely fresh source games for this run.
- Both arms: **1,500 fixed AdamW updates**, effective batch 64, LR 3e-6,
  weight decay 0.01, gradient clip 1, temperature 0.8, KL coefficient 0.05 to
  that arm's own frozen initialization. Train the whole policy and board encoder.
  No auxiliary square loss during RL. Use the existing half-uniform/half-priority
  sampler, identical batches and seed 20260923. Training data stays fixed.
- Fixed final-step comparison; no early stopping, checkpoint selection or
  hyperparameter retries. Evaluate validation periodically for diagnostics.
  Evaluate test once all training is complete. Save resumable optimizer/RNG state.
- Primary tactical comparison is **difference in before/after changes** between
  arms on the same held-out mate targets. Also report ordinary expected reward,
  >=300cp blunder mass, greedy mate errors, raw legality and KL. Paired bootstrap
  over source games, 10,000 draws. Report individual improvements and between-arm
  differences; do not infer representation causality merely from one arm improving.
- Fresh full games: **1,000 post-RL joint versus post-RL control**, plus **400
  post-RL versus own pre-RL games per arm**, balanced colors, temperature 0.8,
  200-ply cap. Pre-RL joint/control comparison uses the fixed 1,200 held-out source
  games, converting source orientation to joint perspective. Report uncertainty
  and unresolved caps separately. No sample extension based on interim scores.
- Refit the same fresh direct-square probe for each post-RL checkpoint using
  the previous readout protocol and 200 development games to assess retention of
  recoverable placement. These readout games are already inspected development
  data, distinct from the fresh tactical test. No readout metric selects RL weights.
- One training seed and modest initial representation difference make this a
  pilot. Failure to separate arms will be reported as inconclusive at this budget,
  not proof that representation cannot help RL.

Remote root: `/data/experiments/board-rl-20260923` on workspace2.
Local root: `data/probes/board-rl-20260923`.
Implementation and checks: `scratch/board_rl_core.py`,
`scratch/modal_board_rl.py`, `scratch/test_board_rl.py`.
