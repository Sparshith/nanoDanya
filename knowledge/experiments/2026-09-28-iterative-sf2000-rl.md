# Iterative SF2000 RL, 2026-09-28

Status: running, not yet a result. User requested ten rounds and selected 1,000
games per round. No production promotion or checkpoint selection by evaluation.

Start from `checkpoints/grpo/own-games-exact-10k-20260918/best.pt`, saved step
5,500, SHA256 `4c42e87b3217f40482d9bf28f60a8c32bf2d7d04fddf90472f390b4b14e1e53a`.
Each refresh round collects 1,000 games against Fairy-Stockfish 14 with
UCI_LimitStrength and UCI_Elo=2000, 20 ms per turn. This is a nominal engine
setting, not a calibrated human Elo claim. There are 100 fixed training openings,
five games as each color per opening, model temperature 0.8, legal mask on,
EOS off, and a 200-ply total limit. Unresolved caps score as draws and are reported.

Sample up to 32 own-turn positions per game starting at ply 16. Grade every legal
move with the unrestricted engine using 50,000 total MultiPV nodes per position.
Use the existing exact legal-conditional expected-reward objective plus raw-policy
KL. Reward is sigmoid(0.00368208 * clamp(mover_cp, -1000, 1000)). Half the training
batch is uniform and half winning or playable recorded blunders. There are no
supervised best-move targets.

Train 300 updates per round, batch 64, microbatch 16, AdamW learning rate 3e-6,
weight decay 0.01, KL coefficient 0.05, training temperature 0.8, gradient clip 1.
The reference model stays fixed at the initial checkpoint. Optimizer and RNG state
carry across rounds. Round one is shared; the fixed-data control then retains the
round-one archive for its remaining 2,700 updates. Both arms receive 3,000 updates.
Every round uses its final checkpoint, with no early stopping or winner selection.

Evaluation uses 50 separate openings with disjoint ECO codes and ending positions.
A frozen probe samples up to eight positions from each of 200 initial-model games;
its board states are excluded from training. Probe metrics include greedy blunders,
expected reward and blunder probability at temperatures 0.2 and 0.8, illegal mass,
and KL. After each round, play 100 SF2000 games per arm at temperature 0.2.
After round ten, play 1,000 H2H games refresh versus fixed and 1,000 refresh versus
initial, plus 500 SF2000 games for each of initial, refresh, and fixed. All final
games use held-out openings and temperature 0.2. Final uncertainty should bootstrap
whole openings, retaining both colors and repetitions together.

All model weights, datasets, and training remain on the workspace2 Modal workspace
(local profile `second`), volume `nanodanya-data`
under `experiments/iterative-sf2000-20260928/`. The volume stores the frozen protocol,
source hashes, per-game records, graded datasets, optimizer resumes, per-round
checkpoints and probe reports. Small local receipts and the frozen source archive
live in `data/probes/iterative-sf2000-20260928/`.

Implementation: `scratch/modal_iterative_policy.py` orchestrates the existing
`modal_train.py` training function with `scratch/train_iterative_policy.py` and
`scratch/iterative_policy_core.py`. Preflight: 13 existing exact-policy/GRPO tests
and three new tests pass (opening isolation and color balance, replay rejection of
corrupt results, and identical next optimizer update after restoring state).
An eight-game cloud smoke test passed replay audits: four games per color, eight
unique games, all normal terminal outcomes. The original detached app
`ap-wbFpKcDc4QQHVTWglHPWDs` was stopped at the user's request to use workspace2.
It completed 200 probe games and candidate preparation, with zero RL rounds.
Twelve files, including the exact initial checkpoint, protocol, game receipts,
and candidate data, were copied directly between cloud workspaces. The initial
checkpoint's checksum matches exactly; no models or datasets were downloaded
locally. Transfer receipt: `data/probes/iterative-sf2000-20260928/workspace-transfer.json`.
Replacement detached app: https://modal.com/apps/workspace2/main/ap-KNUPguFgXEpKJ7IJljIQFU .
workspace2 initialization passed and reused all four saved probe game shards (200 games).
That app subsequently stopped before its first RL update: importing Modal from the
training subprocess failed because the SDK is only available in its parent runtime.
Round-one collection (1,000 games) and grading (27,096 positions, 6,017 priority)
had completed. The fix moves volume reload into the existing Modal training parent.
The original protocol and source archive are preserved as `*-before-runtime-fix.*`;
`runtime-fix.json` records both sets of hashes. RL settings and datasets are unchanged.
Sixteen focused tests pass after the fix. Recovery app:
https://modal.com/apps/workspace2/main/ap-Yq4Nu4itS2Qy1YGWf6iTnP .
Recovery confirmed by round-one optimizer step 50/300, with finite reward, KL,
and gradient norm. The detached ten-round job is training.
Initial checkpoint, engine binary, opening book, and all frozen source hashes
passed cloud validation. The first observed stage was `probe_collection`.

Launch command:

```sh
PYTHONDONTWRITEBYTECODE=1 MODAL_PROFILE=second MODAL_GPU=H100 MODAL_CPU=4 MODAL_MEMORY=32768 MODAL_TIMEOUT=7200 uv run --with modal==1.3.5 python -m modal run --detach scratch/modal_iterative_policy.py::run_loop
```
