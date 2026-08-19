# Log

Append-only, newest last. Big runs get their own file in `experiments/`; the current
picture lives in `STATE.md`.

## 2026-05-01 initial wiki scaffold

Created the original `knowledge/` wiki from `AGENTS.md`, `ARCHITECTURE.md`,
`TRAINING_EXPERIMENT_LOG.md`, and `model_registry.py`. Filed the linear probe results.

## 2026-05-04 retired MODEL_FAMILIES.md

Moved the durable content of root `MODEL_FAMILIES.md` into the wiki; `model_registry.py`
stays the source of truth for aliases and checkpoint paths.

## 2026-05-11 scale H2H suite changed the story

Ran the four-edge games-vs-puzzles H2H suite on Modal. Reframed the narrative from
"puzzle weighting helped" to "plain CE plus scale helped". `plain/puzzles-5m` stays
champion; `plain/games-5m` vs `plain/puzzles-5m` flagged as the missing clean test.
See `experiments/2026-05-11-games-vs-puzzles-scale-h2h.md`.

## 2026-07-04 flattened the wiki

Restructured `knowledge/` from the 9-directory wiki to STATE.md + log.md + experiments/.
Backfilled work the wiki had missed: raw legality by scale (5M/15M models),
eval-head boxed classification (corr 0.83), and the rating-binned puzzle-solve eval
(go/no-go: GO). Dropped model-family stub pages (duplicated `model_registry.py`),
templates, and operations docs. Fixed stale references (deleted `training/train_eval.py`,
`game_runner.py`) and unified naming to registry aliases.

## 2026-07-04 ingested the May benchmark suite from benchmark/results/

The blog post was ahead of this folder; backfilled from artifacts (all numbers
re-extracted from JSONL summaries, all match the blog). New: the full 5x5 H2H matrix
and Stockfish ladder (champion is now the 15M-game model; 5M games vs 5M puzzles is
a tie, settling the source question), the API model legality comparison, and the
king-safety SFT results. Updated the raw-legality table with phase splits and the
L16-vs-L12 arch comparison, and added the 2026-06-25 move-quality run to the
eval-head note (value-guided selection confirmed negative). Rewrote STATE.md:
champion changed, both former open questions resolved, new open questions filed
(15M not in registry, endings/`<eos>`, middlegame tactics, tool-assisted phase).

## 2026-07-04 consolidated training/ and deleted the eval-aware line

Deduplicated the seven copy-drifted trainers into `training/common.py` (hardened
checkpoint resume with full config validation, shared `train_loop` with AMP, grad
accum, best/periodic checkpoints, early stopping). `train.py` and
`train_weighted.py` are now 66/82-line scripts holding only data and loss; both
smoke-tested on Modal including resume (loss trajectories match pre-refactor).
Deleted `train_eval.py` and `train_combined.py` (dead regression-head lines), then
the entire eval-aware branch for a rewrite: `train_from_scratch.py`,
`train_ab_eval.py`, `train_eval_boxes.py`, volume checkpoints
`checkpoints/eval-aware/` (v1, v2, weighted-ft), `eval_boxes_test/`, `ab_test/`,
and the three eval-aware registry entries. `datasets/eval/base` kept for the
rewrite. New env vars: `CKPT_DIR` picks the checkpoint directory (the old
per-dataset default paths predate the volume reorg); `WEIGHT_DECAY` and
`EVAL_BATCH_SIZE` removed (never used).

## 2026-07-04 deleted the king-safety SFT line

Continuing the "keep only the blog story" cleanup: removed `sft/` (data builder,
trainer, validator), `benchmark/king_safety_legality.py`, the king-safety mode and
functions in `modal_benchmark.py`, and the volume `king_safety_sft/` directory
(SFT data bins plus both checkpoints). Benchmark result JSONLs in
`benchmark/results/` are kept as evidence for the experiment note. The repo now
carries only the plain scale-story models plus the weighted ablation; eval-aware
and king-safety exist as findings in knowledge/ and recipes in git history.

## 2026-07-04 refresh pass after registry/cleanup changes

The 15M checkpoints are now registered: `plain/games-15m` (L16, marked champion,
also `modal_benchmark.py`'s default) and `plain/games-15m/l12`; checkpoints moved
from `/data/actual_15m/` to `checkpoints/plain/games-15m/`. Updated STATE.md
(champion bullet reconciled with the registry, registration open question closed,
noted the lichess bot still serves `plain/puzzles-5m`) and fixed now-stale "not in
registry" and old-path claims in the scale-matrix and raw-legality notes. Added
deletion pointers to the eval-head and king-safety notes so they don't reference
removed scripts as if present.

## 2026-07-04 dropped the 3m model and bootstrap notebooks

Deleted `checkpoints/plain/games-3m/` from the volume and its registry entry; the
scale story keeps 500k/5m/15m and the 3m rung was never in the blog numbers.
`datasets/games/3m` deleted from the volume too (raw sources remain). Also deleted the four February
bootstrap notebooks (DetermineElo, Inference, Preprocess, Train), superseded by
`training/`, `inference/`, and the prep scripts.

## 2026-07-04 lichess bot now serves the champion

Switched `inference/serve.py` from `plain/puzzles-5m` to `plain/games-15m`
(`checkpoints/plain/games-15m/l16_best.pt`) and redeployed the `nanodanya-chess`
Modal app. Verified via `/health` (reports the champion) and a live `/move` call
(legal reply in a Petrov). The bot app needed no change; it calls the endpoint.

## 2026-07-04 dropped the L8 checkpoint and the weighted line

Deleted `plain/games-500k/l8` (the old chess_min.pt) and `weighted/puzzles-5m`:
volume checkpoints, registry entries, and `training/train_weighted.py`. The
weighted ablation is a settled negative result; the plain puzzle controls it
also trained are reproducible with `training/train.py` (plain CE ignores the
weight bins in the puzzle datasets). `training/` is now common.py + train.py.
README_LAYOUT.md rewritten to match the volume (eval-aware and 3m references
were stale) and re-uploaded.

## 2026-07-04 retired the arena rig

The January `arena/` dir (untracked LLM full-game rig, RESULTS.md, stray .env) is
retired; the 9-0 result is preserved in
[experiments/2026-01-31-llm-arena.md](experiments/2026-01-31-llm-arena.md).
API-model comparison now happens only via the benchmark legality mode.

## 2026-07-04 folded the standalone local docs into knowledge/

ARCHITECTURE.md and TRAINING_EXPERIMENT_LOG.md are deleted; the parts knowledge/
didn't already cover moved into [architecture.md](architecture.md) (pipeline mental
model, token contract, which-knob-moves-which-metric, experiment checklist) and
[training-history.md](training-history.md) (timeline, recipes, dataset facts,
reproducibility notes). New [modal.md](modal.md) documents how we use Modal (apps,
volume layout, training/benchmark commands, env knobs, smoke-test pattern); the
local README_LAYOUT.md is deleted, its volume-side copy stays. knowledge/ is now
the only knowledge location.

## 2026-07-04 tracked the dataset prep scripts in data_prep/

Moved the four live dataset builders out of gitignored dirs into tracked
`data_prep/`: `prepare_actual_data.py`, `prepare_eval_data.py`,
`download_games.py`, `build_weighted_data.py`. Deleted the abandoned
board-token puzzle format scripts (download_puzzles, convert_puzzle,
generate_puzzle_vocab) and `puzzles/puzzle_training.md` after lifting its untried
eval-as-tokens idea into STATE.md open questions. `rc_extract_actual_games.py`
lives on RapidCanvas, not here; its filters are recorded in the modal.md lineage
section. Note: the extracted game shards went away with `archive/`, so rebuilding
a games dataset differently means re-extracting from the public Lichess dumps.

## 2026-07-04 lichess bot matchmaking

The bot had played 4 games ever, all vs spar93, none in 18 days. Added capped
matchmaking to `lichess_bot/app.py`: when idle and under a 5 games/day cap (2h min
gap between games), it challenges a random online bot (rapid rating 1000-2200) to a
rated rapid 10+0 game; one game at a time, stale outgoing challenges cancelled.
Incoming bullet/blitz now declined (Modal T4 cold starts would lose on time), and
incoming challenges beyond the cap declined with "later". Estimated cost ~$0.15 per
rapid game, ~$22/month at the cap, on top of ~$6/month for the always-on listener.
Deployed; had to stop duplicate listener containers, since concurrent listeners
fight over the single per-token Lichess event stream.

## 2026-07-04 first public matchmade game, won

After fixing a self-challenge echo (our outgoing challenge comes back on the event
stream; the bot tried to accept it and crashed the loop) and adding 65s backoff on
Lichess 429s, the bot's first matchmaking tick challenged uSunfish-l1 (1203 rapid)
and won as black by checkmate: https://lichess.org/7PVg5aou (rated rapid 10+0).
Champion model's first public game. Daily counter at 1/5; next challenge after the
2h gap.

## 2026-07-05 first day of public matchmaking: 6-4, ~1725 rapid

Ten rated rapid games in the first ~24h, exactly at the 5/day cap with 2h gaps; no
timeout losses (every game ended in mate), so T4 cold starts are a non-issue at 10+0.
Notable: 1-1 vs maia9 (~1760) and a win vs maia5, but 0-2 vs croco_little_bot (1333)
while beating stronger bots, likely a stylistic weakness worth a game-record look.
Rating 1725 provisional after 10 games, consistent with the sf1500 42.5% benchmark.

## 2026-07-05 croco_little_bot losses analyzed: conversion, not confusion

Stockfish per-move analysis of both losses (lichess AllzfJyU, EDBlGadE) kills the
out-of-distribution-opening theory: nanoDanya reached +4.5 as white and +7 (later a
forced mate) as black out of croco's junk openings. It lost by failing to cash in:
11.Nxh8 trapped its own knight (+4.5 to -1), missed Bxa8 winning a rook, played
31...Rc2 with 31...Rc1+ forced mate on the board, traded queens with 20...Qxd2+
instead of 20...Nxd2 winning one, then 31.Reg5?? walked into Nf4# and 53...Kxf5??
lost a won pawn race (53...h2 promotes). Pattern: single-move eval cliffs in
conversion and endgames, consistent with the sf1500 42.5% gap. Maia paradox
explained: humans blunder back, engines don't. These cliffs are exactly what a
blunder-veto using the boxed eval head should catch (open question in STATE.md).

## 2026-07-06 draw analysis: one stalemate howler, two perpetual-check saves

Engine-checked the three draws and the turochamp loss. Only one draw was a blown
win, but maximally so: vs sargon-3ply (lichess 01HFuMDv) the model had forced mate
(+9997), promoted a8=Q, then played Qxf7?? stalemating the bare king one move before
mate. The other two draws were rescues: perpetual check from -5 vs sargon-2ply
(gPaar9I5) and from -4 vs bernstein-2ply (kVmopkYC), an emergent save behavior
presumably learned from human games. The turochamp-2ply loss (nMQVuiyq) repeated the
croco pattern: +4.8 early, slow slide, opponent promotes and mates. Running theme
across every non-win: winning positions reached, conversion cliffs lost them; the
stalemate is the strongest single argument yet for a cheap blunder/stalemate veto.

## 2026-07-06 direction call: no runtime vetoes, eval work must be end to end

Rejected the blunder/stalemate veto idea that came out of the public-game analyses:
no rule-based or judge-based inference interventions; the model itself has to learn
conversion. The eval-head rewrite should aim at the policy, not a sidecar judge:
candidate paths are joint policy+boxed-value training, eval-as-tokens (in-band,
already in open questions), and outcome-signal fine-tuning. Success metrics come
from the existing rigs: move-quality blunder rate and stalemate-from-winning rate,
plus the bot's ongoing public games as a live eval.

## 2026-07-06 searchless-chess lit review for the eval rewrite

Re-read DeepMind's ChessBench paper (arXiv 2402.04494) with the end-to-end
constraint in mind. Mechanism: amortized search; Stockfish 16 annotates 15B
(position, move) pairs offline, a 270M transformer learns (FEN, move) -> binned
win% (classification, plateaus ~32 bins), inference is argmax over
library-enumerated legal moves (2895 blitz Elo vs humans). Their target ablation:
action-value > state-value > behavioral cloning. Implications: our June value-guided
failure used a state-value head with shallow-eval labels, so it does not refute the
action-value recipe; nanoDanya is the BC variant; their inference harness (external
move enumeration + argmax) violates our end-to-end rule but their data recipe does
not. Sequence-native imports for the rewrite: action-value objective on (history,
move) with search-derived labels, or eval-as-tokens. Our puzzle ndjson (per-move
Stockfish evals) is the local ChessBench ingredient. Bonus: sequence input keeps
game history, so no FEN repetition blindness.

## 2026-07-07 win% bin histogram for datasets/eval/base

Measured label supply for a binned win% objective (ChessBench-style, 64 bins,
k=0.00368208 sigmoid) over datasets/eval/base: 260.0M tokens, 252.4M labeled,
7.6M missing (bos/eos/no-eval). The distribution is draw-peaked but not
degenerate: bin 32 (win% 0.500-0.516) holds 14.25% alone, top-4 bins (32, 33,
34, 63) hold 30.4%, the middle 8 bins (win% 0.44-0.56) hold 40.5%, and the
outer 16 tail bins still hold 21.2%. Mate-ish spikes at both ends (bin 0 4.11%,
bin 63 4.22%). Mean-normalized 1/sqrt(count) class weights span only 0.273 to
1.27, so rebalancing, if needed at all, is gentle.

Transition supply for conversion-focused objectives: 249.3M adjacent labeled
ply pairs, of which 12.14% move win% by more than 0.10, 4.98% by more than
0.25, 2.61% by more than 0.40. Cliff moves (mid 0.25-0.75 into a tail
<=0.125/>=0.875) are 1.53%, 3.8M examples. Plenty of signal for either
eval-as-tokens or a boxed value target; class imbalance is not a blocker.

Script: scratch/eval_bin_histogram.py (one-off), rerun via
`MODAL_GPU=A10G uv run modal run modal_train.py --datasets datasets/eval/base
--script scratch/eval_bin_histogram.py --env-overrides
"DATASET_DIR=datasets/eval/base"`.

## 2026-07-26 settled lichess rating: 1420 rapid over 140 games

The provisional 1725 from 2026-07-05 was inflation, not strength. After 140 rated
rapid games the bot sits at 1420 (rd 45, established, prog -9), 59W/57L/28D. Rating
history: 1763 on 07-04, crashed to 1418 by 07-08 as rd tightened, rebounded to 1486
by 07-10, then a slow drift down to 1420. Call the honest public-play number
~1420-1450, roughly a 300-point haircut on the first-day figure.

Volume check: 140 rated games in 22 days is ~6.4/day against a 5/day matchmaking
cap, so a good share are accepted incoming challenges rather than the capped bot
pool `lichess_bot/app.py` seeks out; the opponent distribution is not controlled.

Caveat on the split: Lichess's bulk export endpoint (`/api/games/user/{u}`) 404s for
every user from here, so the W/L/D counts are from `/api/user/{u}` aggregates.
`/api/user/{u}/current-game` still works for single games, and the profile HTML
(`/@/nanoDanya/all?page=N`) is a workable fallback for the game list.

Opponent pool is 100% bots: all 144 games span 21 distinct opponents, every one
`title: BOT` (verified via `/api/users` and the per-row `data-bot` flag). Most-played
uSunfish-l0 (21), bernstein-2ply (19), turochamp-1ply (11), Boosted_Maia_1300 (10),
then maia9 / EdwardKillick / sargon-2ply / schnecken_bot at 8. No human has ever
challenged it, though nothing blocks one: `handle_challenge` filters only on variant,
speed, and the daily cap. So 1420 is a bots-only rating against a pool skewed to weak
engines (bernstein-2ply 851, uSunfish-l0 1095) plus maia variants at 1600-1760, and
is not directly a human-equivalent number.

## 2026-08-10 GRPO pivot + phase 0 sampling probe: gate passed

Direction change: the tool-assisted / action-value phase (FEN-snapshot rebuild,
flagged after the puzzle-eval GO) is superseded before starting. A play-time
Stockfish toolcall is the biggest harness of all under the end-to-end rule; instead
we RL the model itself. Plan: GRPO on puzzle positions, binary reward against the
known solution first move, KL leash to the base model, measured by the existing
rating-binned puzzle eval with raw legality as the canary.

Phase 0 probe on the champion (plain/games-15m l16_best): for 200 puzzles per
100-point bin from 600 to 2800 (4400 total), replay the game to the puzzle
position, sample K=8 moves raw (no legality mask, temp 0.8), grade against the
solution first move (mating moves also accepted). GRPO only learns from mixed
groups (some hits, some misses), so mixed fraction is the go/no-go.

Result: gate passed everywhere. Mixed groups 53.5-66.5% per bin, 59.5% overall
(analytic expectation from p_correct agrees at 59.6%, so not sampling noise).
pass@8 declines 76-83% (easy bins) to 64.5% (2700-2800); greedy 58-66% down to
36.5%; all-fail rises 16.5% to 35.5% but never dominates. Every bin is trainable,
no difficulty filtering needed. Raw sampling at temp 0.8 draws illegal tokens
1.97% of the time (vs 0.29% raw top-1 illegal), so GRPO's zero reward on those
should also push legality up as a side effect. Greedy here (raw argmax, no mask)
runs below the puzzle eval's first_move_acc (58% vs 79% at 600-700), consistent
with the mask and temp-0 protocol difference.

Script: scratch/grpo_sample_probe.py, rerun via
`MODAL_GPU=A10G uv run modal run modal_train.py --datasets puzzles_raw
--script scratch/grpo_sample_probe.py` (env overrides: MODEL, K, TEMPERATURE,
PER_BIN, RATING_MIN/MAX, SEED).

Next: phase 1, training/train_grpo.py (8 samples/position, group-mean baseline,
clipped update, KL to frozen base), pilot run before any long run.

## 2026-08-16 GRPO phase 1: data prep + pilot runs, loop is stable and learning

Data: `data_prep/prepare_grpo_positions.py` materialized 500k puzzle positions to
`datasets/puzzles/grpo` (490k train / 10k val by puzzle-id hash, 0 skipped, avg
prefix 60.5 tokens, natural rating mix peaking 1000-1400). First 100k games in file
order are excluded because `benchmark/run.py puzzles` samples its eval positions
from the file head; training must never see them.

Trainer: `training/train_grpo.py` (reuses `common.train_loop`). Single-move GRPO
collapses to one forward pass per batch: logits at the puzzle position, K=8
multinomial draws at temp 0.8, binary reward vs the correct token set, advantage =
reward minus group mean (no std division), loss = -adv-weighted logprob + exact
full-distribution KL to the frozen init model (kl_beta 0.05, lr 1e-5, batch 64).
On-policy single update per rollout, so no PPO ratio/clip. Val is analytic (prob
mass, no sampling): neg_reward, pass@K, illegal mass, KL, entropy.

Pilot 1 (300 steps, A10G) found two issues. First an OOM: computing full-sequence
logits and padding to 512 blows past 22GB; fixed with per-batch length cropping and
`forward_hidden`+`move_logits` at the last position only. Second, a reward-spec bug:
the champion's vocab is annotated (14,750 tokens; `Re8`/`Re8+`/`Re8#` are distinct
ids for one move), and prep stored a single variant as correct, so the illegal dial
read 8.8% vs the probe's 2% and KL crept while GRPO taught annotation pedantry.
Fixed at training load: correct sets and legal masks expand to all stripped-form
variants (matching benchmark semantics). Note the probe and prep also undercount
this way; training-side expansion corrects grading, prep records stay single-variant.

Pilot 2 (300 steps, post-fix): val reward 0.50 -> 0.70 (best step 75, plateau
after), pass@8 0.80 -> 0.87, illegal mass 2.9% -> 2.6% (flat, matches probe
baseline), KL rises to ~0.5 then holds 0.52-0.65, entropy 1.03 -> ~0.55 and stable.
All three dials pass. Notable: nearly all reward gain lands in the first ~75 steps
(~5k positions). Checkpoint: `checkpoints/grpo/pilot/grpo_pilot2_best.pt`.

Phase 2 benchmarks launched on the pilot-2 best checkpoint vs champion
(plain/games-15m L16): rating-binned puzzles, raw legality (both models, games/5m
val), sf1500 400 games (both models), H2H 400 games. Results pending.

## 2026-08-16 GRPO v0 phase 2 verdict: puzzle specialist, weaker player

Benchmarked `grpo_pilot2_best.pt` (step 75) against the champion, same-day A10G
protocol. Puzzles (held-out): first-move 54.9% -> 66.4%, full-solve 38.1% -> 47.9%,
easy bins up hugely (600-700 full-solve 76.7% -> 95.3%), hard tail flat. Strength:
sf1500 53.25% -> 28.75% over 400 games each; H2H 35.5% vs its own init over 400.
Raw legality 0.20% -> 0.59% real-illegal. The probe-day guess that GRPO would
improve legality as a side effect was wrong.

Also settled: the L16 champion's own sf1500 number is 53.25%/400 under today's
protocol (the old 41% was a 50-game file; the blog 42.5% is the L12/200). GRPO
machinery is validated, the binary puzzle-first-move reward is misspecified for
strength. Champion unchanged, bot untouched. Full note:
`experiments/2026-08-16-grpo-puzzle-first-move-v0.md`.

## 2026-08-16 GRPO drift anatomy: where the model changed and whether the changes are good

Two probes on grpo_pilot2_best vs the champion, same 4,000 game + 4,000 puzzle
positions (seed 0, games/5m val cut points + grpo val puzzles).

Disagreement rate (different greedy top move): openings 8.8%, middlegames 25.2%,
endgames 23.3%, puzzles 29.5%. KL (temp 0.8, the training-leash quantity) tells the
same story: median 0.017 / 0.137 / 0.127 / 0.259. The leash only saw puzzle
positions, so off-distribution drift went unpenalized; openings are too burned-in
to move anyway.

Stockfish referee on the disagreements (0.05s, +/-30cp threshold, clamp 1500):
puzzles 67.6% better / 23.3% worse, mean +581cp (found tactics are huge). Every
game phase leans net worse: middlegames 33.2% better / 42.3% worse (mean -30cp),
openings 22.7/37.3 (-16cp), endgames 24.6/29.2 (-62cp, n=65). Mechanism for the
strength loss confirmed: 1 in 4 middlegame moves changed, and the changes leak ~a
third of a pawn on net, compounding over a game.

Also: modal_train.py image now installs fairy-stockfish (needed here, and for the
eval-delta GRPO v1 reward later). Probes: scratch/kl_by_distribution.py,
scratch/disagreement_quality.py. Learner-facing writeup with charts:
plot/grpo-kl-drift/ (local only, gitignored).
