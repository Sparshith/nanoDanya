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
lives on another machine, not here; its filters are recorded in the modal.md lineage
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

## 2026-08-26 GRPO v1 data prep: mixed positions with per-legal-move winprob tables

v1 recipe (from the v0 verdict): mixed positions, dense eval reward, same GRPO
machinery. `data_prep/prepare_grpo_eval_positions.py` materialized 120k positions
to `datasets/mixed/grpo-eval` (117,548 train / 2,452 val by game-id hash, first
100k games still excluded): 50% ordinary game plies (random ply >= 16, openings
skipped), 50% puzzle positions (post lead-in, as v0). Each record stores a SAN ->
winprob table over every legal move from one fairy-stockfish MultiPV search
(movetime 0.2s, avg depth 10.6, avg 30.7 legal moves, zero multipv misses).
Winprob = 1/(1+exp(-0.00368208*cp)), cp clamped +/-1000, mover's perspective.

Quality checks: mean winprob spread within a position 0.63 (reward separates
moves); the played move / puzzle solution ranks top-1 in the table 69.8% of the
time, avg rank 2.74 of 30.7. Prep cost ~25 min of engine time on 17 workers.

`training/train_grpo.py` rewritten in place for v1 (v0 in git history): reward =
sampled move's winprob (all annotation variants), illegal draws get
ILLEGAL_REWARD=0.0 (below the 0.025 clamp floor). Group-mean baseline cancels the
per-position winprob offset, so the gradient signal equals eval-delta. The KL
leash now sees mixed positions, the fix for v0's off-distribution middlegame
drift. Analytic val: expected winprob, gap to engine-best, p(best move), illegal
mass, KL, entropy. 300-step pilot launched, results pending.

## 2026-08-26 GRPO v1 pilot: dials pass, gentler and steadier than v0

300-step pilot on the mixed winprob dataset (A10G, micro-batch 16 x grad-accum 4 =
effective 64, K=8, temp 0.8, kl_beta 0.05, lr 1e-5). Two false starts: OOM at
micro-batch 64 (mixed-position prefixes run 300-500 tokens, much longer than v0's
puzzle prefixes), and VAL_BATCHES defaults to 1 (16 positions/eval, pure noise;
v0's pilot dials were also thin). Final run: EVAL_INTERVAL=25, VAL_BATCHES=64
(1,024 positions/eval).

Curve over 1-300: expected winprob 0.524 -> 0.568 (best step 275), gap-to-best
0.167 -> 0.126, p(best) 0.236 -> 0.250, illegal mass 1.93% -> 1.40% (DOWN; v0 was
flat then regressed on benchmark), KL rises to ~0.20 and holds (v0: 0.5-0.65),
entropy 1.05 -> 0.74. Unlike v0's fast 75-step saturation, v1 was still improving
at step 300; dense reward has more to teach. Checkpoint:
`checkpoints/grpo/eval-v1/grpo_v1_pilot_best.pt`. Same-day verdict suite launched
(puzzles / legality 4096 / sf1500 400 / H2H 400 vs champion); champion baselines
reused from the 2026-08-16 same-protocol files.

## 2026-08-26 GRPO v1 verdict: strength kept, ~80% of the puzzle gains kept

Same-day suite on `grpo_v1_pilot_best.pt` (step 275) vs the 2026-08-16 champion
files: puzzles first-move 63.6% / full-solve 46.0% (champion 54.9/38.1, v0
specialist 66.4/47.9); raw real-illegal 0.27% (champion 0.20%, v0 0.59%); SF1500
199/400 = 49.75% (champion 53.25%); H2H vs champion 212/400 = 53.0% (v0 35.5%).
Both strength reads are within ~1 sigma of even: strength is a tie, not a loss.
The v0 reward-misspecification failure is fixed. Champion unchanged; promotion
needs SF1500 clearly above 53.25%. Pilot was still improving at step 300 (KL
plateau 0.20), so the next lever is a longer run with early stopping on the
1,024-position winprob dial. Results renamed to `*_grpo_v1_pilot_*`; full note
`experiments/2026-08-26-grpo-eval-v1.md`; dial charts plot/grpo-v1-pilot-dials/.

## 2026-08-26 Human blunder-rate baseline from the puzzle-games evals

Question: what do humans score on the move-quality blunder rule (cp-300)?
Needed as the yardstick for the model's rate. Answered without any engine
time: `scratch/human_blunder_rate.py` reads the per-move Stockfish evals already
stored in `puzzles_raw/puzzle_games_ndjson.txt`, computes cp loss from the
mover's POV (mate = +/-100000, same MATE_SCORE as benchmark/run.py), counts
loss >= 300 as a blunder, and bins each move by the mover's own rating. Full
pass: 3,190,827 games / 249.4M moves scored, 2,938 games skipped (missing or
mismatched analysis). ~1h on A10G (CPU-bound json parsing).

Blunder% by mover rating (mistake% = loss >= 100): 800-1000 24.1%/39.4%,
1400-1600 10.6%/22.5%, 1600-1800 9.5%/21.0%, 1800-2000 8.3%/19.0%, 2000-2200
7.3%/17.2%, 2200-2400 6.6%/15.7%, 2400-2600 6.2%/14.8%, 2600-2800 5.8%/13.7%,
2800-3000 5.5%/12.9%. All moves: 8.07% blunder, 18.44% mistake, median cp loss
14. Monotonic in rating; bins past 3000 are tiny (<31k moves) and noisy.

Correction while placing the model: the 11.8% previously quoted as "the
champion's" blunder rate is NOT the champion. The 2026-06-25 move-quality run
tested `eval-aware/v2` (deleted 2026-07-04) on only 600 positions. No direct
champion measurement existed; one is now running (8,192 positions,
stockfish-time 0.1). Caveats on the human side: every game here
produced a puzzle, so it contains at least one big mistake by construction
(reads slightly high, which is generous to the model); and the human number is
lichess-analysis evals over full games while the model number is a fixed-limit
engine over sampled positions, so treat the comparison as approximate. This is
the missing baseline for the "metrics to move" line in STATE.md.

## 2026-08-26 Champion move-quality: 7.6% blunder rate, ~2000-level move safety

Direct measurement at last (the old 11.8% was eval-aware/v2 on 600 positions).
`modal_benchmark.py --mode move-quality --model plain/games-15m --max-positions
8192 --stockfish-time 0.1` (deeper than the old 0.02 default, closer to the
lichess server analysis behind the human baseline). 8,192 positions = every ply
of ~110 eval/base val games, 0 failures, 42.5 min on A10G. Required an OOM fix
first: `last_logits_for_prefixes` yielded views into each batch's full logits
tensor, pinning every batch on the GPU (~15GB at 8k positions); `.clone()` on
the yielded slice (run.py:317) flattens memory. Old <=600-position runs never
tripped it.

Numbers: blunder_rate_cp_300 7.59% (binomial 95% CI ~+/-0.6, wider with
game clustering), median cp loss 15, avg cp loss 1865, best-move match 46.4%.
Result file `benchmark/results/modal_move_quality_20260826_181046.jsonl`.

Placement on the human ladder (same cp-300 rule, 2026-08-26 baseline): between
1800-2000 humans (8.3%) and 2000-2200 humans (7.3%); median cp loss matches the
human overall median (15 vs 14). So per-move safety looks ~2000-level, NOT the
"blunders like a sub-1400" story the stale 11.8% suggested. Caveats: positions
come from human games (the model's own games may drift into worse territory,
cf. the conversion-cliff evidence), the human baseline reads slightly high
(puzzle-bearing games), and the eval depths still differ. The gap between
~2000-level per-move safety and much weaker full-game results points at
compounding/conversion, not per-move blunder frequency, as the thing to fix.

## 2026-08-28 Bot's own games: blunder rate 10.3%, exactly its rating class

Follow-up to the champion's 7.59% on human positions: same cp-300 rule over the
lichess bot's own 381 rapid games (BOT nanoDanya, rating 1446, 162W 62D 157L vs
other bots). `scratch/bot_blunder_rate.py` scores every position once with
fairy-stockfish 0.1s (mate +/-100000), loss from the mover's POV, bot moves
only (16,432). Games fetched with the bot's own token via
`scratch/fetch_bot_games_modal.py` (anonymous user-export of nanoDanya is
permanently 429/"one at a time"; the authenticated slot works).

Overall: 10.31% blunder / 22.35% mistake, median cp loss 16; 92.7% of games
contain >=1 bot blunder. On the human ladder 10.31% = the 1400-1600 bin
(10.6%), i.e. in its OWN games the model blunders exactly like its 1446 rating,
while on human-game positions it scores 7.59% (~2000-level). The 2.7-point gap
is compounding/off-distribution error, measured directly.

Splits are the interesting part. By phase: ply 1-20 2.3%, 21-40 9.7%, 41-60
14.3%, 61-80 15.8%, 81+ 12.6% (openings are memorized; the late middlegame is
the cliff). By position state before the move: equal 3.96%, winning (>+200)
12.47%, losing 15.01% - the model is very clean in balanced positions and
throws away winning ones at 3x that rate, the conversion-cliff story now
quantified on its own games. By result: 8.0% in wins, 10.3% in draws, 12.6% in
losses. Caveats: opponents are bots (1000-2200 rapid), and the human baseline
has no winning/losing split to compare against yet.

## 2026-08-29 GRPO v1 extended run launched (resume pilot -> 1500 steps)

Lit review (GRPO-in-chess) first: our v0->v1 arc replicates the published
finding (binary best-move reward fails on small models, dense engine-score
reward works, arxiv 2507.00726); the field's main trap is "dead groups" (all K
draws same reward, 28-45% of batches wasted with binary rewards); published
useful windows are tens-to-low-hundreds of steps, so the long run must catch
the turn, not assume it isn't there. Added a dead_groups val metric to
train_grpo.py (K sampled draws per val position, fraction of groups with zero
reward spread).

Run: pilot best (step 275) copied to grpo_v1_long.pt on the volume, resumed
with optimizer state. A10G, micro 16 x accum 4, MAX_ITERS=1500,
EVAL_INTERVAL=25, VAL_BATCHES=64 (1,024 positions/eval), CKPT_INTERVAL=100,
early stop on neg_reward: patience 10 evals (250 steps), min_steps 500,
min_delta 5e-4. Best checkpoint grpo_v1_long_best.pt. Results pending; verdict
suite (puzzles/legality/sf1500/H2H) on the best checkpoint when it finishes.

## 2026-08-29 GRPO v1 extended run verdict: ceiling reached, champion equaled, not beaten

Resumed pilot (275) ran to early stop at 875; best step 625 (expected winprob
0.5733 vs pilot 0.568). Dial anatomy: slow grind 275-625, then 250 flat steps
with KL creeping (0.20 -> 0.27) and dead_groups rising (37% -> 47%): the
sharper the policy, the more groups sample the same move 8 times and teach
nothing. Same-protocol verdict suite on grpo_v1_long_best.pt
(`*_grpo_v1_long_*` files):

| metric | champion | v1 pilot (275) | v1 long (625) |
|---|---|---|---|
| puzzle first-move | 54.9% | 63.6% | 63.3% |
| puzzle full-solve | 38.1% | 46.0% | 45.8% |
| raw real-illegal | 0.20% | 0.27% | 0.20% |
| vs SF1500, 400 | 53.25% | 49.75% | 52.1% |
| H2H vs champion, 400 | — | 53.0% | 50.1% |

Reading: the extension recovered legality to exactly champion level and pulled
SF1500 back within noise of champion (52.1 vs 53.25); puzzles unchanged from
pilot. Net position of the v1 line: champion strength + ~8 points of puzzle
tactics + champion legality, from one A10G recipe. No promotion (bar: SF1500
clearly above 53.25). Single-position winprob GRPO on human-game positions is
now harvested; 350 extra steps moved the dial +0.005 and no benchmark beyond
noise. Next lever is the position distribution: self-play positions
(winning/late-middlegame oversampled) targeting the conversion cliff the
2026-08-28 bot-games analysis quantified. Extension details appended to
experiments/2026-08-26-grpo-eval-v1.md.

## 2026-09-18 Own-game exact policy pilot: positive strength result

User authorized trying broad own-game positions with full-distribution expected
reward after reviewing the tiny stalemate experiments' transfer failures.
Prepared 11,676 engine-graded positions from 400 archived champion-vs-SF1500
games: complete-game splits 320/40/40, positions 9,449/1,067/1,160, cross-split
matching boards removed. Full legal coverage at 50k MultiPV nodes; average
minimum depth 12.31. Half uniform / half winning-or-playable-blunder training
sampling. Retained sigmoid reward/clamp; exact legal-conditional objective plus
raw KL 0.05, temp 0.8, all parameters, AdamW 3e-6, batch 64, 300 fixed updates.

Step 300 selected using the fixed full validation set and predeclared KL/illegal
mass bounds. Untouched test reward 0.605658 -> 0.608512; blunder probability
14.96 -> 14.21%, game-bootstrap 95% change -0.918 to -0.606 percentage points.
Raw top choice outside legal mapping (including EOS) worsened from 25 to 31
of 1,160 positions. This measure differs from standard real-illegal benchmarks.

Fresh strength tests: H2H vs champion 194W/66D/140L, 56.75%, estimated +47.2 Elo
(approximate independent-game 95% +16.1 to +79.1). SF1500 230W/38D/132L, 62.25%,
versus fresh champion 185W/31D/184L, 50.125%. Existing harness scores caps as
draws: H2H 4, candidate SF 12, champion SF 7. All 1,200 games replayed legally
and outcomes/scores verified. Nineteen focused tests and a CPU integration smoke
passed. The original champion checkpoint, serving, and registry are unchanged.

This is a positive pilot on both held-out engine feedback and full games,
not a confirmed >=50 Elo gain or a public rating estimate. Changes to data,
objective, scope, and budget are not causally isolated. Independent replication
is the next step before promotion. Full protocol, artifacts, and limits:
`experiments/2026-09-18-own-games-exact-policy.md`.

## 2026-09-18 10,000-game scale run launched

User requested a 10,000-source-game follow-up to the positive exact-policy pilot.
Collection is running with saved-shard recovery after an initial attempt failed
on engine-response timeouts. Planned split: 8,000/1,000/1,000 complete games;
original champion initialization, same reward and effective batch, 7,500 updates.
A10G memory smoke via scratch/check_exact_batch_memory.py passed two AdamW
updates at physical/effective batch 64 and maximal 200-token histories: allocated
18.05 GiB, reserved 18.48 GiB. No smoke weights saved. Use batch 64 directly to
avoid accumulation overhead. Protocol and active status:
experiments/2026-09-18-own-games-exact-10k.md. No scale-run result yet.

## 2026-09-18 10k collection and grading complete

All 10,000 source games replay-audited; 8,000/1,000/1,000 game split yielded
231,377/28,901/29,194 positions. CPU grading completed with complete legal-move
tables, mean minimum depth 12.295; engine binary and reward match the pilot.
Fresh champion control: SF1500 181W/42D/177L (50.5%), 12 capped games; raw
legality 8/4,096 real-illegal top choices. Training 7,500 updates launched on
A10G, app ap-m3KSwjOSY5ylzqYC8NWOYp. No scale candidate result yet.

## 2026-09-18 10k exact-policy scale run complete

Completed all 7,500 updates, selected step 5,500 from fixed validation. A
pre-update vocabulary assertion failure was fixed to match inference support:
27 training / 3 validation / 1 test positions contain unrepresentable legal SAN,
mostly promoted-piece disambiguation. Keep complete engine tables and positions;
those moves have zero policy probability. No vocabulary or reward change.
Twenty-six focused tests pass, including inference action-support equivalence.

Fresh 1,000-game H2H vs champion: 501W/194D/305L, 59.8%, +69.0 Elo
(approximate independent-game 95% +49.6 to +88.9). Fresh 1,000 vs pilot:
465W/194D/341L, 56.2%, +43.3 Elo (+24.0 to +62.9). SF1500 candidate
250W/38D/112L (67.25%) vs fresh champion 181W/42D/177L (50.5%), 400 each.
All 2,800 matches replay-audited. Caps scored as draws: 29/9/9/12 respectively.
One H2H worker checkpoint-read failure recovered by automatic retry before play.

Untouched test 29,194 positions / 999 games: expected reward .614319 -> .619790;
blunder probability 16.17 -> 14.68%, game-bootstrap 95% change [-1.573, -1.425]
percentage points. Own-game raw top-choice illegality 760 -> 828 / 29,194;
standard raw legality nearly unchanged, 8 -> 9 / 4,096. Both data and update
budget increased, so their causal contributions are not isolated. Point estimate
exceeds +50 Elo, but a strict >=50 lower confidence bound is not established.

Selected weights: /data/checkpoints/grpo/own-games-exact-10k-20260918/best.pt
on nanodanya-data. Reports: data/traces/grpo/own-games-exact-10k-20260918/.
Full recipe/results: experiments/2026-09-18-own-games-exact-10k.md.
Champion, registry, and live bot unchanged. No commit or push.

## 2026-09-19 new exact-policy model deployed to Lichess

User requested playing other bots to measure live rating improvement. The existing
scheduled listener was already active; the previous endpoint still served the
plain 15M L16 model. Captured baseline immediately before deployment: nanoDanya
rapid 1418, 514 rapid games, rating deviation 45; zero active games. Old-model
public play had continued since July, rather than being paused.

Changed inference/serve.py to the selected step-5,500 checkpoint at
/data/checkpoints/grpo/own-games-exact-10k-20260918/best.pt. Added the loaded
checkpoint identity to startup logs and disabled autograd for inference.
Staging on the serving T4 passed six legal-move checks (including 100/190-ply
histories); five existing inference tests passed. Deployed nanodanya-chess at
about 08:28 IST on 2026-09-19. Production logs confirm the actual step-5,500
load; /health identifies the new path; six live HTTP move checks passed at
bot temperature 0.8. Cold first request 12.85s, remaining requests about 1.2s.

The existing listener, matchmaking, and endpoint URL remain in place, with no
restart or duplicate listener. Rated 10+0, temperature .8, the existing daily
counter of five, and two-hour matchmaking gap remain. Registry benchmark
baseline is still plain/games-15m. No commit or push. Public rating improvement
is not measured yet; compare subsequent rapid games against this starting point.
Evidence: data/traces/lichess/2026-09-19-exact-10k/{baseline,predeploy,
staging_smoke,deployed_http_smoke,deployment}.json. Staging app:
ap-zSIC1Ep8dKwKZh50lqgJEL. Production app: ap-RE14skCzOhsQrN6oN6WtmD.

## 2026-09-19 one-day 50-game Lichess allowance

User asked to play all 50 games today. Confirmed four games already played on
2026-09-19 IST, all before the model switch; account rapid baseline still 1418
at 514 games. The old configured cap was five, with volatile per-process state.

Added lichess_bot/schedule.py: cap 50, one-minute start/finish cooldown and
matchmaking tick on 2026-09-19 only. At 2026-09-20 00:00 IST, automatically return
to five games/day and two-hour start gap / five-minute tick. Games in progress
finish normally. Today therefore has room for 46 more starts, subject to opponents
accepting. Rated 10+0, model checkpoint, temperature, and opponent rating range
remain unchanged.

Persist started-game IDs/timestamps and last finish in Modal Dict
nanodanya-lichess-state, reconciled from two days of Lichess history on startup.
Incoming challenges share the cap, spacing, and busy/pending guards. A lock
serializes admission; max_containers=1 prevents duplicate listeners. Reconnects
keep the active game reserved and do not double-count starts. Terminal game
statuses release the slot; player color is retained from gameFull instead of
fetching the game on every move. Finish active games before scheduled handoff.
Pinned existing berserk 0.14.0/chess 1.11.2, plus requests 2.34.2.

Six scheduling tests cover IST expiry, cap 50 after persistence/reload, duplicate
starts, overnight games, finish cooldown, and restoration of the normal cap/gap;
five inference tests also pass. Two remote preflights authenticated successfully
and found 4/50, 60-second gap/tick, no active games. Lichess bulk game export works
now, unlike the July observation. Stopped the idle old listener before deploying
and starting one replacement at 08:42 IST. New app ap-MGyut1JTvJhHl83znXeBjM,
call fc-01M2VTDFSXNKEQ8R1S28TZWPM1. Logs confirm the effective 4/50 budget.
No commit or push. Baseline evidence:
data/traces/lichess/2026-09-19-exact-10k/burst_predeploy.json.

The first three faster challenges were rejected because the opponents themselves
had reached Lichess's rolling 100 bot-game allowance (bernstein-4ply,
bernstein-2ply, maia1). Added persistent opponent cooldowns using the server's
retry deadline; unknown/account-wide rate limits pause all outgoing challenges.
Non-rate-limit rejections cool down that opponent for 30 minutes. Scan up to
200 online bots while retaining the same preferred rating range. Two additional
cooldown tests pass, eight scheduling tests total. After another idle replacement,
final listener app is ap-s4PQOPgzSbbEynydynVMdg, call
fc-01M2VTNPQMQ9DPA1PWKMRZEH6W. One container confirmed; startup again reports
4/50, and the first new challenge went to Annie_Archy. Reaching 50 depends on
opponent availability; no completed additional game was claimed at activation.
Final deployment evidence: data/traces/lichess/2026-09-19-exact-10k/burst_deployment.json.

## 2026-09-19 first live batch and move-delivery repair

At 15:42 IST the 50-start allowance was exhausted: four pre-switch starts plus
46 with the exact-10k model. New model: 43 completed (11W/7D/25L), three aborted,
rapid 1418 -> 1409 (-9), average opponent rating 1614.7. Four losses were on time.
Logs confirm game 7R8k6PiW lost the move-submission response at 15:00:21 and
waited until an outoftime event at 15:10:02. Do not attribute the other three
timeouts to infrastructure without evidence. No live rating gain established.
Evidence: data/traces/lichess/2026-09-19-exact-10k/progress_summary.json.

Shared GZrvqz3j, Black vs dala-1300 (1277), as a chess loss for manual analysis:
36...g5 37.Rh2#, Black still had 9:22. Saved its PGN in the same artifact directory.

Extracted the game loop into lichess_bot/game.py. Submission exceptions now
close the stream and reconnect after five seconds to get authoritative state.
If the position is unchanged, retry the same selected move; if Lichess accepted
it despite losing the response, do not replay it. Suppress duplicate successful
submissions for the same history; terminal snapshots end recovery. HTTP 429
backs off 65 seconds. Existing inference-failure fallback and cold-start behavior
are unchanged. Seven recovery tests plus eight scheduling and five inference
tests pass (20 total), including both outcomes of an ambiguous submission.

Remote preflight confirmed idle, 50/50, no ongoing games. Stopped the old listener
before replacement. Deployed app ap-Qq3yIOBQ0y3CMjq0pguBD0 and started call
fc-01M2WJX4EA126JZAGWVD5Y52JB at 15:50 IST. One listener container verified;
startup retained 50/50. No further games authorized today. Recovery is tested
with simulated failures, not yet observed in a subsequent live game.
Deployment evidence: data/traces/lichess/2026-09-19-exact-10k/reliability_deployment.json.
No commit or push.

Manual-loss audit: replayed GZrvqz3j and exhaustively checked every legal Black
move at move 36 against every legal White reply using python-chess. Exactly
one of 15 legal moves, g5, permits mate in one (Rh2#); the other 14 avoid
immediate mate. This does not establish their engine values or longer-term
survival. The trainer selects on aggregate expected engine reward with KL and
legality guards, not a dedicated mate-in-one blunder metric. Reward clipping
also maps forced losses and cp <= -1000 to the same reward, but its relevance
to this position is untested. Need baseline/candidate move distributions on
this position and a held-out suite before attributing the loss to sampling,
training regression, reward saturation, or generalization.

## 2026-09-19 probability of the mate-in-one blunder

Reconstructed all 71 plies before Black's 36th move in GZrvqz3j. Ran the
unchanged production ChessModel loader and get_move path on a Modal T4,
checkpoint exact-10k step 5500, temperature 0.8, legal mask, EOS suppressed.
Captured logits at the sampler without changing production deployment.
g5 token probability 0.4980331957; Kg5 0.4794963300; Ra3 0.0052861390.
g5 is the highest-probability legal token, so greedy selection also blunders
in this reconstruction. It is not merely a low-probability tail event.
The existing mask admits check/mate suffix variants; g5+/g5# together add
approximately 0.0000012054 probability, immaterial at the reported precision.
This is recomputed probability, not a probability recorded during the game.
No baseline comparison yet; it does not establish a training-induced regression.
Probe app ap-nRvbgIhdBQYvFvQyZOqZBk completed. Scratch script:
scratch/probe_lichess_g5.py. Full distribution and history:
data/traces/lichess/2026-09-19-exact-10k/g5_probability.json.

Pre-RL comparison completed with the original initialization checkpoint
/data/checkpoints/plain/games-15m/l16_best.pt (loader reports step 148400).
Identical 71-ply history, T4 production inference implementation, temperature
0.8, legal support and EOS suppression. g5: 55.2913606% before -> 49.8033196%
after (-5.4880410 percentage points). Kg5: 41.1459863% -> 47.9496330%.
g5 ranks first in both. This position's mate-threat weakness predates RL;
fine-tuning reduced its probability but did not fix the ranking. This single
position does not establish broader mate-threat improvement. Source checkpoint
was substituted only inside the isolated probe; production serving unchanged.
App ap-TnwmdpWmiYJiFUTSkUcbGH completed. Artifacts alongside the candidate probe:
g5_probability_baseline.json and g5_comparison.json.

## 2026-09-19 mate-threat sampling experiment launched

User authorized the bounded real-game mate-threat experiment. Frozen protocol
and mining audit: experiments/2026-09-19-mate-threat-oversampling.md. Exact
labels, 1,000/250/299 target positions from disjoint source games, original
pre-RL initialization for both 300-update arms, uniform versus half-targeted
sampling. Training curves and unseen-game test are separate; final checkpoints
fixed in advance, followed by 400 fresh targeted-versus-control games.
Sixteen focused tests pass. No synthetic data or production changes.

## 2026-09-19 mate-threat sampling experiment completed

Both equal-budget 300-step arms and all 400 fresh games completed. Targeted
training mate-in-one probability 6.193% -> 0.626%; validation 5.746% -> 4.795%.
On 299 game-disjoint test threats, baseline 4.617%, uniform control 4.227%,
targeted 3.749%. Extra benefit -0.478 pp, paired game bootstrap 95%
[-0.831, -0.143]. Greedy errors 14/299 -> 13 control / 12 targeted; greedy
improvement not established. Ordinary test reward improves similarly in both;
raw illegal mass/top-choice legality worsens with targeting (25/999 raw invalid
versus control 18/999, original 19/999; includes EOS/notation mapping).

Fresh targeted-v-control H2H: 168W/81D/151L, 52.125%/400, approximate Elo
+14.8 [-15.7, +45.4], six caps. All games legally replayed and outcome-audited,
200 per color. No established overall strength gain, no equivalence claim.
Known g5 diagnostic: original 55.291%, uniform control 61.254%, targeted
44.920%; targeted now ranks Kg5 first at 51.843%. This is exploratory evidence,
not an independent test. Strong training fit plus small held-out benefit suggests
limited generalization; more distinct real threat coverage is worth testing
before additional repetition. Full details and limitations:
experiments/2026-09-19-mate-threat-oversampling.md.

Scripts: scratch/mine_mate_threats.py, train_mate_threats.py,
audit_mate_selection.py, report_mate_threats.py, probe_lichess_g5.py;
16 focused tests passed. Source-game/board separation, frozen protocol hashes,
identical baselines and final checkpoint steps verified. All experiment apps
completed. Reports: data/traces/grpo/mate-threats-20260919/summary.json.
No production change, commit, or push.

## 2026-09-19 expanded coverage follow-up and literature check

User authorized more distinct threat examples at fixed training budget and asked
about DeepMind Searchless Chess. Expanded pool contains 2,208 distinct source
games (original 1,000 is a subset), still 300 updates with half-targeted batches.
Original pre-RL initialization; compare against saved narrow targeted checkpoint.
Collecting 2,000 fresh champion-v-SF1500 games for a new primary evaluation,
excluding old archive board states. Fixed 400-game expanded-v-narrow H2H planned.
Protocol: experiments/2026-09-19-mate-threat-coverage.md. No production change.
Promoted reusable scripts to data_prep/prepare_mate_threats.py and
training/train_mate_threats.py; 16 focused tests still pass.

Primary-source research corrects the premise: DeepMind's original Searchless
Chess used Stockfish-supervised training; AlphaZero used self-play RL with MCTS.
The Searchless Chess paper reports tactical mistakes and winning-position
conversion trouble, with a Stockfish-assisted live-play workaround. A newer
independent self-play RL preprint also separates tactical gains from game
strength. Sources and interpretation:
experiments/2026-09-19-searchless-chess-comparison.md.

## 2026-09-19 expanded mate-threat coverage completed

Expanded 1,000 -> 2,208 distinct training-game threats at fixed 300 updates.
The new 2,000-game source yielded 500 eligible threat positions from separate
games and 1,000 ordinary positions, with all old archive boards excluded.
Frozen selection preceded inference. Fresh mate-in-one blunder probability:
original 6.846%, narrow 5.176%, expanded 4.911%. Primary expanded-minus-narrow
-0.265 pp, paired 10,000-resample 95% [-0.478, -0.064]. Greedy errors 26/16/18;
expanded-minus-narrow +0.4 pp [-0.6, +1.4]. Ordinary reward essentially unchanged
between trained models (0.63475/0.63485); raw illegal top choices 33/35 of 1,000.

Audited expanded-versus-narrow H2H: 151W/93D/156L, 49.375% over 400 games,
Elo -4.3 with approximate 95% [-34.3, +25.6]. Four unresolved caps counted as
draws. More coverage gives a small probability improvement, but no established
strength gain or decisive-move improvement. One training seed only. No live
model change. Fresh inference app ap-CmyYGyyewMRlerJdGtlUQW; full results in
experiments/2026-09-19-mate-threat-coverage.md and local
data/traces/grpo/mate-coverage-20260919/summary.json.

Reusable trainer and mate-label helpers promoted to training/train_mate_threats.py
and data_prep/prepare_mate_threats.py; dataset path is configurable. Sixteen
focused mate-threat/exact-policy tests pass. Next recommended diagnostic is
explicit-board/value supervision, with game strength checked separately from
tactical scores; this is a recommendation, not an executed follow-up.

## 2026-09-20 second Modal workspace, raw game shards recovered from GCS

Added a second Modal profile `second` (workspace workspace2) next to `sparshithsampath`
in `~/.modal.toml`; `sparshithsampath` stays active, use `MODAL_PROFILE=second` for
workspace2. workspace2 already had a `nanodanya-data` volume with `checkpoints/plain/games-15m/`
(l16_best.pt), `datasets/lichess-2026-08-elo2000-rapid-classical-min30/` and
`raw/lichess/standard/2026-08`.

The January 2025 shards behind `datasets/games/15m` are gone (deleted with `archive/`
on 2026-07-04, not on the laptop either). But six later months survive as another machine
artifacts in the staging GCS bucket
(`a private storage bucket`):
12.1M games, 49 shards + manifest/summary per month, 3.9 GB, filters Elo>=2000,
>=30 plies, Rapid+Classical+Blitz (about 85% Blitz, unlike the Rapid/Classical-only
January set). Copied them via laptop into the workspace2 volume at `games_raw/` (61 files).
Nothing was copied between Modal accounts; the old volume is untouched.

## 2026-09-21 INT8 browser candidates benchmarked

Ran the canonical Modal suite for INT8 exports of the 5M-game L12 (105.5M params,
138.5 MB) and 15M-game L12 (107.6M params, 143.7 MB). The 15M version improved
SF1000/SF1500 scores from 60/29.5% to 69.25/44%, full puzzle solves from 30.28%
to 36.55%, and blunder rate from 9.38% to 8.33%. Both used the same benchmark
positions and 8,800 puzzles; 1,200 INT8 games were replay-audited.

A fresh 15M FP32 control scored 76.75% at SF1000 versus INT8's 69.25% (200 games
each), so preserved FP32 strength remains unproven. Native CPU tests, not browser
GPU or public Elo measurements. The canonical benchmark runner now accepts ONNX
policy exports. Details: experiments/2026-09-21-int8-browser-models.md.

## 2026-09-22 grpo sampling code removed; exact policy is the RL line

Deleted `training/train_grpo.py` (v0 puzzle first-move, v1 winprob reward),
`data_prep/prepare_grpo_positions.py` and `data_prep/prepare_grpo_eval_positions.py`.
Both GRPO variants are recorded in `experiments/2026-08-16-*` and `2026-08-26-*`;
the code stays in git history before this date. `training/grpo.py` now holds only
the reward-table and exact-loss helpers that `train_exact_policy.py` and
`train_mate_threats.py` import. Unused paired-contrast helpers removed with it.

## 2026-09-22 mate-threat scripts deleted

`data_prep/prepare_mate_threats.py`, `training/train_mate_threats.py` and their test
were never committed and are now removed. The two Sep 19 runs (coverage,
oversampling) both ended with no established strength gain; results and the exact
protocol are in `experiments/2026-09-19-mate-threat-*.md`. Dataset
`own-games/mate-threats-20260919` and both checkpoints stay on the volume.

## 2026-09-24 browser challenge entry checker

Added a local ZIP compatibility checker with a temporary 150,000,000-byte cap.
The packaged existing L12 INT8 export is 119,742,259 bytes. Headless Chromium on
Modal passed four history checks (1, 8, 32, 64 tokens), played `1. e4 c5`, undid
the turn, and opened with `1. e4` when playing White. GPU execution used software
WebGPU; timings are not measurements of the user's Mac. Custom inference runs
in a worker on a separate origin with network access blocked by CSP. Browser
checks also rejected oversized ZIPs, wrong-length and non-finite logits, verified
report download and stopping a pending prediction, and showed no overflow at
390px width. Twelve local archive/downloader tests passed. Compatibility does
not establish decoder eligibility or playing strength. Logs and screenshots:
`output/playwright/challenge-{browser,edges}.{txt,png}`; disposable harness:
`scratch/check_challenge_browser.py`. No public deployment or model promotion.

## 2026-09-24 tournament Elo, sf1500 anchor

Added DeepMind-style tournament Elo: `benchmark/run.py tournament` and `elo`, with Modal
modes of the same names. It uses 200 fixed CC0 openings with both colors, greedy
play, and 1300 cp adjudication, then a BayesElo fit with sf1500 (Fairy-Stockfish 14,
depth 1) pinned at 1500. The 15M L12 INT8 CPU export rates **1766 +18/-17** over 1,200
games. The depth-1 pool compresses: sf2000 1663, sf1000 1242. The same model scored 44%
vs sf1500 under the old temp-0.8/no-adjudication protocol and 78% here, so the old
numbers are not comparable. The challenge leaderboard now shows this Elo. Details:
experiments/2026-09-24-tournament-elo.md.

## 2026-09-24 challenge page: match vs leader

The checker page can now play a 10-game match (5 fixed openings from
`benchmark/openings.tsv`, both colors, greedy, 300-ply cap as draw) between the
uploaded entry and the leader ZIP, with each model in its own sandboxed runner. Headless
Chromium on Modal (software WebGPU) loaded a uniform-logit fake entry, then the real
L12 INT8 leader. Both sides moved from the Evans Gambit opening, two runner frames were
active, and Stop ended the match with the controls re-enabled. The match is casual and
unrated. Harness: `scratch/check_challenge_match.py`, output `output/playwright/challenge-match.*`.

## 2026-09-24 first under-10 MB browser distillation

Distilled the 15M-game L12 float teacher into a 16.3M-parameter tied-embedding
6x384 decoder, then ran INT4-aware tuning. Final self-contained browser ZIP is
9,424,814 bytes and passed actual headless checker/play tests and 512-token
numerical parity. Main training retained ~409M tokens, plus 32.8M QAT tokens.
B200 at batch64 reached ~370k tokens/s versus H100 ~209k, with the same effective
batch. Measured function compute was $2.86 plus unmeasured build/startup costs;
conservative reservations including overhead were $38.41 of the $100 cap.

Size succeeded; teacher strength was not preserved. At temperature 0.2 over 100
games each against SF1000, teacher scored 89%, pre-QAT float student 23.5%, and
exported q4 student 28.5%. Q4 scored 9% H2H against the teacher. Max-200-ply draws
numbered 1/5/13 in those SF runs; no claimed q4-over-float improvement. Most loss
was already present before quantization. All 400 games replay-audited cleanly.
Heldout mixed-color positions: q4 1.172% raw-illegal versus teacher 0.098%, and
66.41% legal top-choice agreement. This is a small runnable baseline, not a
champion replacement. Recipe, artifact hash, protocol and exact results:
`experiments/2026-09-24-browser-student-distillation.md`.

## 2026-09-24 challenge page: Elo estimate from leader match

The 10-game match is replaced by "Find my Elo". The entry plays the leader continuously
through the 200 fixed openings with both colors, until paused. It shows
Elo = 1766 + 400·log10(s/(1-s)), where s includes two virtual draws. The ± is
1.96 × the binomial SE on the Elo scale, combined with the leader's ±18. This is a
browser estimate, not the official BayesElo. The Elo method explanations moved into
info-icon popovers. The Modal Playwright run (`scratch/check_challenge_match.py`)
confirmed pass, popover open and close on hover, match start, an estimate after game 1
(uniform fake entry lost: 1646 ±417), pause, and a switch to human play.

## 2026-09-24 INT4 student Bayesian Elo

Rated the actual 9.42 MB export on the canonical tournament protocol: 200 fixed
openings, both colors, greedy play and adjudication, 400 games each against
sf1000/sf1500/sf2000. Scores were 52.875%/23.0%/9.375%. Joint BayesElo fit with
the existing pool gives **1260 +17/-18** (95% interval 1242–1277), anchored at
sf1500=1500. Existing L12 INT8 CPU baseline is 1768 +19/-19 in the same fit.
All 1,200 new games replay-audited; none unscored/capped/context-limited. This
is the internal tournament scale, not the earlier temperature-0.2 game scaffold.
The rating run added $0.54 measured compute, bringing the experiment to $3.40
before build/startup overhead and $40.92 in conservative reservations. Details:
`experiments/2026-09-24-browser-student-distillation.md` (Bayesian Elo follow-up).

## 2026-09-24 challenge page: estimate removed, head-to-head only

The browser Elo estimate is removed. A model far from the baseline gets wide intervals
(139 games at a 7.8% score gave ±108). No weaker same-architecture checkpoints exist
for a ladder. The card now shows a plain head-to-head score against the baseline, with
wins, draws and losses. The leaderboard Elo stays the Modal BayesElo number, and a new
way to rate uploads is still open. The Modal Playwright rerun passed: check, popover,
match start, score after game 1, pause, and a switch to human play.

## 2026-09-24 challenge page: match and play split into tabs

The tester and board are reorganized. After a pass, the drop zone collapses to a file row
with a Replace link. The match summary shows the score, a strip of result squares (the
hover gives the opening and the end reason: checkmate, repetition, stalemate, 300-ply cap
and so on) and win/draw/loss counts. Metrics and the JSON report sit under a Details
disclosure. The board has Match and Play-yourself tabs, name bars with scores above and
below it, a last-move highlight, and a 20-game match (10 openings × 2 colors) with a
progress bar. The button reads Start, Pause, Resume, then "Play 20 more". Rules moved
below the board. The Modal Playwright rerun confirmed the file row, popover, match start
with labels, the result strip with its hover title, progress, the last-move highlight,
the tab switch to human play (e4 answered), and the return to a paused match.

## 2026-09-24 challenge moved to the repo root, entries bring their own vocabulary

`browser/challenge/` became `challenge/` with three folders a challenger can read
top to bottom: `starter/` (stub entry, default vocabulary, tokenizer example, pack
script, README), `baseline/` (the INT4 model's inference code and exporter, formerly
`browser/student/`), and `dashboard/` (the checker page, its own server and esbuild).
The full contract for agents is `challenge/llms.txt`; the page's "Copy prompt for your
agent" button copies five lines that point at it on GitHub and ask the agent to set up
the project from `starter/` and stop before any model work.

The contract no longer fixes the vocabulary. An entry returns its own list of SAN
strings (must contain `<bos>`, may contain `<unk>`); the page encodes the board history
with each model's list and picks the highest-scoring legal move from that list. A move a
model cannot encode forfeits the game in a match. The shared `runtime.BrowserEngine`
hook was dropped from the contract. Verified with a 5,674-token suffix-free stub and
with the starter ZIP: check passes, play and undo work, no console errors.

Deleted the September 21 ONNX browser prototype (`browser/index.html`, `app.js`,
`export_model.py`, `head_to_head.py`, `benchmark_onnx/`, `tests/test_browser_h2h.py`)
and the 120 MB int8 challenge ZIP; its numbers stay in
`experiments/2026-09-21-int8-browser-models.md`. `browser/play/` (the deployed play
site), `browser/assets/` and `browser/results/` are untouched.

## 2026-09-26 browser play site deleted

`browser/` is gone entirely: the Cloudflare Workers play site (`browser/play/`, its
wrangler config and `dist/` build), the 1.6 GB of local model exports in
`browser/assets/`, and the September 21 measurement files in `browser/results/`. The
numbers those files backed remain in `experiments/2026-09-21-int8-browser-models.md`.
The only browser code left is `challenge/`. The deployed worker at
nanodanya-play.sparshithsampath.workers.dev was not removed from Cloudflare by this
change; it needs `wrangler delete` or the Cloudflare dashboard.

## 2026-09-28 challenge entries are one ONNX file; no more custom JavaScript

Entry contract v2. An entry is a single `model.onnx` at most 10 MB, standard ops,
opset 17+, one int64 input `[1, tokens]`, one float32 logits output, with the SAN
vocabulary, context length and name in `metadata_props`. The page reads the metadata
with a 40-line protobuf scanner, runs the model with ONNX Runtime Web 1.30 (wasm) in a
plain worker, and picks the best legal move from the entry's own list. The sandboxed
cross-origin frame, the ZIP reader and the per-entry `entry.js` adapter are deleted;
nobody executes an entrant's code any more, only their graph.

`challenge/starter/` is now `model.py` (2-layer, width-96 PyTorch transformer, tied
embeddings, 6.9 MB fp32, 2.5 MB with `--int8`), `export.py` (torch.onnx legacy
exporter, `do_constant_folding=False` so the tied embedding is stored once),
`tokenize_game.py`, `vocabulary.json`, `requirements.txt`. A file named `tokenize.py`
shadowed the stdlib module and broke every import; hence the rename.

`challenge/baseline/export.py` builds the baseline's ONNX graph by hand: each matrix as
packed uint8 nibbles plus fp32 scales per 64 values, unpacked with BitwiseAnd, BitShift,
Cast, Where, Mul and Reshape; RMSNorm, RoPE with bf16-rounded tables, causal softmax
attention, relu², softcap 15. 9,473,041 bytes for 16,280,832 parameters. Matches the
PyTorch StudentGPT within 2e-3 on 1/8/64-token inputs (test_export.py). It reads
either a checkpoint or the legacy INT4 ZIP. `load_quantized_model` now rebuilds the
PyTorch model from the ONNX initializers, and `inference/onnx_policy.py` runs any
contract-v2 file through the tournament's existing `last_logits` interface, so the
offline puzzles and Stockfish games score the same file the browser tested.

Headless Chrome drive: starter file passes all checks in 0.3 s, 3 ms per move at 64
tokens on CPU; play and undo work; a 20-game match against the baseline runs on CPU
(2 games in 2 s, both baseline wins by mate, as expected for random weights); no console
errors. Dev deps added: onnx 1.23.0, onnxruntime 1.23.2 (last line with Python 3.10 wheels).

## 2026-09-28 exact-policy low-temperature confirmation launched

User requested 1,000 fresh own-game exact-10k versus original champion games at
0.2 after the RL consolidation. Protocol fixed before collection: 500 per color,
200-ply cap, legal masking/EOS disabled, no KV cache, eight A10G shards, fresh
seed family 2026092800. No retraining or deployment. Checkpoint hashes match the
September 23 audit exactly. Champion metadata records step 148,400, correcting
the registry note's 200,000; candidate is step 5,500. Three preflight-only launch
failures were resolved before games, two packaging failures and the stale step
assertion. Results pending. Full protocol:
`experiments/2026-09-28-exact-policy-temperature-02.md`.

## 2026-09-28 exact-policy low-temperature confirmation complete

Exact-10k versus original champion at temperature 0.2: 364W/320D/316L,
52.4% over the fixed 1,000 games. Color-stratified bootstrap 95% 49.90–54.95%,
implied H2H Elo +16.7 [-0.7, +34.5]. A strength gain is not established at this
level; a modest positive effect remains plausible. The earlier 59.8%/+69 Elo
result is specific to temperature 0.8 and cannot be carried over unchanged.
White/Black scores 51.5%/53.3%, 500 each, 949 distinct sequences. Three caps;
all-loss/all-win bounds 52.25–52.55%. All games replayed, all shard settings
verified, historical checkpoint hashes unchanged before/after. Also re-audited
the historical 1,000-game result. Existing shard tests: 2 passed. No retraining,
sample extension, deployment, commit or push. Updated STATE and the experiment
note; receipts and source snapshot: data/probes/exact-10k-t02-20260928/.

## 2026-09-28 policy ranking versus probability diagnostic launched

User requested checking whether exact-10k RL changes preferred moves or mainly
redistributes probability. Frozen-checkpoint inference on all 29,194 previously
held-out positions from 999 games, original engine labels, temperatures 0.8/0.2.
Primary metrics: greedy blunders fixed/introduced and expected blunder mass;
separate token suffix changes from chess-move changes. Partition gains by changed
versus unchanged greedy move, with paired source-game bootstrap intervals.
No training or deployment. Protocol and caveats:
`experiments/2026-09-28-policy-ranking-diagnostic.md`.

## 2026-09-28 policy ranking diagnostic complete

All 29,194 original exact-10k test positions / 999 games scored under frozen
original and RL checkpoints, FP32. Preferred chess move changes on 2,924 (10.02%):
1,196 materially better, 962 worse (>30cp), 766 near-equal. Greedy >=300cp
blunders fall 13.369% to 12.814%, paired game-bootstrap change -0.555pp,
95% [-0.742, -0.374]. 421 fixed, 259 introduced, 3,482 retained: 89.2% of old
errors persist. Expected blunder mass: 16.175% -> 14.676% at 0.8, and
13.558% -> 12.872% at 0.2. At 0.8, 81.74% of expected-reward gain occurs on
unchanged-top positions; at 0.2, 40.85%. Policy concentration is substantial,
but claiming no rank improvement would be wrong. Raw top-token illegality rises
2.548% -> 2.812%. This is previously inspected archive evidence, not new
low-temperature match positions or a causal explanation of H2H strength.

All action tables/reward maps/history normalizations verified, 128 histories
replayed, production support and 256 greedy comparisons checked. Historical
checkpoint hashes unchanged; native forward max error 1.526e-5. Three focused
metric tests passed; source-game bootstrap and aggregate decomposition audits
passed. No training or deployment. Reports: data/probes/policy-ranking-20260928/.
Full interpretation: experiments/2026-09-28-policy-ranking-diagnostic.md.

## 2026-09-28 starter ships the baseline, not random weights

`challenge/starter/model.py` is now the baseline architecture itself (6×384, RoPE,
RMSNorm, QK-norm, relu², softcap, tied embeddings, optional int4 fake-quant for
training), self-contained, no import from `training/`. `baseline.pt` (9.2 MB) holds the
packed int4 codes and per-group scales straight from the served model plus the config;
`export.py --int4` reuses those scales so the export is bit-identical in weights to what
the page serves (test: 1e-4 on logits at 1/8/64 tokens). It must be published as a GitHub
release asset at `releases/download/baseline-v1/baseline.pt`; until then export.py's
download 404s and the file has to be copied by hand.

`export.py` precisions on the baseline: fp32 65.5 MB, int8 22.4 MB (folded export,
quantize_dynamic hits MatMul and Gather; logits drift up to 2.0 but argmax holds), int4
9.63 MB. int4 is a generic pass over any 2-D float initializer with width divisible by
64, so it also works for other architectures exported through torch. The ONNX metadata
now carries `config` so `inference/onnx_policy.load_torch_from_onnx` can rebuild the
PyTorch model for the batched benchmarks; `OnnxPolicy` serves the tournament path.
`challenge/baseline/` is down to the test and a README. Page drive with the starter's
export: check 0.6 s, 33 ms/move at 64 tokens on CPU, 2 draws vs the served baseline.

## 2026-09-28 iterative SF2000 RL launched

User selected 1,000 games per round for a ten-round refresh experiment. Frozen
protocol: 100 training openings, both colors, collection temperature 0.8,
300 exact-reward RL updates per round, initial exact-10k checkpoint as fixed KL
reference. Round one shared with a fixed-archive control, 3,000 updates per arm.
Separate 50-opening evaluation at temperature 0.2, frozen position probe, and
final H2H versus control and initial checkpoint. All data and training on Modal.
16 focused tests passed; eight-game cloud smoke passed replay and outcome audits.
Cloud initialization verified checkpoint, engine, openings, and source hashes.
Detached app ap-wbFpKcDc4QQHVTWglHPWDs reached probe_collection. Results pending.
Protocol: experiments/2026-09-28-iterative-sf2000-rl.md. No commits or deployment.

## 2026-09-28 iterative RL moved to workspace2

User requested workspace2 instead of sparshithsampath. Stopped the original detached
app ap-wbFpKcDc4QQHVTWglHPWDs and verified zero remaining tasks. No RL training had
started; 200 held-out baseline games and candidate preparation were complete.
Copied 12 files directly between the Modal workspaces, including the 927,112,254-byte
initial checkpoint with its original verified SHA256. No model or dataset was
downloaded locally. Transfer app ap-PXos9pkvN23Gj4BpzgR28H completed and stopped.
Replacement detached run ap-KNUPguFgXEpKJ7IJljIQFU uses workspace2 (profile second),
with the identical frozen protocol and source hashes. Its runtime is building.

## 2026-09-28 starter gets a local checker

`challenge/starter/check.py` runs the page's three checks with ONNX Runtime on the
laptop (size, load + metadata, finite logits on the 0/7/31/63-move prefixes of the
same test game the page uses) and then plays N games against `baseline.onnx` (exported on
demand) from the page's first ten openings, both colors, 300-ply cap, greedy legal move
from each model's own vocabulary. Baseline vs itself: 2-2 in 4 games, 8 ms per move.
The page is now only for the file a person wants to submit. python-chess 1.11.2 joins the
starter requirements.
workspace2 initialization passed all checkpoint, engine, opening-book, protocol, and
source-hash checks. The replacement job reused all four saved baseline game shards
(200 games) and continued to grading. Migration complete, ten-round run active.

## 2026-09-28 iterative RL launcher recovery

Status check found workspace2 app ap-KNUPguFgXEpKJ7IJljIQFU had stopped before any RL
updates. It completed 1,000 round-one games and 27,096 engine-graded positions
(6,017 priority), but the trainer subprocess could not import Modal for volume
reload. Moved reload into the existing Modal parent function. Preserved the
original protocol/source archive and recorded old/new source hashes in
runtime-fix.json; no RL settings or rewards changed. Existing ONNX image dependencies
from separate work are unused here. Sixteen focused tests pass. Recovery app
ap-Yq4Nu4itS2Qy1YGWf6iTnP launched on workspace2, passed initialization, and is reusing
saved data. First actual optimizer updates still to be verified.

## 2026-09-28 dashboard deploys to www.sparshith.com/nd-challenge

`challenge/dashboard/deploy.py` builds, packs and copies seven runtime files (page,
stylesheet, two bundles, wasm runtime + loader, baseline.onnx; 24.1 MB) into the blog
repository's `nd-challenge/`, which GitHub Pages serves. The worker now imports
`onnxruntime-web/wasm` (no WebGPU/jsep build), so the wasm shipped is the 14 MB
`ort-wasm-simd-threaded.wasm` instead of the 28 MB jsep one. Without cross-origin
isolation headers ORT runs single-threaded: 41 ms/move at 64 tokens vs 33 locally.
Headless drive from the blog root passed check, play and a 2-game match. Bundles and
`ort/` are gitignored in nanoDanya; the blog holds the built copy. The page carries the
blog's Cloudflare analytics loader and its header links to the nanoDanya post.
Recovery verified: round one reached optimizer step 50/300 on H100, finite reward
0.71538, KL 0.00700, gradient norm 0.39174. Detached ten-round job is now training.

## 2026-09-28 dashboard moved out of challenge/

`challenge/dashboard/` is now `dashboard/` at the repo root, so `challenge/` holds only what an
entrant needs (starter, baseline test, llms.txt, README). Paths in pack.py, deploy.py, the
openings import, the metadata test and the baseline test follow.

## 2026-09-28 iterative SF2000 RL completed

Recovery app ap-Yq4Nu4itS2Qy1YGWf6iTnP finished ten rounds and final evaluation at
19:11 IST. 10,000 training games, 277,211 graded positions, 3,000 updates per arm
(first 300 shared). Verified checkpoint lineage, fixed-data reuse, and budgets.
All 3,500 final evaluation games checksum-checked and replay-audited locally.
Refresh versus initial: 339W/298D/363L, 48.8%, opening-cluster 95% [46.05, 51.55].
Refresh versus fixed archive: 338W/308D/354L, 49.2%, 95% [46.10, 52.20].
SF2000 scores over 500 games each: initial 17.0%, refresh 16.7%, fixed 18.9%.
Frozen-probe greedy blunders: initial 11.54%, refresh 10.53%, fixed 11.10%.
Refresh-minus-initial equal-game-weight paired difference -1.021 pp, 95%
[-1.792, -0.312]. Small move-safety improvement, no established strength gain.
One baseline SF shard retried after an unsupported opponent SAN token, noted
as a comparison limitation. No H2H outcome selection. No deployment or commit.
Report: experiments/2026-09-28-iterative-sf2000-rl.md; local reports under
 data/probes/iterative-sf2000-20260928/.

## 2026-09-29 lichess bot restarted for a 100-game rapid day
The bot had been idle since 2026-09-22: the `drain` flag was left set after the
last listener replacement, so every scheduled run exited without matchmaking.
Temperature-0.2 trials at that point: rapid 78/100 completed (1412 -> 1525 rapid,
RD 48), bullet 28/100. Both archived under `budget.past_trials`; `bullet_config`
removed, so no more bullet. New rapid trial `temperature-02-100-games-20260929`:
100 rated 10+0 games, 10 concurrent, baseline 1525 over 650 rapid games. A
finished trial now archives itself and the bot returns to 5 games/day with a 2h
gap (previously a finished trial stopped matchmaking forever).

## 2026-09-29 leaderboard: d9-200k on top, one pooled rating fit

Built BayesElo locally (same tarball and hash as the Modal image) and fit all 21,600
games in `output/small-depth-long-20260929/bayeselo-games/` with sf1500 pinned at 1500:
nanodanya-l12-int8 1757 ±22, sf2000 1646, d9-200k 1517 ±12 (2,800 games), d15-200k 1505,
d28-200k 1475, deep 1352, long 1337, distilled 1273 ±13 (2,400 games), ce 1240, sf1000
1236. "distilled" is the served 6x384 starter baseline (1260 in its own 2026-09-24 fit).
The leaderboard now shows d9-200k first (9.45 MB, 2,023/8,800 puzzles, 1517 ±12) and the
baseline second at 1273 ±13, both from this one fit, plus a "By" column with the GitHub
user. d9-200k already follows contract v2 (metadata with vocabulary, context_length,
config; int4 pass); starter check.py: all OK, 3.5-0.5 in 4 quick games vs the baseline.

Starter gains `data.py`: the baseline's training set, `datasets/games/15m` (15M Lichess
2025 games, 13.5M/1.5M split, 1,045,627,425 + 116,242,624 uint16 ids, `<bos> moves <eos>`
per game), to be published at hf.co/datasets/Sparshith/nanodanya-games-15m by
`data_prep/publish_dataset.py` (Modal function; needs a `huggingface-token` secret, not
yet created). Not uploaded yet.

## 2026-09-29 training data published on Hugging Face

`data_prep/publish_dataset.py` uploaded `datasets/games/15m` from the Modal volume to
https://huggingface.co/datasets/Sparshith/nanodanya-games-15m (train.bin 2,091,254,850
bytes, val.bin 232,485,248, vocabulary.json, dataset card, CC0). `challenge/starter/data.py`
loads it: val split 116,242,624 ids, first id 0 (`<bos>`), random 256-token blocks match
the training sampler. The HF write token lives in the Modal secret `huggingface-token`.
