# Own-game exact-10k versus original champion at temperature 0.2

Status: complete. All 1,000 games and checkpoint/replay audits passed. A strength
gain at temperature 0.2 is not established at the predeclared 95% level. User
requested this confirmation after reviewing the RL experiment history.
No retraining or deployment occurred.

## Result

| Comparison | W / D / L | Score | 95% score interval | Implied H2H Elo |
| --- | --- | ---: | --- | --- |
| Exact-10k vs original champion, temperature 0.2 | 364 / 320 / 316 | 52.4% | 49.90–54.95% | +16.7, interval -0.7 to +34.5 |

The primary interval is a 10,000-resample game bootstrap stratified by color.
It includes 50%, so this run does not establish an improvement or equivalence.
A modest positive effect remains plausible. The historical temperature-0.8
result was 59.8%/1,000, about +69 Elo; its advantage cannot be assumed to hold
at 0.2. This comparison alone does not isolate whether RL mainly reshaped the
distribution away from lower-ranked mistakes or improved move ranking.

Candidate score was 51.5% as White (169W/177D/154L) and 53.3% as Black
(195W/143D/162L), each over 500 games. There were 949 distinct complete move
sequences. Three unresolved 200-ply caps were scored as draws; treating all as
candidate losses/wins bounds the observed score at 52.25–52.55%.
Terminations: 680 checkmates, 257 fivefold repetitions, 43 stalemates,
17 insufficient-material draws, and three caps.

All 1,000 games were replayed from the initial position with matching legal
moves, results, terminations, identities, complete numbering and balanced colors.
All eight shard requests, seeds, legal masks, disabled EOS and no-cache flags
passed. No game IDs overlap earlier own-game H2H files. Checkpoint hashes were
unchanged after collection and match the September 23 audit; configs and
tokenizers match. Source snapshot and hashes are saved locally. The old
temperature-0.8 1,000-game file was independently replay-audited as well.
Two existing shard persistence tests passed; runner and analysis compile, and
whitespace checks pass. Collection wall time after preflight was 112.5 seconds.

No sample extension or checkpoint selection followed this result. Future
training comparisons should measure gains at the intended evaluation temperature
and retain the original champion as a control. A matched refreshed-position
versus fixed-archive training comparison remains a proposal, not a launched run.

## Fixed protocol

- Candidate: `/data/checkpoints/grpo/own-games-exact-10k-20260918/best.pt`, step 5,500.
- Opponent: `/data/checkpoints/plain/games-15m/l16_best.pt`, saved step 148,400.
- Exactly 1,000 fresh H2H games, 500 per candidate color, initial chess position.
- Both models use temperature 0.2, legal masking, EOS disabled, no KV cache,
  batch size 64, 200-ply cap. Eight 125-game shards on A10G, two CPU cores each.
- Seed 2026092800 plus shard game offset. Completed shards are saved for resumption.
- Primary metric: candidate score, wins plus half draws. Report a 95% bootstrap
  interval stratified by color, 10,000 resamples, seed 2026092801, and implied
  H2H Elo difference. Intervals condition on the fitted models and this protocol.
- Report color splits, distinct move sequences and unresolved caps. Caps count
  as draws, with all-loss/all-win sensitivity bounds. No interim-score stopping
  or sample extension. Replay every game and verify shard seeds and identities.
- Record checkpoint, tokenizer and source hashes; verify checkpoint hashes again
  after collection. Models and tokenizer must match the expected lineage.

The historical 59.8%/1,000 result used temperature 0.8. This run tests whether an
advantage is established at 0.2; it is not a paired replay of those old games.

Local receipts: `data/probes/exact-10k-t02-20260928/`.
Games: `benchmark/results/h2h_own_games_exact_10k_champion_t02_20260928_1000.jsonl`.
Runner: `scratch/modal_exact_t02.py`; analysis: `scratch/analyze_exact_t02.py`.
Cloud run: https://modal.com/apps/sparshithsampath/main/ap-1uboUNe3egd9c8qfpNa3Jp

Two earlier preflight launches failed on remote module imports before games
started and were stopped. Packaging was corrected without changing the protocol.
A third stopped before games because the champion's saved step is 148,400, not
the 200,000 claimed by the registry notes. Both checkpoint SHA256 values match
the September 23 RL audit exactly, so identity is established by those historical
hashes. Candidate step 5,500 and identical tokenizer/config checks passed.
