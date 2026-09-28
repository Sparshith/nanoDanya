# Own-game exact-10k versus original champion at temperature 0.2

Status: running. User requested the matched low-temperature confirmation after
reviewing the RL experiment history. No retraining or deployment is part of this run.

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
