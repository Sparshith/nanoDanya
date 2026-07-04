# Handoff: Games vs Puzzles Scale Benchmarks

Date: 2026-05-10 / 2026-05-11
Repo: `/Users/sparshith/workspace/nanoDanya`

## Purpose

We ran a question-driven H2H benchmark set to clarify whether the observed strength jump is mainly from:

- more games / more tokens,
- puzzle-derived source data,
- or actual-game scaling.

The core comparison is not a full tournament. It is a four-edge experiment tree:

1. same scale, different source,
2. same source type, larger actual corpus,
3. same source type, larger puzzle-linked corpus,
4. large actual corpus vs large puzzle-linked corpus.

## Benchmark Harness

Canonical benchmark entrypoint:

```bash
uv run python benchmark/run.py games --game-mode h2h ...
```

Modal wrapper used for these runs:

```bash
modal run modal_benchmark.py ...
```

Protocol for all four H2H runs:

```text
games: 200
temperature: 0.8
max_plies: 200
batch_size: 64
shards: 4
legal_mask: on/default
allow_eos: off/default
kv_cache: false
```

Important implementation details:

- `modal_benchmark.py` writes full game logs locally as JSONL.
- Each JSONL contains a `run` record, 200 `game` records, and a final `games_summary` record.
- Sharded Modal runs now pass distinct seeds via `seed + game_offset`, so shards do not duplicate stochastic samples.
- Explicit `/data/...` checkpoint paths now bypass model-registry filename interception.

## Dataset / Checkpoint Meaning

```text
plain/games-500k
= 500k actual-games baseline
= local file models/chess_L12_H6_E768.pt
= Modal path /data/models/chess_L12_H6_E768.pt
```

Verified local `data/processed`:

```text
train games: 449,656
val games:    49,962
total games: 499,618
total tokens: 39,677,081
```

```text
plain/puzzles-500k
= 500k puzzle-linked full-game corpus, uniform loss
= Modal path /data/models/chess_puzzle_plain_500k_L12_H6_E768_best.pt
```

```text
plain/games-3m
= recovered January 2025 actual-games corpus
= 3,092,525 games
= train tokens 215,685,734
= val tokens 23,876,938
= Modal path /data/plain/games-3m/chess_plain/games-3m_uniform_L12_H6_E768.pt
```

```text
plain/puzzles-5m
= full puzzle-linked corpus, uniform loss
= about 4,925,800 games
= about 402,314,086 total tokens
= Modal path /data/models/chess_puzzle_plain_L12_H6_E768.pt
```

Caveat: `plain/games-3m` vs `plain/puzzles-5m` is not size-matched. Puzzle-full has about `1.6x` more games and roughly `1.7-1.9x` more tokens depending on train/total comparison.

## Results

### 1. Same Scale, Different Source

Question: at roughly 500k games, does puzzle-derived source beat actual-games baseline?

```text
A: plain/games-500k
B: plain/puzzles-500k
```

Command:

```bash
modal run modal_benchmark.py \
  --model-a /data/models/chess_L12_H6_E768.pt \
  --model-b /data/models/chess_puzzle_plain_500k_L12_H6_E768_best.pt \
  --games 200 \
  --batch-size 64 \
  --shards 4 \
  --seed 1000 \
  --output benchmark/h2h_baseline_vs_puzzle_plain_500k_200.jsonl
```

Result:

```text
plain/games-500k:      82W 84D 34L, score 124.0/200 = 62.0%
plain/puzzles-500k: 34W 84D 82L, score  76.0/200 = 38.0%
avg plies: 129.7
terminations: CHECKMATE 116, FIVEFOLD_REPETITION 29, STALEMATE 18, INSUFFICIENT_MATERIAL 2, max_plies 35
```

Artifact:

```text
benchmark/h2h_baseline_vs_puzzle_plain_500k_200.jsonl
```

Conclusion:

```text
At 500k scale, puzzle-derived source does not beat the original actual-games baseline.
Source type alone is not the magic.
```

### 2. Same Source Type, Larger Actual Corpus

Question: does scaling actual games from ~500k to ~3M help?

```text
A: plain/games-500k
B: plain/games-3m
```

Command:

```bash
modal run modal_benchmark.py \
  --model-a /data/models/chess_L12_H6_E768.pt \
  --model-b /data/plain/games-3m/chess_plain/games-3m_uniform_L12_H6_E768.pt \
  --games 200 \
  --batch-size 64 \
  --shards 4 \
  --seed 2000 \
  --output benchmark/h2h_baseline_vs_plain/games-3m_200.jsonl
```

Result:

```text
plain/games-500k: 18W 50D 132L, score  43.0/200 = 21.5%
plain/games-3m:   132W 50D 18L, score 157.0/200 = 78.5%
avg plies: 112.9
terminations: CHECKMATE 150, FIVEFOLD_REPETITION 19, STALEMATE 14, INSUFFICIENT_MATERIAL 5, max_plies 12
```

Artifact:

```text
benchmark/h2h_baseline_vs_plain/games-3m_200.jsonl
```

Conclusion:

```text
Scaling actual games from 500k to 3M produces a large strength jump.
This strongly supports the scale thesis within games data.
```

### 3. Same Source Type, Larger Puzzle-Linked Corpus

Question: does scaling puzzle-linked games from 500k to full corpus help?

```text
A: plain/puzzles-500k
B: plain/puzzles-5m
```

Command:

```bash
modal run modal_benchmark.py \
  --model-a /data/models/chess_puzzle_plain_500k_L12_H6_E768_best.pt \
  --model-b /data/models/chess_puzzle_plain_L12_H6_E768.pt \
  --games 200 \
  --batch-size 64 \
  --shards 4 \
  --seed 3000 \
  --output benchmark/h2h_puzzle_plain_500k_vs_full_200.jsonl
```

Result:

```text
plain/puzzles-500k:   4W 18D 178L, score  13.0/200 =  6.5%
plain/puzzles-5m: 178W 18D   4L, score 187.0/200 = 93.5%
avg plies: 82.2
terminations: CHECKMATE 182, STALEMATE 13, FIVEFOLD_REPETITION 3, INSUFFICIENT_MATERIAL 1, max_plies 1
```

Artifact:

```text
benchmark/h2h_puzzle_plain_500k_vs_full_200.jsonl
```

Conclusion:

```text
Scaling puzzle-linked data from 500k to full corpus gives an enormous jump.
Within this source family, scale dominates.
```

### 4. Large Actual vs Large Puzzle-Linked

Question: is plain/games-3m competitive with current plain/puzzles-5m champion?

```text
A: plain/games-3m
B: plain/puzzles-5m
```

Command:

```bash
modal run modal_benchmark.py \
  --model-a /data/plain/games-3m/chess_plain/games-3m_uniform_L12_H6_E768.pt \
  --model-b /data/models/chess_puzzle_plain_L12_H6_E768.pt \
  --games 200 \
  --batch-size 64 \
  --shards 4 \
  --seed 4000 \
  --output benchmark/h2h_plain/games-3m_vs_puzzle_plain_full_200.jsonl
```

Result:

```text
plain/games-3m:          31W 28D 141L, score  45.0/200 = 22.5%
plain/puzzles-5m: 141W 28D  31L, score 155.0/200 = 77.5%
avg plies: 97.1
terminations: CHECKMATE 172, STALEMATE 14, FIVEFOLD_REPETITION 10, INSUFFICIENT_MATERIAL 2, max_plies 2
```

Artifact:

```text
benchmark/h2h_plain/games-3m_vs_puzzle_plain_full_200.jsonl
```

Conclusion:

```text
plain/puzzles-5m remains the champion.
plain/games-3m is much stronger than baseline, but it is not competitive with plain/puzzles-5m yet.
This is not a clean source-only comparison because plain/puzzles-5m is also substantially larger.
```

## Overall Interpretation

The benchmark arc is now coherent:

```text
500k actual baseline > 500k puzzle-linked
3M actual >> 500k actual baseline
4.9M puzzle-linked >> 500k puzzle-linked
4.9M puzzle-linked >> 3M actual
```

Main conclusion:

```text
Scale is the strongest demonstrated lever.
Puzzle-linked source is not automatically superior at small scale.
The current plain/puzzles-5m champion benefits from both source and substantially larger scale.
```

The missing clean experiment is:

```text
plain/games-5m vs plain/puzzles-5m
```

That would answer whether puzzle-linked data is still superior when actual-games data has comparable scale.

## Follow-Up Data Build Discussion

We already have:

```text
plain/games-3m from recovered January 2025 RC shards
```

Need:

```text
about 2M additional high-quality actual games
```

Options discussed:

1. Use official Lichess monthly dumps and stream/filter another month.
2. Prefer Modal for durable sharded outputs.
3. If using RapidCanvas again, use artifact helpers to upload closed shards mid-run.

RapidCanvas helper likely relevant:

```python
Helpers.upload_artifact_file(
    context=context,
    artifact_id=ARTIFACT_ID,
    absolute_file_path=local_path,
    artifact_relative_remote_path="shards/",
)
```

A small RC test script was copied to clipboard to verify whether `upload_artifact_file` persists artifacts during the run before final `Helpers.save()`.

## Notes For Next Agent

- Do not overinterpret H2H as raw model legality; game benchmark uses the harness/legal mask.
- For raw model behavior, run `benchmark/run.py legality`, `termination`, and `loss-by-ply` on the core checkpoints.
- Modal benchmark logs now preserve full game records locally.
- There are uncommitted local changes around benchmark seeding, Modal logging, and model_registry explicit path resolution that should be reviewed/committed separately from unrelated dirty files.
