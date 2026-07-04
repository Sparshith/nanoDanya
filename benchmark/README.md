# Benchmarks

`benchmark/run.py` is the canonical benchmark entrypoint. Do not add one-off
benchmark scripts under `benchmark/`; add a subcommand here instead.

Benchmarking is split into four layers. Keep these separate when interpreting
results:

1. **Legality**: does the checkpoint assign probability to legal chess moves?
2. **Termination**: does the checkpoint try to end ongoing games with `<eos>`?
3. **Move policy**: when forced to choose a legal move, does it choose a good one?
4. **Runtime player**: does the full scaffold play useful games?

## Commands

Fast legality benchmark:

```bash
uv run python benchmark/run.py legality \
  --model plain/puzzles-5m \
  --max-positions 4096 \
  --batch-size 128 \
  --output benchmark/results/legality_plain_puzzles_5m.jsonl
```

This reports:

- `raw_top1_illegal_rate`
- `raw_top1_real_illegal_rate`
- `avg_legal_mass`
- phase buckets for opening, middlegame, and endgame

`raw_top1_illegal_rate` is the strict SAN-token failure rate. It counts any
favorite token that is not a legal SAN continuation. `raw_top1_real_illegal_rate`
subtracts under-disambiguated SAN cases, where the move is semantically plausible
but omitted the required origin hint, like saying `Ng4` when the legal moves are
`Nfg4` and `Nhg4`.

Current reference numbers on 4096 held-out `actual_5m` validation positions:

| Model | Strict raw illegal | Under-disambiguated | Real raw illegal |
| --- | ---: | ---: | ---: |
| 500k actual-game model | 167 / 4096 = 4.077% | 9 / 4096 = 0.220% | 158 / 4096 = 3.857% |
| 500k puzzle-linked model | 163 / 4096 = 3.979% | 25 / 4096 = 0.610% | 138 / 4096 = 3.369% |
| 5M actual-game model | 22 / 4096 = 0.537% | 4 / 4096 = 0.098% | 18 / 4096 = 0.439% |
| 15M actual-game model, 200k | 12 / 4096 = 0.293% | 2 / 4096 = 0.049% | 10 / 4096 = 0.244% |

OpenRouter API-model legality benchmark:

```bash
export OPENROUTER_API_KEY=...

uv run python benchmark/run.py api-legality \
  --models openai/gpt-5.4-mini google/gemini-3.1-flash-lite anthropic/claude-sonnet-4.6 \
  --data-dir data/actual_5m \
  --split val \
  --max-positions 100 \
  --output benchmark/results/api_legality_smoke.jsonl
```

Start with 100 positions, then run the full 4096 positions once the prompt and
rate limits look stable. Useful first-pass model set:

- `openai/gpt-5.4-mini`: OpenAI small-model baseline
- `google/gemini-3.1-flash-lite`: cheap Google baseline
- `anthropic/claude-sonnet-4.6`: premium Claude baseline
- `x-ai/grok-4.3`: popular xAI baseline
- `meta-llama/llama-3.3-70b-instruct`: open-weight style reference

This benchmark is raw prompted behavior. It does not show legal moves to the API
model. It only asks for one SAN move and checks whether the response is parseable
and legal.

Fast termination benchmark:

```bash
uv run python benchmark/run.py termination \
  --model plain/puzzles-5m \
  --max-positions 4096 \
  --batch-size 128 \
  --output benchmark/results/termination_plain_puzzles_5m.jsonl
```

This reports whether `<eos>` is too attractive on non-terminal positions:

- `eos_top1_rate`
- `avg_eos_prob`
- `median_eos_rank`

Fixed-position move-quality benchmark:

```bash
uv run python benchmark/run.py move-quality \
  --model plain/puzzles-5m \
  --max-positions 512 \
  --stockfish-depth 4 \
  --output benchmark/results/move_quality_plain_puzzles_5m.jsonl
```

This asks the model for the best legal move on validation positions and scores
the centipawn loss with Stockfish.

Held-out loss by ply:

```bash
uv run python benchmark/run.py loss-by-ply \
  --models plain/games-500k plain/puzzles-5m \
  --max-games 4000 \
  --output benchmark/results/loss_by_ply_models.jsonl
```

This buckets next-token validation loss by game ply, which is useful for seeing
whether a model is mostly strong in openings or holds up deeper into games.

Full scaffolded games:

```bash
uv run python benchmark/run.py games \
  --game-mode stockfish \
  --model plain/puzzles-5m \
  --stockfish-elo 500 \
  --stockfish-depth 1 \
  --games 50 \
  --batch-size 16 \
  --output benchmark/results/games_plain_puzzles_5m_sf500.jsonl
```

Head-to-head:

```bash
uv run python benchmark/run.py games \
  --game-mode h2h \
  --model plain/puzzles-5m \
  --opponent-model plain/games-500k \
  --games 50 \
  --batch-size 16 \
  --output benchmark/results/games_plain_puzzles_5m_vs_games_500k.jsonl
```

Summarize artifacts:

```bash
uv run python benchmark/run.py summarize benchmark/*.jsonl
```

Modal H2H benchmark:

```bash
modal run modal_benchmark.py \
  --model-a plain/games-3m \
  --model-b plain/puzzles-5m \
  --games 50 \
  --batch-size 64 \
  --shards 1
```

`modal_benchmark.py` is only a remote wrapper around `benchmark/run.py games
--game-mode h2h`. It exists because the checkpoints and A100 runtime live on the
Modal volume.

The Modal wrapper writes the aggregate summary and every game record to a local
JSONL file under `benchmark/modal_h2h_*.jsonl` by default. Each `game` record
contains the full SAN move list, color assignment, result, outcome, termination,
and ply count.

Modal Stockfish benchmark:

```bash
modal run modal_benchmark.py \
  --mode stockfish \
  --model plain/puzzles-5m \
  --stockfish-elo 500 \
  --stockfish-depth 1 \
  --games 200 \
  --batch-size 64 \
  --shards 4 \
  --seed 5000 \
  --output benchmark/results/sf500_plain_puzzles_5m_200.jsonl
```

Run the same command with `--stockfish-elo 1000` or `--stockfish-elo 1500` and
an `sf1000_...` or `sf1500_...` output name for the other baselines. The Modal image installs `fairy-stockfish` because
the low Elo baselines are below regular Stockfish's usual supported Elo range.
It uses the upstream Fairy-Stockfish release binary rather than Debian's package
so that the low UCI_Elo settings are accepted.

For the four-model comparison set, run the Elo baselines for each model:

```bash
for model in plain/games-500k plain/puzzles-500k plain/games-5m plain/puzzles-5m; do
  safe_model="${model//\//_}"
  for elo in 500 1000 1500; do
    modal run modal_benchmark.py \
      --mode stockfish \
      --model "$model" \
      --stockfish-elo "$elo" \
      --stockfish-depth 1 \
      --games 200 \
      --batch-size 64 \
      --shards 4 \
      --seed 5000 \
      --output "benchmark/sf${elo}_${safe_model}_200.jsonl"
  done
done
```

For the `games-5m` training arc, use [scripts/actual_5m_runbook.md](/Users/sparshith/workspace/nanoDanya/scripts/actual_5m_runbook.md). The main benchmark is:

```bash
modal run modal_benchmark.py \
  --model-a plain/games-5m \
  --model-b plain/puzzles-5m \
  --games 200 \
  --batch-size 64 \
  --shards 4 \
  --seed 5000 \
  --output benchmark/results/h2h_plain_games_5m_vs_puzzles_5m_200.jsonl
```

## Interpretation

- Raw metrics measure model behavior before runtime fixes.
- Termination metrics measure raw `<eos>` propensity before runtime fixes.
- Move-quality metrics measure the model's legal-move preference.
- Loss-by-ply metrics measure next-token fit as the game gets deeper.
- Game metrics measure the model plus legal masking, EOS policy, sampling, and
  opponent settings.

Do not use game win rate as the primary legality metric. Do not use legality
rate as the primary strength metric.

## Artifact Policy

Generated game logs and rendered reports are artifacts, not source:

- `benchmark/*.jsonl`
- `benchmark/*.html`

Keep durable findings in `knowledge/wiki/evaluation/` as summarized markdown,
not as raw run output.
