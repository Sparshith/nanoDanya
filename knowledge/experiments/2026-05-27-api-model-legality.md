# API Model Legality Comparison

Date: 2026-05-27 (artifact mtimes). Backfilled 2026-07-04 from `benchmark/results/`.

Frontier chat models given the same next-move task as nanoDanya: predict one SAN move
from the game history, scored for legality. Same 4,096 validation positions as the
model legality runs (`actual_5m` val split, max ply 140), via OpenRouter, 2 retries.
The blog states temperature 0; the run configs do not record it.

Unparseable responses count against legality (see `parseable`). nanoDanya rows are
raw top-1 legality from `benchmark/results/legality_*_on_actual_5m_4096.jsonl` on the
same positions.

| Model | Legal | Opening | Middle | End | Parseable |
|---|---:|---:|---:|---:|---:|
| nanoDanya 15M games (L12) | 99.71% | 99.91% | 99.63% | 99.66% | - |
| nanoDanya 5M puzzles | 99.61% | 99.91% | 99.59% | 99.15% | - |
| nanoDanya 5M games | 99.46% | 99.91% | 99.38% | 98.98% | - |
| gemini-3.1-flash-lite | 98.95% | 100.0% | 98.64% | 98.30% | 100% |
| gemini-3.5-flash | 98.17% | 99.07% | 97.86% | 97.79% | 100% |
| claude-opus-4.7 | 93.19% | 99.17% | 91.48% | 89.27% | 100% |
| gpt-5.4 | 91.04% | 98.52% | 89.75% | 82.62% | 99.95% |
| grok-4.3 | 90.53% | 97.31% | 88.84% | 85.01% | 100% |
| nanoDanya 500k puzzles | 96.02% | 99.72% | 94.44% | 95.74% | - |
| nanoDanya 500k games | 95.92% | 99.63% | 94.69% | 94.21% | - |
| gpt-5.4-mini | 74.44% | 90.83% | 69.70% | 63.88% | 100% |
| claude-sonnet-4.6 | 47.24% | 96.39% | 32.77% | 16.70% | 79.4% |

Artifacts: `benchmark/results/api_legality_*_4096.jsonl`
(`api_legality_stronger_models_4096.jsonl` holds gemini-3.5-flash; it also contains
claude-opus-4.7 and gpt-5.4 duplicates of the dedicated files).

## Read

- The 5M+ nanoDanya models beat every tested frontier model on this task; even the
  500k models beat everything except the two Geminis.
- The universal pattern is phase decay: everyone is near-perfect in the opening and
  degrades through middlegame and endgame. nanoDanya 15M is the only model that
  holds ~99.7% across all phases.
- claude-sonnet-4.6's number is dominated by parse failures (79.4% parseable), so
  its 47% mixes format failure with chess failure.
- Exact model ids matter for reproduction; the blog's informal names map to the ids
  above (e.g. "Gemini 3.1 Flash" is `google/gemini-3.1-flash-lite`).
