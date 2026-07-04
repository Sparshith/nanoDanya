# Scale Matrix: 5x5 H2H and Stockfish Ladder

Date: 2026-05-13 to 2026-05-31 (artifact mtimes). Backfilled 2026-07-04 from
`benchmark/results/`; all numbers below re-extracted from the JSONL `games_summary`
records, and they match the blog post.

Extends the four-edge suite in
[2026-05-11-games-vs-puzzles-scale-h2h.md](2026-05-11-games-vs-puzzles-scale-h2h.md)
to the full matrix over five checkpoints, adding `plain/games-5m` and the 15M-game
model.

## Protocol

200 games per pairing, temperature 0.8, max 200 plies, legality mask on, batch 64.
"15M" is the L12 arch, 200k-step best checkpoint, at run time
`/data/actual_15m/chess_actual_15m_uniform_L12_H6_E768_best.pt`, since registered as
`plain/games-15m/l12` (`checkpoints/plain/games-15m/l12_best.pt`).

## H2H matrix (row model's score)

| | 500k games | 500k puzzles | 5M games | 5M puzzles | 15M games |
|---|---:|---:|---:|---:|---:|
| **500k games** | - | 62.0% | 9.0% | 13.0% | 7.0% |
| **500k puzzles** | 38.0% | - | 7.75% | 6.5% | 4.75% |
| **5M games** | 91.0% | 92.25% | - | 47.25% | 30.25% |
| **5M puzzles** | 87.0% | 93.5% | 52.75% | - | 32.5% |
| **15M games** | 93.0% | 95.25% | 69.75% | 67.5% | - |

Artifacts: `benchmark/results/h2h_*_200.jsonl`.

Key readings:

- **The matched-scale source test is settled: a tie.** `plain/games-5m` vs
  `plain/puzzles-5m` = 47.25 / 52.75 (W75 D39 L86). Puzzle-linked source gives no
  meaningful edge once actual games reach the same scale. Scale was the whole story.
- **15M is the clear champion**, beating both 5M models by ~68-70%.
- `plain/games-3m` runs (not in the blog matrix): 25.25% vs `plain/games-5m`,
  22.5% vs `plain/puzzles-5m`. Consistent with the scale trend.

## Architecture at 15M

`L16_H8_E1024_best` vs `L12_H6_E768_best` on the same 15M data: 54.75% over 200
games, 54.45% over 1000 games (`h2h_actual_15m_L16_H8_E1024_best_vs_L12_H6_E768_best_1000.jsonl`).
Real but small; data scale moved the needle far more than architecture.

## Stockfish ladder (200 games each, fairy-stockfish, elo-limited, 0.02s/move)

| Model | sf500 | sf1000 | sf1500 |
|---|---:|---:|---:|
| 500k games | 91.5% | 7.75% | 1.75% |
| 500k puzzles | 88.25% | 7.5% | 1.5% |
| 5M games | 97.5% | 62.5% | 29.25% |
| 5M puzzles | 96.75% | 53.5% | 27.75% |
| 15M games | 97.0% | 78.25% | 42.5% |

Artifacts: `benchmark/results/sf{500,1000,1500}_*_200.jsonl`. sf500 saturates for the
5M+ models; sf1000/sf1500 are the discriminating levels now.

## Side results from the same period

- `puzzle_highrated_500k` (high-rated puzzle-linked 500k variant) vs 500k games
  baseline: 48.5%. Another 500k-scale source variant that failed to beat the baseline.
- `eval-aware/weighted-ft` vs `plain/puzzles-5m`: 36% over 50 games. The eval
  fine-tune line is not competitive with the plain champions.
- Temperature-0 H2H is degenerate (deterministic games collapse to one game per
  color: 0% or exactly 50% outcomes in the temp0 artifacts). Keep H2H at temp 0.8.

## Caveats

- Two 15M-vs-5M runs exist with different scores (63.75% and 69.75%) against the
  same checkpoint path; the checkpoint file was updated between runs. The 69.75%
  run (`h2h_actual_15m_200k_*`) is the 200k-step version and matches the blog.
- All games use the legality mask; this measures playing strength, not raw legality.
