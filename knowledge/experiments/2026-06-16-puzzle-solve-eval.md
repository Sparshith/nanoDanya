# Puzzle-Solve Eval: Rating-Binned Go/No-Go

Date: 2026-06-16. Backfilled from session notes during the 2026-07-04 restructure.

## Data contract

`puzzle_games_ndjson.txt` (3.2M games, per-move Stockfish evals) joins
`puzzle_metadata.txt` (5.8M games keyed by game id) at 100%; metadata is a strict
superset. Each puzzle: `{puzzle_id, fen, moves (UCI), rating, move_num}`.

- `move_num` is the 1-indexed ply of the puzzle's lead-in move; the position after
  `move_num - 1` plies of the game equals `fen` (verified by replay).
- Solution `moves` is always even-length: `moves[0]` is the opponent lead-in, solver
  moves are `moves[1::2]`, forced replies `moves[2::2]`.
- Raw files on the Modal volume at `/data/puzzles_raw/`.

## Rig

`benchmark/run.py puzzles` subcommand, on Modal via `modal_benchmark.py --mode puzzles`.
The model has no FEN input, so a puzzle is presented by replaying the real game history
up to the lead-in move. Multi-step solve re-batches survivors each ply. Stratified
streaming sampler fills rating bins. Metrics per bin: `first_move_acc` and
`full_solve_acc` (steps the whole line with forced replies; any mating move accepted,
the Lichess exception).

```bash
uv run modal run modal_benchmark.py --mode puzzles --per-bin 400 --write-positions
```

## Result (15M-game model, `/data/actual_15m/...L16_H8_E1024_best.pt`, no puzzle training)

Clean monotonic decline with rating, not flat: full-solve 76.8% (600-700) down to
2.5% (2700-2800); first-move 79% down to 39%. The first-move vs full-solve gap widens
from 2pts on easy puzzles to 36pts on hard ones: hard puzzles fail on the deep line,
not move one.

Verdict: the metric discriminates. GO for the tool-assisted phase.
