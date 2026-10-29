# Probes

Mental model:

```text
positions -> hidden states -> linear probes -> metrics
```

## Pipeline

```bash
python probes/build_positions.py --data data/processed/val.bin --out data/probes/positions_val_small.pt
python probes/cache_hiddens.py --model plain/games-500k --positions data/probes/positions_val_small.pt --out data/probes/hiddens_plain_games_500k_val_small.pt
python probes/run.py --positions data/probes/positions_val_small.pt --hiddens data/probes/hiddens_plain_games_500k_val_small.pt
```

Add Stockfish labels when building positions if you want the eval probe:

```bash
python probes/build_positions.py --stockfish /opt/homebrew/bin/stockfish
```

## Core Probes

- `pieces`: board reconstruction, `64 x 13` piece classes.
- `side_to_move`: whose turn it is.
- `in_check`: whether the current side is in check.
- `legal_moves`: legal continuation set over tokenizer move IDs.
- `stockfish_eval`: eval bucket, only when positions include Stockfish labels.

The primary output is `model, layer, task, metric, value, baseline`. Rendered HTML
and charts are reports, not core probe logic.

## Inspect a game

Recovered from `7463f94^`. Board readouts can now be saved with
`probes/run.py --heads-out data/probes/heads --pieces-only` and applied with
`probes/inspect_game.py --model <checkpoint-path> --pgn <game.pgn> --ply 71
--heads data/probes/heads`. Ply 71 means the first 71 half-moves have been played;
the next move is excluded from the input. Use the same checkpoint for both steps.

Train and inspect on Modal without downloading model weights:

```bash
modal run modal_probes.py::main \
  --pgn data/traces/lichess/2026-09-19-exact-10k/loss_GZrvqz3j.pgn \
  --ply 71 \
  --model /data/checkpoints/grpo/own-games-exact-10k-20260918/best.pt \
  --out data/probes/GZrvqz3j
```

This fits board readouts on 100 validation games, splitting 80/20 by game,
then decodes the position before `36...g5` (which allowed `37.Rh2#`). The target
game is never used to fit or select the readout. Layer numbers are zero-based;
selection uses held-out occupied-square accuracy. The JSON report includes
all layers, per-square predictions, and probabilities. Models, cached states,
and readouts stay on the Modal volume under the directory named in the report.

Reuse saved readouts for a sequence of decisions, without retraining:

```bash
modal run modal_probes.py::timeline \
  --run /data/probes/20260920-191333 --plies 61,63,65,67,69,71 \
  --out data/probes/GZrvqz3j/timeline.json
```

A decoded board measures linearly recoverable information. Readout errors do
not establish that the model believed a different board, and a correct board
does not establish that it can anticipate mate. Sparse endgames can be harder
than the aggregate validation score suggests.
