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
