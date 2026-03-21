# nanoDanya

## Rules

- Do not modify Jupyter notebooks (.ipynb files). Use plain Python scripts instead.
- Do not modify the nanochat submodule.
- Follow coding style guidelines from the user-level CLAUDE.md (flat scripts, no unnecessary abstractions, match existing patterns).

## nanochat

nanochat is installed as an editable local dependency via `pyproject.toml`. Import directly:

```python
from nanochat.gpt import GPT, GPTConfig
```

Do not use `sys.path` hacks to import nanochat.

## Data

Main dataset: `data/puzzle_games_ndjson.txt` (3.2M Lichess games with per-move Stockfish evals). Puzzle metadata mapping game IDs to puzzle positions: `data/puzzle_metadata.txt`.

Other files in `data/` (raw/, processed/, puzzle_weighted/, eval/, etc.) are older/smaller datasets kept for reference.

## Modal Training

Run training on Modal with:

```bash
uv run modal run modal_train.py --datasets <dataset_names> --script <training_script>
```

Examples:

```bash
# default (processed dataset, training/train.py)
uv run modal run modal_train.py

# weighted puzzle training
uv run modal run modal_train.py --datasets puzzle_weighted --script training/train_weighted.py

# multiple datasets (comma-separated)
uv run modal run modal_train.py --datasets eval,puzzle_weighted --script training/train_eval.py
```

`--datasets` is a comma-separated list of subdirectory names under `data/` (and matching volume paths). `--script` is the training script path relative to project root.

Do not create separate modal_train files for different training runs. Always use the same `modal_train.py` with different arguments.
