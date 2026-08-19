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

## Data & Models

All datasets, checkpoints, training, inference, and benchmarking live on the Modal volume `nanodanya-data`. Do not download data or checkpoints locally; run experiments on Modal (`modal_train.py`, `modal_benchmark.py`).

Volume layout: `puzzles_raw/` (main dataset: `puzzle_games_ndjson.txt`, 3.2M Lichess games with per-move Stockfish evals, plus `puzzle_metadata.txt` mapping game IDs to puzzle positions), `datasets/` (prepared train/val bins), `checkpoints/` (plain, weighted, eval-aware), `archive/` (source shards, probes). See `README_LAYOUT.md` on the volume.

Local `data/` holds only small working files; it is gitignored and not the source of truth.

## Knowledge

`knowledge/` is the project research memory: `STATE.md` (current story, champion model, open questions), `log.md` (append-only dated entries), `experiments/` (one file per significant run or benchmark). For questions about project state or past results, read `STATE.md` first.

After a significant training run, benchmark, or conclusion change, update it as part of the work: append a `log.md` entry, add an `experiments/` file if the run warrants one, and keep `STATE.md` current. Keep it flat; do not add new directories, templates, or index files.

## Scratch

One-off analysis and probe scripts go in `scratch/` (gitignored, but still shipped to Modal). Before a scratch script is done, file its numbers and conclusion in `knowledge/log.md`; the script itself is disposable. A script graduates to a tracked directory (`data_prep/`, `benchmark/`, ...) only when it gets rerun against different inputs.

## Plots

- Do not send plots or visualizations to external services. Keep all plots on the local disk.
- Make each plot as one HTML file that contains all its resources.
- Put each plot in its own subfolder in the `plot/` directory.
- Give the subfolder a name that identifies the contents of the plot.
- Put the plot data and the build script in the same subfolder.
- The `plot/` directory is not part of the git repository, and it does not go to Modal.

## Modal Training

Run training on Modal with:

```bash
uv run modal run modal_train.py --datasets <dataset_names> --script <training_script> --gpu <gpu_type>
```

Examples:

```bash
# plain training on the 5m games dataset
uv run modal run modal_train.py --datasets datasets/games/5m --script training/train.py \
  --env-overrides "DATASET_DIR=datasets/games/5m,CKPT_DIR=/data/checkpoints/plain/games-5m"

# weighted puzzle training
uv run modal run modal_train.py --datasets datasets/puzzles/5m --script training/train_weighted.py \
  --env-overrides "DATASET_DIR=datasets/puzzles/5m,CKPT_DIR=/data/checkpoints/weighted/puzzles-5m"
```

`--datasets` is a comma-separated list of volume paths (symlinked into `data/`; pass the same path as `DATASET_DIR`). `--script` is the training script path relative to project root. GPU type is set via the `MODAL_GPU` env var (default A100, use A10G for cheaper runs); the `--gpu` flag is a no-op because the GPU is fixed at import time.

Do not create separate modal_train files for different training runs. Always use the same `modal_train.py` with different arguments.
