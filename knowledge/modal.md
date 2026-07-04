# How we use Modal

Everything lives on Modal: datasets, checkpoints, training, benchmarks, serving.
Nothing model-sized is stored locally. Volume: `nanodanya-data`, mounted at `/data`.

## Apps

- `nanodanya-train` (`modal_train.py`): training runs
- `nanodanya-benchmark` (`modal_benchmark.py`): all benchmark modes
- `nanodanya-chess` (`inference/serve.py`): the serving endpoint the lichess bot calls

## Volume layout

```
puzzles_raw/          main dataset: puzzle_games_ndjson.txt (3.2M games with
                      per-move Stockfish evals) + puzzle_metadata.txt
datasets/games/       500k, 5m, 15m   (train.bin, val.bin, meta.pkl per dir)
datasets/puzzles/     500k, 5m        (also *_weights.bin)
datasets/eval/base    tokens + weights + evals bins, for eval-aware work
checkpoints/plain/    games-500k, games-5m, games-15m, puzzles-500k, puzzles-5m
archive/              dated source shards, backup only
README_LAYOUT.md      volume-side copy of this layout
```

A dataset dir is a contract: `meta.pkl` defines the tokenizer, and checkpoints are
only compatible with data that shares it.

## Training

```bash
uv run modal run modal_train.py --datasets datasets/games/5m --script training/train.py \
  --env-overrides "DATASET_DIR=datasets/games/5m,CKPT_DIR=/data/checkpoints/plain/games-5m"
```

- `--datasets`: comma-separated volume paths, symlinked into local `data/`; pass the
  same path as `DATASET_DIR`.
- GPU comes from the `MODAL_GPU` env var (default A100; A10G for cheap runs). The
  `--gpu` flag is a no-op because the GPU is baked into the decorator at import.
- Timeout is 12h; the trainer checkpoints periodically and resumes from the periodic
  checkpoint, so preempted or timed-out runs just get relaunched. Resume validates
  all six config keys and raises on mismatch instead of overwriting.
- Env knobs (`training/train.py` via `train_loop`): `MAX_ITERS`, `BATCH_SIZE`, `LR`,
  `GRAD_ACCUM_STEPS`, `EVAL_INTERVAL`, `CKPT_INTERVAL`, `VAL_BATCHES`,
  `EARLY_STOP_METRIC/PATIENCE/MIN_STEPS/MIN_DELTA`, `CKPT_NAME`, `CKPT_DIR`,
  and arch overrides `N_LAYER/N_HEAD/N_KV_HEAD/N_EMBD`.
- Smoke-test pattern: `MAX_ITERS=200,EVAL_INTERVAL=50,CKPT_INTERVAL=100` with
  `CKPT_DIR=/data/checkpoints/smoke`, then `modal volume rm ... checkpoints/smoke -r`.
- Torch is pinned (`2.5.1+cu124`) in the image; keep local and image versions in
  sync when touching checkpoint serialization.

## Benchmarks

```bash
uv run modal run modal_benchmark.py --mode <mode> --model <registry-alias>
```

Modes: `h2h` (shardable), `stockfish` (shardable, fairy-stockfish in the image),
`legality`, `move-quality` (cp loss vs Stockfish), `puzzles` (rating-binned solve
rates + PNG curve). Results land in `benchmark/results/` (gitignored, kept locally
as the evidence base for knowledge/ notes). Model refs are registry aliases from
`model_registry.py`.

## Volume CLI

```bash
uv run modal volume ls nanodanya-data <path>
uv run modal volume rm nanodanya-data <path> -r
uv run modal volume put nanodanya-data <local> <remote> --force
```
