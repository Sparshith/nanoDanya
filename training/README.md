# Training

Use `train.py` with a JSON recipe for next-move pretraining. Dataset names are
arbitrary; the trainer has no built-in human/Leela split.

```text
train.py          Recipe-only entry point
pretrain.py       Shared updates, validation and checkpointing
recipe.py         Configuration, dataset loading and sampling
recipes/          Experiment settings (JSON)
*_batches.py      Batch builders for pretraining and open whole-game experiments
```

Distillation (`train_distill.py`) and RL (`train_exact_policy.py`) keep their
separate entry points. Model and checkpoint helpers remain shared here.

```sh
MODAL_GPU=H100 MODAL_CPU=4 MODAL_MEMORY=32768 uv run modal run modal_train.py \
  --recipe training/recipes/l16-all-once.json \
  --checkpoint-dir /data/checkpoints/plain/all-once
```

Add `--sources human` or `--sources leela` and a **different checkpoint directory**
for a single-source run. This command launches training; it is not a dry run.
One pass through differently sized datasets is not a matched-token ablation.

The recipe separates three choices:

- `sampling.mode: coverage`: shuffle a combined window index and visit every
  move/EOS target in every selected training dataset exactly once. A larger
  dataset contributes more targets. The final batch is padded without repeating
  examples. BOS and padding never receive loss. Games longer than the context
  use overlapping context, with each target supervised once.
- `sampling.mode: mixture`: set `windows_per_source` and `steps` for random
  windows with replacement. Missing-history fragments and BOS targets are
  ignored. To cover one source once while sampling the others, replace `steps`
  with `coverage_source: "source_name"`; its coverage determines the run length.
- `loss_weights: null`: average over all supervised tokens in the update.
  Explicit source weights, such as `{"human": 0.5, "leela": 0.5}`, average each
  source separately, then combine those averages. This requires mixture mode,
  and supervised targets from every source in each update. An update with an
  entirely masked source fails instead of silently changing the objective.

`l16-mixed-windows.json` expresses the earlier 32/32, 200k-update recipe.
These are new runs, not bit-for-bit reproductions of old experiments: validation
seeds and checkpoint selection differ. `best.pt` selects the lowest held-out CE
(including EOS), with configured source weights or estimated training token
proportions; `final.pt` is the fixed training endpoint for comparisons.

`batch_size` is the number of windows per optimizer update. `microbatch_size`
controls how many are processed at a time; accumulated gradients use the full
batch's token counts. Execution is currently single-GPU, with optional compilation
and BF16 on CUDA. CPU supports small local checks. `model_type` can be `gpt` or
`student`; quantization, distillation and RL retain their separate trainers.

CUDA training compiles the model and token-weighted loss together. The L16
presets use `compile_mode: "max-autotune"` for longer runs; use `"default"` for
short probes where startup matters more. Cold H100 autotuning took about seven
minutes. The Modal recipe launcher keeps a PyTorch-version/GPU-specific compiler
cache on the volume; `TORCHINDUCTOR_CACHE_DIR` in `--env-overrides` overrides it.
Loss calculations use FP32 while model computation uses BF16. Compilation
changes rounding, so it does not promise identical weights.

The run saves its resolved recipe, dataset/tokenizer hashes, validation metrics,
token counts and atomic checkpoints. `latest.pt` includes optimizer and sampling
state. Rerun the same command and directory to resume; changed recipes or data
are rejected. Open Leela/whole-game runs retain their original scratch trainers.
The Modal launcher commits checkpoints during training and at exit.

For local synthetic-data checks, use paths in a local recipe with `device: "cpu"`:

```sh
PYTHONPATH=.:nanochat .venv/bin/python -m training.train \
  --recipe /tmp/tiny-recipe.json --output /tmp/tiny-training --stop-after-step 2
```

`--stop-after-step` pauses without shortening the planned learning-rate schedule.
Remove it to finish. `--time-limit` stops between updates and then saves; on Modal
the default reserves 600 seconds before the function timeout for final validation
and checkpointing.

The main entry point requires a recipe. The previous environment-variable trainer
has been removed; its implementation remains in Git history.
Closed board/piece experiment trainers and their dedicated dependencies have been
removed. Board-conditioned checkpoint loading remains supported.

- `pretrain.py`, `recipe.py`: shared next-move update loop, data loading and sampling.
- `train_distill.py`: tied-embedding browser student training; `DISTILL_ALPHA=0`
  trains from recorded moves only and does not load a teacher. The default `0.5`
  mixes recorded-move CE with teacher KL. Both modes support INT4-aware training
  with `QAT=1` and a separate `CKPT_DIR` initialized through `INIT_FROM`.
- `train_exact_policy.py`: exact expected-reward updates over legal moves, with checkpoint selection and resume.
- `common.py` and `grpo.py`: shared model, checkpoint, reward-table, and exact-policy loss helpers.

The open Leela pilot, whole-game and paused full-Leela experiments live in the
Git-ignored `scratch/` directory, alongside a few operational utilities. Their
entry points and tests are listed in `scratch/README.md`.

Run the pretraining tests with:

```sh
PYTHONPATH=.:nanochat .venv/bin/python -m pytest tests/pretraining
```

These use production data loading, sampling, loss, training and checkpoint loading
with tiny GPT/student models on CPU. They check exact target coverage and context,
weighted gradients, FP32 loss precision, real parameter updates, resume integrity,
incompatible-input rejection, and both recipe CLIs. CUDA compilation/performance
requires separate GPU verification. Use `pytest tests` for the full repository suite.
