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
batch's token counts. Single-source runs support multiple GPUs through DDP, with
optional compilation and BF16 on CUDA. CPU supports small local checks. `model_type` can be `gpt` or
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

Single-source runs support DDP via `torchrun --standalone
--nproc_per_node=8 -m training.train ...`, or `MODAL_GPU=H100:8` through the Modal
launcher. `batch_size` remains global and must divide evenly across workers;
`microbatch_size` is a per-worker maximum. Each worker deterministically builds
the same global batch and trains on its own slice. This preserves coverage and
token normalization, including an entirely padded final slice. Data indexing and
CPU batch construction are currently replicated. Validation and file writes run
on rank zero. Resume requires the same world size.

Run the bounded, single-source H100 comparison with an explicit budget:

```sh
MODAL_PROFILE=second MODAL_GPU=H100 modal run --detach modal_scaling.py --source human --budget 50
```

The launcher runs CPU correctness tests first, then sequential 1/2/4/8-H100 jobs.
Each job measures three blocks of 200 updates after 20 warmup updates, with global
batch 64. Startup/compilation is reported separately; timed updates include data
sampling, transfers, backward, gradient synchronization, clipping, and AdamW.
The timing loop uses a fixed peak learning rate on all GPU counts; it measures
throughput, not learning quality. Validation/checkpoint overhead is excluded from
steady-state measurements. The operational mode uses the actual training schedule.
Results and GPU telemetry are saved on the volume under `benchmarks/scaling/`.
The local `benchmark/results/scaling-*` folder records the budget and call ID;
retrieve completed JSON results with `modal volume get`. A remote CPU controller
keeps the sequential sweep running if the local connection drops. The CPU
controller is non-preemptible so GPU-call wall-clock limits survive GPU preemptions.
Runtime limits, CPU/memory limits, no error retries, and the cached pricing snapshot
bound the sweep's
estimated cost. Actual billing may differ from the estimate.

Use `--operational --counts 1,8` for a separate 20-update run including startup,
initial/final validation, checkpoint writes to the volume, and actual Volume
commits. This is a short-run overhead measurement, not steady-state throughput.
Its runtime cap is shorter.

Linux tests additionally launch two Gloo workers to check raw gradient equality
with uneven token counts and an empty worker slice, exact target coverage, and
resume equivalence. Those integration tests are skipped on non-Linux hosts.

CUDA DDP compiles the full loss graph with `torch._dynamo.config.optimize_ddp=False`.
This avoids interleaving per-bucket CUDA graph capture with pending NCCL work,
which prevented the initial eight-GPU probe from completing warmup within its
budgeted window. It trades communication overlap for simpler capture behavior;
scaling measurements must use this same setting across distributed runs.
