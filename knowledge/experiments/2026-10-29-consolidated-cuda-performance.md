# Consolidated trainer: H100 correctness and performance

Completed under a user-authorized $10 benchmark cap. Keep a 64-window microbatch
on this H100: the consolidated loop preserves the old loop's throughput, while
smaller microbatches trade speed for lower memory. This establishes a single-GPU
baseline, not a training speedup or an Elo result.

## Measurements

NVIDIA H100 80GB HBM3, four CPU cores, 32 GiB host RAM; PyTorch 2.5.1+cu124,
CUDA 12.4, NumPy 2.4.6. L16/H8/E1024, 512-token windows, a fixed global batch of
64 windows. Each row has 25 warmup updates followed by 200 timed updates.

| Recipe / loop | Microbatch | ms/update | Targets/sec | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| Old mixed loop | 64 | 116.09 | 247,794 | 28.70 |
| Consolidated mixed | 64 | 116.96 | 245,971 | 28.70 |
| Consolidated mixed | 32 | 123.46 | 233,009 | 16.74 |
| Consolidated mixed | 16 | 135.77 | 211,877 | 10.33 |
| All datasets once | 64 | 116.48 | 241,830 | 28.70 |
| All datasets once | 32 | 123.53 | 228,038 | 16.74 |
| All datasets once | 16 | 135.84 | 207,360 | 10.33 |

The same-data comparison is 0.74% slower for the consolidated loop, with no
material throughput regression in this run. Peak reserved memory for its
64-window arms was 31.53 GiB, also comfortably below capacity. Timings include
sampling, CPU-to-GPU copies, forward/backward, gradient clipping and fused AdamW,
with CUDA synchronization at both timing boundaries. Validation and checkpoint
I/O are excluded. Compilation caches are shared in this process, so first-update
times in the receipts are not independent cold-start comparisons.

Loading and hashing the real datasets took 29.91 seconds; constructing and
shuffling both coverage indices took 35.72 seconds. Their 17,853,607 windows
require 278,963 updates. At microbatch 64 this projects to **9.03 hours of update
time**, plus startup, validation and checkpointing. This is a short-run projection,
not an observed full-pass duration. One pass through both sources has a different
data balance and update count from the old 50/50 recipe.

## Correctness

The old and new mixed samplers produced exactly identical inputs and targets
for ten checked batches. All measured mixed arms consumed identical target
counts, as did the three coverage arms. Corpus hashes match the previous
full-Leela preparation receipts.

For meaningful nonzero gradients, each arm started from the same mixed-200k
checkpoint (SHA256 `f3340d750c3aad73ec8f16f0f26716b6fc0a2d66c66c6dd4acd582a9b90a4202`)
and fresh AdamW at 3e-5, betas (0.9, 0.95), weight decay 0.1. One identical batch
was compared before timing; weights and optimizer were reset before warmup.
The consolidated mixed arms' absolute loss differences were at most 1.2e-7,
gradient relative L2 differences 0.348–0.378%, and weight-update relative L2
differences 4.09–4.41%. They passed the predeclared thresholds of 0.005 absolute
loss, 3% gradient relative L2 and 10% update relative L2. These are numerical
tolerance checks under BF16, not bitwise GPU equivalence or a long-run trajectory
guarantee. The CPU tests separately verify loss semantics and exact resume.

Every timed update checked finite loss and gradient norm. This benchmark exercises
the production sampler and backward path; it does not validate a complete CUDA
save/resume cycle. No checkpoint was promoted or overwritten, and all temporary
weight changes were discarded. The October 28 full-Leela run remains paused.

## Budget and receipts

Existing `modal_train.py`, `MODAL_PROFILE=second`, one H100 container, no configured
retries, 3,600-second function timeout and an independent 3,720-second local
watchdog. Reserved $4.54 for that attempt including startup allowance, plus $2
for build/storage/other overhead. No second attempt was needed.

The function receipt is 395.46 seconds, approximately **$0.483 compute**. Charging
all 416.66 seconds of local elapsed time at the full resource rate gives a
conservative **$0.509 estimate**. These are estimates from
[Modal's published rates](https://modal.com/pricing), not an invoice; the separate
$2 overhead allowance also fits within the $10 cap.

App `ap-7g6USCEUFItC1MOnOgl81v` is verified stopped with zero tasks. Receipts,
per-update timings, numerical errors, audit and source snapshots are in
`output/cuda-recipe-20261029/` and
`/data/experiments/cuda-recipe-20261029/` on `nanodanya-data`.
Runner: `scratch/benchmark_training_cuda.py`; audit:
`scratch/report_cuda_benchmark.py`. No full training run, multi-GPU run or Elo
evaluation was launched.
