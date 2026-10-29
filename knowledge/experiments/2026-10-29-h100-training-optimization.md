# H100 training optimization, 2026-10-29

Completed under a separate $20 budget after the
[$10 consolidated-trainer verification](2026-10-29-consolidated-cuda-performance.md).
The final implementation is 7.1% faster on the measured workload. Conservative
compute estimate: $4.35. All seven apps are stopped with zero tasks. No full
training, Elo evaluation, or resumption of the paused full-Leela run occurred.

## Protocol

Single H100, PyTorch 2.5.1+cu124, FP32 master weights and BF16 computation, fused
AdamW. Global batch remains 64 windows of 512 tokens. Fixed source checkpoint:
`/data/checkpoints/plain/leela-human-scratch-20261013/l16_best.pt`, SHA256
`f3340d750c3aad73ec8f16f0f26716b6fc0a2d66c66c6dd4acd582a9b90a4202`.
Each candidate starts from the same weights and optimizer settings, uses the same
seeded real-data batches, warms up for 25 updates, and times 200 complete updates.
Timing includes sampling, transfer, forward/backward, clipping, and AdamW, with
CUDA synchronization. Compilation, validation, and checkpoint writes are separate.
Timing and one-update comparisons use fresh AdamW at LR 3e-5, betas (0.9, 0.95),
weight decay 0.1 and gradient clip 1.0. Production smoke runs use the recipe's
3e-4 peak LR and its schedule over the nine-update excerpt.

Numerical acceptance limits were set before results: loss absolute error <=0.005,
clipped-gradient relative L2 <=3%, and first AdamW weight-change relative L2 <=10%.
These test implementation consistency under BF16, not equivalent final Elo or
bitwise-identical training. Temporary updated weights are never promoted.

Jobs run sequentially through the existing `modal_train.py`, profile `second`,
volume `nanodanya-data`, with one H100, four CPU cores, 32 GiB RAM, zero retries,
a function timeout and local cancellation watchdog. Each stage reserves its
maximum runtime cost before launch, plus a shared $3 overhead allowance. The
coordinator stops each app when its stage ends.

## Profile and candidate measurements

The synchronized baseline breakdown was approximately 0.92 ms sampling/counting,
0.17 ms transfer, 34.62 ms forward, 2.89 ms loss, 74.83 ms backward, and 4.84 ms
clipping/optimizer. Data handling was under 1%; matrix computation dominated.
Profiler operator totals overlap parent operations and kernels, so they must not
be summed as independent costs.

| Same-allocation autotune comparison | ms/update |
|---|---:|
| Original model-only compilation | 117.83 |
| Compile model and loss together, default | 113.21 |
| Combined loss, reduce-overhead | 112.24 |
| Combined loss, max-autotune without CUDA graphs | 110.22 |
| Combined loss, max-autotune with CUDA graphs | 109.74 |

All four candidates passed the initial numerical checks. Gradients differed by
about 0.91% relative L2 and first AdamW changes by 8.83%; loss error was about
0.0000043. The small gradient discrepancy can change AdamW's first step on
near-zero coordinates, so update error is reported separately.

A fresh allocation measured 120.00 ms baseline, 110.72 ms combined loss with
max-autotune, and 109.65 ms after additionally packing Q/K/V projections. The
roughly 1% incremental gain does not yet justify maintaining another attention
implementation. No packed-projection change is adopted in the trainer.

Max-autotune's cold first update took 426–459 seconds. Later arms in the same
process reused tuning results, so their roughly 26-second compile time is not a
cold-start claim. A version/GPU-specific persistent cache reduced the selected
mode's first update to 26.6 seconds on a fresh allocation. CUDA graph memory
accounting excludes some private allocations from
`max_memory_allocated`; use reserved/device memory rather than claiming the
misleading 2.6 GiB reported by that counter.

The initial cuDNN attention attempts failed to locate `libnvrtc`. Retrying with
the existing wheel's NVRTC library directory on `LD_LIBRARY_PATH` worked, and
measured 108.73 ms versus 109.97 ms for the selected Flash Attention path.
However, its first AdamW change differed by 13.37% relative L2, beyond the
predeclared 10% limit (loss error 0.000113, gradient error 1.68%). This candidate
is rejected; the library-path workaround and cuDNN override are not adopted.
Profiler traces confirmed the selected path used PyTorch Flash Attention and
the alternate path actually used cuDNN attention kernels.

## Production-path confirmation

Using the actual `make_loss`/`backward_batch` functions and full combined coverage
index, run order was baseline, optimized, optimized, baseline. Each arm reset the
model and optimizer and consumed identical batches. Measured update times were
117.76, 109.34, 109.67, and 121.37 ms respectively. All four timed arms supervised
exactly 5,633,746 targets. The baseline varied by about 3%; the optimized arms
stayed within 0.3% of each other. Treat this as roughly an 8–9% throughput gain,
not a precise hardware ceiling.

Both optimized coverage checks and two additional mixed-data batches with new
seeds passed the original numerical limits. After 225 updates, the four final
batch losses ranged from 1.18297 to 1.18311. This short trajectory comparison is
reassuring, but is not a full training-quality or Elo evaluation.

The first production save/resume comparison completed nine updates over real
data excerpts and preserved sample order and counts (129,354 human and 129,947
Leela targets). It **failed** the predeclared strict replay tolerances: model
`atol=1e-6, rtol=1e-5`, optimizer `atol=1e-7, rtol=1e-4`. Maximum weight difference
was 0.000021755; final validation CE differed by 0.00000152. The receipt's
overall `passed: false` is retained. Original-loop and uninterrupted-repeat
controls, plus checks at the exact resume boundary, distinguish ordinary
arithmetic variation from lost checkpoint state below.

That smoke test also exposed early BF16 rounding in validation loss. A separate
CUDA reproduction with zero logits returned 9.625 instead of `log(14750)` =
9.598998, in both eager and compiled-model paths. Although the returned loss
tensor was FP32, explicitly converting logits to FP32 before cross-entropy was
necessary to avoid the early rounding on this PyTorch stack. The shared loss
function now does so, and a uniform-prediction regression test passes. Final
CUDA validation at initialization now measures 9.5989988. This correction retains
BF16 model computation; only the loss calculation explicitly receives FP32 logits.

The final code measured **110.97 ms/update versus 118.87 ms** for the original
loop on the same allocation, a **7.1% throughput improvement** (6.65% less update
time). Use this final measurement for planning, rather than the earlier 8–9%
figures. The combined 278,963-update pass projects to about **8.60 update-hours
versus 9.21**, saving about **37 minutes**. One-time data indexing/compilation,
validation and checkpointing are additional. This is not an absolute H100 ceiling.

All three final-code numerical comparisons passed: largest loss error 0.000116,
gradient relative L2 0.93%, and first AdamW change relative L2 8.83%. These limits
were not loosened after seeing results. Final reserved GPU memory was 26.26 GiB
versus 31.50 GiB for the control, including the benchmark's allocator history.

## Resume and repeatability controls

Both implementations restored model weights, optimizer tensors, the next batch,
CPU RNG and CUDA RNG **exactly**, before the first resumed update. All runs
completed nine updates and covered the same 259,301 real-data targets exactly once.

| Comparison after nine updates | Weight relative L2 | Maximum weight difference | Optimizer-moment relative L2 | Validation CE difference |
|---|---:|---:|---:|---:|
| Optimized resumed vs uninterrupted | 1.35e-7 | 2.18e-5 | 1.80e-4 | 2.48e-7 |
| Original resumed vs uninterrupted | 1.66e-7 | 8.93e-6 | 1.84e-4 | 8.44e-6 |
| Two uninterrupted optimized runs | 1.31e-7 | 8.78e-6 | 1.77e-4 | 2.19e-6 |

All three fail the original strict whole-trajectory tolerances; the audit retains
`strict_repeatability_passed: false`. The comparable overall drift in the original
loop and in uninterrupted repeats, together with exact restored state, supports
GPU arithmetic variation rather than lost resume state. This is not a claim of
bitwise CUDA replay. The maximum individual-coordinate difference is larger in
the optimized resume comparison, although its overall relative error is similar.

38 local tests passed, covering sampler coverage, masks, weighted objectives,
accumulation, CPU exact resume, checkpoint loading, the legacy entrypoint, and
the uniform-loss precision regression. GPU checks used the actual production
training functions. Temporary smoke checkpoints/excerpts were removed after
comparison, without promoting any model.

## Adopted changes

The shared trainer compiles the model and scalar weighted loss together. The L16
presets select `max-autotune`; generic recipes still default to `default` for
lower startup cost. The launcher persists the compiler cache by PyTorch version
and GPU. Loss logits explicitly use FP32; model computation remains BF16. Global
batch, sampler, source weights, learning schedule and checkpoint schema are unchanged.
Packed Q/K/V and cuDNN attention remain experiments only.

## Receipts

Local small receipts: `output/cuda-tuning-20261029/`, including `budget.json`.
Remote traces, results, and source snapshots:
`/data/experiments/cuda-tuning-20261029/{profile,autotune,attention,cudnn,confirmation,validation,audit}/`.
One-time experiment scripts stay in Git-ignored `scratch/`; the remote source
snapshots preserve the exact code used by each stage.

Pricing used for conservative estimates: H100 $0.001097/second, CPU
$0.0000131/core/second, RAM $0.00000222/GiB/second, from
[Modal pricing](https://modal.com/pricing). Estimates charge the entire local
wall time at the full resource rate, including startup; these are not invoices.

Function receipts total **$4.18**; conservative local-wall-time compute totals
**$4.35**. Even adding the separate **$3 overhead allowance** gives $7.35, within
the $20 cap. The allowance is not claimed as actual spending.

| Stage | App ID | Conservative compute |
|---|---|---:|
| Profile | `ap-Vx7MJTZVWhk6wLqlCXumEF` | $0.243 |
| Compiler modes | `ap-ujyF75e732ACg4d1cZqdYN` | $0.903 |
| Attention alternatives | `ap-las2ZlCNyKt5l6w5VxBd5m` | $1.041 |
| cuDNN retry | `ap-adMBEG27tcNs3ZRee9IlG5` | $0.331 |
| Production confirmation | `ap-rgc5WXIxGUEf1jVsXlVtnx` | $0.725 |
| Loss-precision diagnosis | `ap-2RCBMpDM5E7KWDof8K9jTI` | $0.042 |
| Final audit and controls | `ap-RM2GJYSaaUSmjaxwE1nFIJ` | $1.065 |

`shutdown.json` records the final app-list verification: all seven stopped, zero
tasks. Unrelated deployed apps were left alone. `budget.json` and `summary.json`
retain the calculations. No commit or push.
