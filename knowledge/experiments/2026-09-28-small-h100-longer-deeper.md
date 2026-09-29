# Longer training and a deeper small model on H100

Status: complete. Prefer `deep` as the next small-model candidate: it is smaller,
solves more puzzles, and has the highest point estimate in the shared rating fit.
Its direct match against `long` is inconclusive. Both new recipes improve over
the original normally trained model and the earlier distilled model.

## Results

All ratings come from one 9,600-game BayesElo fit, anchored to sf1500 at 1500.
They are ratings within this benchmark, not external human chess ratings.

| Model | ONNX MB (decimal) | Benchmark Elo | Full puzzles solved / 8,800 |
| --- | ---: | ---: | ---: |
| Original normal, 6 × 384, 25k steps | 9.63 | 1244 ±13 | 1,100 (12.50%) |
| Earlier distilled, 6 × 384 | 9.63 | 1277 ±13 | 1,136 (12.91%) |
| Longer normal, 6 × 384, 50k steps | 9.63 | 1341 ±12 | 1,357 (15.42%) |
| Deeper normal, 9 × 320, 50k steps | 9.45 | 1359 ±12 | 1,430 (16.25%) |

Each direct match is 400 games over 200 openings with both colors. Score counts
a draw as half a win. Confidence intervals resample paired openings.

| Match, from first model's perspective | W / D / L | Score | 95% score interval |
| --- | ---: | ---: | ---: |
| Long vs original normal | 230 / 33 / 137 | 61.63% | 57.00–66.00% |
| Long vs distilled | 212 / 31 / 157 | 56.88% | 52.13–61.63% |
| Deep vs original normal | 230 / 41 / 129 | 62.63% | 58.63–66.38% |
| Deep vs distilled | 228 / 40 / 132 | 62.00% | 57.13–66.50% |
| Deep vs long | 192 / 33 / 175 | 52.13% | 47.38–56.88% |

Deep solves 73 more puzzles than long, a 0.83 percentage-point gain (paired
normal 95% interval: +0.10 to +1.55 points). The intervals reflect benchmark
sample uncertainty, not variation between training seeds.

On the identical 1,024 heldout positions, quantized cross-entropy improves from
1.799 (original normal) and 1.777 (distilled) to 1.668 (long) and 1.655 (deep).
Raw top-choice illegal moves are 29, 12, 12, and 14 respectively. Tournament play
uses legal masking; all 4,400 new PGNs replayed legally, with no unscored games,
illegal-move failures, or context-limit endings. Puzzle identities and heldout
position hashes match the earlier experiment. Native ONNX checks passed;
browser execution was not verified in this experiment.

The practical next candidate is `output/small-h100-20260928/deep.onnx`. These
results establish that the new teacher-free recipes beat the earlier distilled
recipe, not that distillation would lose at an equal longer training budget.

The user authorized both longer normal training and a deeper, narrower model on
workspace2 H100, with a $100 total limit. Both candidates train from scratch with
recorded-move CE only, without loading a teacher. Existing model exports and
checkpoints are preserved.

| Candidate | Layers | Width | Heads | Parameters | Final ONNX bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| `long` | 6 | 384 | 6 | 16,280,832 | 9,629,293 |
| `deep` | 9 | 320 | 5 | 15,779,200 | 9,454,887 |

Both use head dimension 64, tied embeddings, 14,750 SAN tokens, and context 512.
Preflight verified the export size before training. Final trained exports passed
native ONNX checks and numerical parity against reconstructed PyTorch weights at
sequence lengths 1, 8, 32, 64, 128, and 512. Maximum absolute errors were 8.11e-6
(`long`) and 1.06e-5 (`deep`). Export hashes are recorded in `exports.json`.

## Fixed recipe

- Dataset: existing `datasets/games/15m` on workspace2, Modal profile `second`.
- Seed 42; independent fresh initialization for each architecture.
- Pilot: 100 steps, batch 32 × 256, LR 3e-4.
- Main: resume through step 50,000, microbatch 32 × 256 with accumulation 2,
  cosine LR 3e-4 to 3e-5, warmup through step 200. First-half validation uses
  eight fixed batches; evaluation/checkpoint intervals are 2,000 updates.
- Pilot plus main: 818,380,800 tokens per candidate.
- QAT: fresh optimizer from float best, 2,000 steps at the same effective batch,
  cosine LR 3e-5 to 3e-6. Another 32,768,000 tokens per candidate.
- Both modes use CE only (`DISTILL_ALPHA=0`). Every matrix is exported as
  symmetric INT4 with float32 scales in groups of 64.

This isolates architecture between the two new arms at matched training tokens.
The longer-training comparison against the previous 25,000-step run also changes
the learning-rate schedule and GPU/software platform (B200/PyTorch 2.7.1 to
H100/PyTorch 2.5.1). It is one seed per recipe, not a multi-seed estimate.

## Evaluation protocol

Each new candidate plays the other, both existing small models, and sf1000,
sf1500, sf2000. Every pair uses the same 200 openings with both colors: 400 games
per pair, 4,400 new games total. Four subprocesses share one H100 with batch 16.
Games use greedy legal-masked play, canonical Fairy-Stockfish 14 at depth 1,
10 ms adjudication at 1300 cp or mate, draw claims, and a 400-ply cap.

The exact final ONNX weights are reconstructed for GPU batching, with CPU ONNX
numerical parity checked separately. The existing small-model export hashes are
verified before evaluation. Each new candidate also gets the same 8,800 puzzles
and 1,024 heldout positions used in the prior comparison. No benchmark result
guides further training in this experiment.

The BayesElo fit includes the prior 5,200 games and all new games, anchoring
sf1500 at 1500. Every new PGN was replay-audited, and puzzle identities and heldout
position hashes were checked against the earlier experiment.

## Budget and reproduction

Separate enforced allocations sum to the shared $100 cap: $30 per training arm,
$30 for evaluation, and $10 for CPU preflight/export. Each includes a $3 overhead
reserve. No automatic retries. Each GPU stage is capped at 5,400 seconds; CPU
stages at 600 seconds. Planned full-time reservations plus overhead total $31.97.
Rates were checked against [Modal pricing](https://modal.com/pricing) on September 28.

Measured function compute estimates total **$6.99**: long training $2.88, deep
training $2.65, evaluation $1.45, CPU preflight/export under $0.01. This is a
resource-time estimate, not the final Modal invoice; startup, storage, and other
overhead are covered by the budget reserves. Training stages took 40.6 and 37.2
minutes respectively, and evaluation took 20.4 minutes. All five experiment
apps were confirmed stopped with zero tasks. No extra training was launched.

`scratch/run_small_h100.py` runs preflight, concurrent training on separate H100s,
export, and evaluation through `training/run_distillation.py`. It refuses an
existing local output directory. `scratch/small_h100.py` implements remote stages;
`scratch/report_small_h100.py` audits and summarizes the downloaded records.

Local protocol, ledgers, logs and final small exports: `output/small-h100-20260928/`.
Remote data: `/data/checkpoints/plain/small-h100-20260928/`, with `long/` and `deep/`
checkpoint directories, each containing `float/` and `qat/`. Full checkpoints stay
on Modal. The prior experiment remains under `output/small-ce-20260928/` locally
and `/data/checkpoints/plain/student-ce-20260928/` on the volume.
