# More depth and longer teacher-free training under 10 MB

Status: training, evaluation, native checks and result audits complete on workspace2 H100. The user explicitly retained the $100 total cap,
including the previous H100 comparison (estimated measured compute $6.99).

## Recommendation

Use `output/small-depth-long-20260929/d9-200k.onnx` as the next baseline.
It is 9,454,890 bytes, reaches 1517 ±12 internal benchmark Elo, and solves
2,023/8,800 puzzles (22.99%). The previous 50k-step model scores 1352 ±9 and
16.25% in the same comparison. The new model scores 71.50% against it over
400 games (266 wins, 40 draws, 94 losses; paired-opening 95% interval
67.50%–75.50%). These are benchmark ratings, not human chess ratings.

Longer training helped more than extra depth in this experiment. Going from
100k to 200k steps raised the fitted ratings by 66, 49, and 30 points for
9, 15, and 28 layers respectively. Puzzle improvements were positive for all
three. The 9- and 15-layer models also won their direct longer-versus-shorter
comparisons; the 28-layer direct match remains inconclusive.

The 9-layer versus 15-layer result is close: 52.88% direct score with a
48.50%–57.38% interval, and a puzzle difference of only 25 solves. Prefer
9 layers because it leads the measured scores, is smaller, and required
less training compute. The 28-layer model trails the 9-layer model in the
direct match (9-layer score 56.12%, interval 51.62%–60.62%). These intervals
measure evaluation uncertainty, not variation across training seeds.

Single-thread native CPU median latency at 256 tokens was 80.0 ms for
9 layers, 83.2 ms for 15 layers, and 87.6 ms for 28 layers. The old 9-layer
model also took 80.0 ms. This measures local ONNX execution on synthetic
inputs, not browser performance.

All six experiment jobs completed and stopped, with shutdown evidence in
the saved app listings and launch logs. The local orchestrator exited 0.
New measured function compute is estimated at $39.23, or $46.21 including
the previous run. All original timeout reservations and overhead total
$96.85, below the retained $100 cap. This is not a final Modal invoice.

## Experiment

| Arm | Layers × width | Heads | Parameters | Preflight ONNX bytes |
| --- | ---: | ---: | ---: | ---: |
| d9 | 9 × 320 | 5 | 15,779,200 | 9,454,885 |
| d15 | 15 × 256 | 4 | 15,572,480 | 9,558,035 |
| d28 | 28 × 192 | 3 | 15,218,304 | 9,835,413 |

All three train from scratch with seed 42, recorded-move CE only
(`DISTILL_ALPHA=0`), and the existing `datasets/games/15m` training split.
There is no teacher in training. Context capacity is 512, training blocks are
256 tokens, and the vocabulary has 14,750 SAN tokens. Head dimension remains 64.
The final export uses symmetric INT4 matrices with float32 group-64 scales.
Random-weight preflight exports on the actual Modal environment passed the
10,000,000-byte limit and ONNX/reconstructed PyTorch parity through 512 tokens.

Each architecture trains for 200,000 optimizer steps, or 3,276,800,000 tokens.
The effective batch remains 64 sequences. Microbatch is now 64 with one
accumulation step instead of 32 with two steps. Learning rate uses one cosine
schedule from 3e-4 to 3e-5 over 200k steps, with a 200-step warmup. A 500-step
pilot resumes into the same schedule and optimizer state, without a restart.
Four fixed validation batches cover the same number of sequences as previously;
only the first half of the validation split is used for checkpoint selection.
Main validation and checkpoint intervals are 5,000 steps.

Best float checkpoints are retained after 50k, 100k, 150k, and 200k steps.
The best checkpoints available at 100k and 200k each receive an independent
2,000-step QAT pass, adding 32,768,000 tokens per exported candidate. QAT uses
a fresh optimizer and cosine 3e-5 to 3e-6. This produces six evaluated exports.
QAT occurs after float training completes, so it cannot affect the continuing
float trajectory. Selection uses validation CE, not puzzle or tournament scores.

Architecture comparisons match training tokens and recipe. The 100k snapshots
come from the same 200k learning-rate schedule, not separately tuned shorter
runs. Comparisons to the old 50k model also change schedule, warmup/pilot, and
microbatch grouping. One seed per architecture does not measure seed variance.

## Evaluation

Each of the six exports plays the previous 9 × 320, 50k-step `deep` model and
sf1000/sf1500/sf2000. There are also three within-architecture 100k versus 200k
matches and all three pairs among the 200k candidates. Thirty pairs each use
200 openings with both colors, for 12,000 new games. Four subprocesses share
one H100 with batch 16, using the prior benchmark's exact engine binary,
depth-1 opponents, greedy legal-masked moves, 10 ms adjudication, and 400-ply cap.

All six candidates receive the same 8,800 puzzles and 1,024 heldout positions.
The evaluator uses reconstructed final ONNX weights for GPU batching; final
exports also undergo native ONNX numerical checks. Puzzle identities and heldout
hashes must match the previous run. All new PGNs are replay-audited. The shared
BayesElo fit includes 9,600 previous games and 12,000 new games, anchoring sf1500
at 1500. Ratings are internal benchmark estimates, not human chess ratings.

## Budget and artifacts

The new launch preserves the previous $31.972716 in full timeout reservations
and overhead, even though measured compute was only $6.99. New allocations total
$68: d9 $16, d15 $18, d28 $22, evaluation $10, CPU export/preflight $2. Each new
allocation includes $1 overhead reserve. The combined allocation ceiling is
$99.972716. No automatic retries; all GPU work uses H100.

Training timeouts are 12,000 / 13,500 / 17,000 seconds; evaluation is limited to
7,200 seconds. The capped launcher enforces both Modal function and local wall
timeouts. Rates were rechecked against [Modal pricing](https://modal.com/pricing)
on September 29, 2026. Resource-time estimates exclude final invoice overhead.

Orchestrator: `scratch/run_small_depth_long.py`. Remote stages:
`scratch/small_depth_long.py`. Audit/report: `scratch/report_small_depth_long.py`.
Local protocol, ledgers, logs, exports and results:
`output/small-depth-long-20260929/`. Remote artifacts:
`/data/checkpoints/plain/small-depth-long-20260929/` on workspace2 volume
`nanodanya-data`, using Modal profile `second`. Full checkpoints remain remote.
Seventeen source files were archived locally under `source/`; their hashes match
all three training snapshots. The final audit checks that the workspace versions
still match, so source drift cannot silently pass the report.

## Results

All rows use the same 21,600-game internal benchmark rating fit.

| Model | ONNX MB | Benchmark Elo | Puzzles solved / 8,800 |
| --- | ---: | ---: | ---: |
| deep | 9.455 | 1352 +9/-9 | 1430 (16.25%) |
| d9-100k | 9.455 | 1451 +14/-14 | 1829 (20.78%) |
| d9-200k | 9.455 | 1517 +12/-12 | 2023 (22.99%) |
| d15-100k | 9.558 | 1456 +14/-14 | 1768 (20.09%) |
| d15-200k | 9.558 | 1505 +12/-12 | 1998 (22.70%) |
| d28-100k | 9.835 | 1445 +14/-14 | 1665 (18.92%) |
| d28-200k | 9.835 | 1475 +12/-12 | 1936 (22.00%) |

`deep` is the previous 9-layer, 50k-step model. All six new candidates are teacher-free.

| Direct match | Score | Paired-opening 95% score interval |
| --- | ---: | ---: |
| d15-100k vs deep | 62.12% | 57.50%–66.75% |
| d15-100k vs d15-200k | 43.25% | 38.62%–47.88% |
| d15-200k vs deep | 70.75% | 66.62%–74.75% |
| d15-200k vs d28-200k | 54.50% | 49.62%–59.38% |
| d28-100k vs deep | 61.50% | 56.62%–66.25% |
| d28-100k vs d28-200k | 47.12% | 42.25%–52.12% |
| d28-200k vs deep | 64.38% | 59.75%–68.88% |
| d9-100k vs deep | 62.62% | 57.88%–67.25% |
| d9-100k vs d9-200k | 43.00% | 38.50%–47.38% |
| d9-200k vs deep | 71.50% | 67.50%–75.50% |
| d9-200k vs d15-200k | 52.88% | 48.50%–57.38% |
| d9-200k vs d28-200k | 56.12% | 51.62%–60.62% |

Score counts a draw as half a win. Intervals describe benchmark sampling, not training-seed variance.

All 12,000 new PGNs replayed legally. All six native ONNX checks, export hashes,
puzzle identities and heldout position hashes passed. Native CPU latency is recorded
in `native-latency.json`; browser execution was not tested.

Measured function compute estimate: $39.23 for this run,
$46.21 including the previous H100 comparison.
Combined timeout reservations and overhead: $96.85 of the $100 cap.
Estimates are based on resource receipts, not a final Modal invoice.
