# Under-10 MB browser student

A working 9,424,814-byte browser decoder was produced, but this first distillation
recipe loses substantial playing strength. Most of that loss is already present
in the float student. This is a runnable small baseline, not a replacement champion.

## Artifact and inputs

Final ZIP: `output/distill-20260924/student-int4.zip` and
`/data/checkpoints/distill/student-l6e384/student-int4.zip` on the workspace2 Modal
volume `nanodanya-data` (`MODAL_PROFILE=second`). SHA-256:
`22371450dc37ca7887b1ed6f62a8cd52867ed877bdef3b47ef7441b31277d89f`.

Teacher: `/data/checkpoints/plain/games-15m/l12_best.pt`, the 107.6M-parameter
L12 underlying the existing browser export. Distillation and teacher benchmarks
use the original float checkpoint, not its INT8 browser approximation. The
unqualified registry alias `plain/games-15m` selects a different, larger L16.

Teacher and `datasets/games/15m` were copied directly from the original Modal
workspace into workspace2. Input hashes are in `output/distill-20260924/assets.json`
and `/data/experiments/distill-20260924/asset-copy.json`. No large checkpoint or
dataset was downloaded locally; only the final ZIP and small reports were.

Student: 6 layers, width 384, 6 attention heads, context 512, 14,750 SAN tokens,
tied input/output embedding, 16,280,832 parameters. RMSNorm, Q/K normalization,
RoPE, ReLU-squared MLP, logit softcap 15. Every matrix uses signed symmetric INT4
in groups of 64 with float32 scales. ZIP includes weights, vocabulary, RoPE table,
and self-contained WebGPU inference code. Inference only predicts next-token
scores; the existing game harness applies the legality mask.

## Training

Equal-weight recorded next-token CE and teacher KL at temperature 2, including
T-squared KL scaling. AdamW, weight decay 0.1, gradient clipping 1.0, BF16 autocast.

- Pilot: 100 updates, batch 32 x 256, LR 3e-4. Validation CE 5.3380.
- Main: effective batch 64 x 256, total step 25,000, cosine LR 3e-4 to 3e-5,
  warmup through step 200. Retained lineage: 408,780,800 tokens including pilot.
  Best step 25,000: combined validation loss 1.5880, CE 2.0408.
- QAT: initialize fresh optimizer from float best; 2,000 updates, batch 64 x 256,
  cosine LR 3e-5 to 3e-6, no warmup. Another 32,768,000 tokens. Best step 2,000:
  combined validation loss 1.6254, CE 2.0694. Initial QAT CE was 2.1709.

Main started on H100 at batch 32 with accumulation 2, around 209k tokens/s. At the
user's request for speed, it resumed from step 13,000 on B200. A 60-second B200
pilot resumed through 14,117; batch 64 with accumulation 1 then finished at about
370k steady-state tokens/s (362.5k including validation/startup inside the loop).
The H100 had reached 13,540 before stopping, so 540 updates were discarded and
repeated. B200 uses PyTorch 2.7.1+cu128; H100 used 2.5.1+cu124.

Float checkpoints: `/data/checkpoints/distill/student-l6e384/`.
QAT checkpoints: `/data/checkpoints/distill/student-l6e384-qat/`.
Each contains `student_best.pt` and resumable `student_latest.pt`.

## Results

Canonical `games` scaffold, temperature 0.2, seed 5000, alternating model colors,
100 games per comparison, legal-move masking, no model search. Fairy-Stockfish 14
binary matches the canonical SHA-256. Maximum 200 plies; capped games count as
draws. These scores are not the newer tournament Elo protocol or a public rating.

| Model / opponent | Wins | Draws | Losses | Score | Capped draws |
| --- | ---: | ---: | ---: | ---: | ---: |
| Float L12 teacher / SF1000 | 84 | 10 | 6 | 89.0% | 1 |
| Pre-QAT float student / SF1000 | 13 | 21 | 66 | 23.5% | 5 |
| Exported INT4 student / SF1000 | 15 | 27 | 58 | 28.5% | 13 |
| Exported INT4 student / teacher | 4 | 10 | 86 | 9.0% | 0 |

All 400 recorded games were replayed locally without illegal moves or result
mismatches. The five-point float/q4 difference is not evidence of improvement:
100-game samples are noisy, and q4 has eight more capped draws. The major loss
relative to the teacher is already visible before quantization.

Final heldout report uses 1,024 fixed positions from complete games after the
midpoint of `val.bin`, at plies 7/8, 19/20, 39/40, 79/80 (both colors). Checkpoint
selection used only fixed first-half validation batches; no further training
followed the heldout/game results. An earlier black-only diagnostic remains in
`heldout.jsonl`; final claims use `heldout-balanced.jsonl`.

| Metric | Teacher | Exported INT4 |
| --- | ---: | ---: |
| Human next-move CE | 1.3626 | 1.7766 |
| Strict raw top-1 illegal | 0.098% | 1.172% |
| Mean legal probability mass | 98.90% | 95.65% |
| Teacher legal top-1 agreement | n/a | 66.41% |

The paired evaluator's `student_float` row means the QAT checkpoint's shadow
weights with fake quantization disabled, not the pre-QAT checkpoint. The separate
float game run above measures the actual pre-QAT student.

## Browser, budget, reproduction

Actual ZIP passed the headless Chrome challenge checker at 1/8/32/64 tokens and
played as both colors (`e4`, then `c5` against human `e4`). All 14,750 logits were
finite at 128 and 512 tokens; maximum difference from Python's dequantized model
was 1.24e-5. Reports and screenshots are under `output/distill-20260924/`.

Measured Modal function compute: $2.8586, excluding startup, image builds and
storage. Conservative reservations total $38.41 including $10 for overhead,
within the authorized $100. Full reservations were retained after early stops
and failures. This is not a billing invoice. All experiment Modal apps are
stopped and automation browser sessions closed; local preview remains available.

Setup fixes: CPU asset copy initially retried a conditional-secret dependency
mismatch; B200 image build first hit a concurrent file edit; the first native
evaluator lacked `requests`. The launcher now snapshots source before upload,
includes pinned requests, caps resources and wall time, and logs reservations
and receipts in `output/distill-20260924/budget.json`.

One-command fresh reproduction is documented in `training/DISTILLATION.md`:
`uv run python training/distill_pipeline.py --run-name student-l6e384-repro --profile second`.
It uses B200 by default and the same shared budget ledger. It downloads only the
ZIP and five JSONL reports. Native evaluation uses dequantized weights from the
actual ZIP; browser inference is verified separately. Full machine-readable
results: `output/distill-20260924/run-report.json`.

## Bayesian Elo follow-up

The exported INT4 student rates **1260 +17/-18** (95% interval 1242–1277) after
1,200 games on the canonical opening-based tournament protocol. This is the
internal benchmark scale with sf1500 fixed at 1500, not a human rating.

The same 200 fixed openings were played with both colors against sf1000,
sf1500 and sf2000: 400 games per opponent. Greedy legal-masked model play,
Fairy-Stockfish 14 at depth 1, 10 ms full-strength adjudication at 1300 cp or mate,
draw claims enabled, maximum 400 plies. This protocol differs from the earlier
100-game tests above, so their scores are not interchangeable.

| Opponent | Wins | Draws | Losses | Student score |
| --- | ---: | ---: | ---: | ---: |
| sf1000 | 204 | 15 | 181 | 52.875% |
| sf1500 | 79 | 26 | 295 | 23.0% |
| sf2000 | 33 | 9 | 358 | 9.375% |

All 1,200 games replay-audited cleanly. No unscored, capped or context-limit games:
1,150 adjudications, 49 threefold repetitions, one insufficient-material draw.
Four subprocesses on one budgeted B200 split the openings into disjoint groups
of 50 (`training/distill_tournament.py`). The artifact was loaded directly from
its ZIP as dequantized float32 weights, with browser numerical parity established
above. Openings SHA-256:
`c08db594e480821a1d5335c5344dfba12b075fc9e1f3d3444a4e4ef9724e5db8`.

BayesElo `mm`, `exactdist`, then `offset 1500 sf1500` fits these games together
with the existing 2,400 pool/baseline games. The original fit was reproduced
exactly before adding the student. The joint 3,600-game fit is:

| Player | Elo | 95% error | Games |
| --- | ---: | ---: | ---: |
| Existing L12 INT8 CPU baseline | 1768 | +19/-19 | 1200 |
| sf2000 | 1667 | +16/-16 | 1600 |
| sf1500 anchor | 1500 | fixed reference | 1600 |
| Exported INT4 student | 1260 | +17/-18 | 1200 |
| sf1000 | 1240 | +17/-17 | 1600 |

The baseline shifts from 1766 to 1768 because the pool is refitted jointly.
BayesElo source tarball matches canonical SHA-256
`95f3d32381932ba9ccd1ab221ff33273cd3e48abedcfeba4c4f3f5a9b303a2a4`.

Reports: `output/distill-20260924/student-bayeselo.jsonl`,
`tournament-summary.json`, and `bayeselo-games/student-int4-1200.jsonl`.
Volume games: `/data/checkpoints/distill/student-l6e384/tournament-1200.jsonl`.
This follow-up added $0.5412 in measured function compute; experiment total is
now $3.3998 plus build/startup/storage overhead. Cumulative reservations are
$40.92 including the original $10 overhead allocation, within the $100 cap.
