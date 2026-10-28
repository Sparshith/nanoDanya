# Browser student distillation

The complete experiment runs from the already copied Modal dataset and teacher
checkpoint with one command:

```bash
uv run python training/distill_pipeline.py --run-name student-l6e384-repro --profile second
```

B200 is the default after a measured training pilot reached about 370,000
tokens/second, versus about 209,000 on H100 at the same effective batch of 64.
Use `--gpu H100` to run on H100. The B200 main and QAT stages use batch 64 with
one accumulation step; H100 uses batch 32 with two. The pilot keeps batch 32 on
both. `modal_train.py` selects PyTorch 2.7.1+cu128 for B200 and 2.5.1+cu124 for
H100 automatically. Export runs on CPU and paired heldout evaluation on L4.

It uses the shared `output/distill-20260924/budget.json` ledger and stops if a
stage would exceed the $100 reservation. It runs a 100-step pilot, resumes the
float student with a cosine learning-rate schedule, starts QAT from the best
float checkpoint in a separate directory, exports the q4 browser ZIP, and
compares teacher, float, fake-quantized, and exported weights on 1,024 fixed
positions from complete games in the untouched second half of `val.bin`. The
`student_float` row uses the QAT checkpoint's shadow weights with quantization
disabled. It does not measure the untouched pre-QAT student. Separate Stockfish
1000 runs for teacher, pre-QAT float student, and q4 student show how much
strength was lost during distillation and after quantization. The pipeline also
runs 100 q4 versus teacher games. Each game check uses 100 games, temperature
0.2, and the same seed.

Only the final ZIP and five small JSONL reports are downloaded to
`output/student-l6e384-repro/`. Full checkpoints and datasets stay on the Modal
volume. The ZIP must be under 10,000,000 bytes. A 100-game result is an early
strength estimate; use more games before claiming the student matches the
teacher.

This command starts from a fresh run name. If that name already has checkpoints,
budgeted stages, or a local output directory, the pipeline refuses to proceed.
For an intentional new run, pass a new `--run-name`; it keeps the same budget
ledger. If a stage fails, inspect its log under `output/distill-20260924/`
before deciding whether to start a new run.

The first exported student also measured **1260 BayesElo (+17/-18)** in a separate
1,200-game canonical tournament, anchored at sf1500=1500. The existing INT8
baseline was 1768 in the same fit. This uses greedy play, fixed openings and
adjudication, so it differs from the pipeline's temperature-0.2 game checks.
The budgeted tournament entry point is `training/distill_tournament.py`; rating
and protocol details are in `knowledge/experiments/2026-09-24-browser-student-distillation.md`.
