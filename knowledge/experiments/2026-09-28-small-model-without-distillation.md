# Small model trained without distillation

The teacher-free model fits in **9.63 MB** and is competitive with the distilled
student. This matched-token comparison favors distillation by **38 internal Elo**:
1243 versus 1281. Direct play and puzzle differences are small. Teacher-free
training runs roughly twice as fast, making equal-GPU-time training a useful next
experiment.

## Matched recipe

Both models have 16,280,832 parameters: six layers, width 384, six attention heads,
tied embeddings, 14,750 SAN tokens, and context 512. Both train on
`datasets/games/15m`, then use symmetric INT4 weights in groups of 64.

The new model starts from random weights and uses recorded next-token CE only.
`training/train_distill.py` now supports `DISTILL_ALPHA=0`, which skips teacher
loading and inference entirely, including during quantization-aware training.
Checkpoints record the objective, and resume rejects an objective mismatch.

- Pilot: 100 updates, batch 32 × 256, LR 3e-4.
- Main: resume through update 25,000, batch 64 × 256, cosine LR 3e-4 to 3e-5,
  warmup through update 200. Pilot plus main: 408,780,800 tokens.
- INT4-aware training: fresh optimizer from float best, 2,000 updates,
  batch 64 × 256, cosine LR 3e-5 to 3e-6. Another 32,768,000 tokens.
- Best float CE: 2.135741 at update 25,000. Best INT4-aware CE: 2.147489 at
  update 2,000. The earlier distilled model reported 2.0408 and 2.0694 respectively.

The new run used B200 throughout. The earlier distillation run used H100 and then
B200. Both use seed 42, but initial student weights are not bitwise matched:
constructing the teacher consumed random state in the earlier trainer. This is
one training seed per recipe, with matched architecture and token budget, not
matched wall-clock compute. Checkpoint selection uses first-half validation
batches; the heldout and game results did not guide further training.

## Results

Both checkpoints were freshly exported to the current ONNX format and evaluated
from those exported weights. The earlier distilled ZIP is not the test input.

| Metric | Recorded-move CE only | Distillation |
| --- | ---: | ---: |
| ONNX bytes, including vocabulary | 9,629,291 | 9,629,298 |
| Internal BayesElo | 1243 (+15/-16) | 1281 (+15/-15) |
| Complete puzzles solved | 1100/8800 (12.50%) | 1136/8800 (12.91%) |
| Correct puzzle first move | 2597/8800 (29.51%) | 2670/8800 (30.34%) |
| Heldout next-move CE | 1.79865 | 1.77660 |
| Raw top-1 illegal rate | 2.832% | 1.172% |
| Mean probability on legal moves | 90.54% | 95.65% |

Direct match: distilled won 198, lost 172, and drew 30, scoring 53.25% in 400
games. Bootstrap resampling the 200 opening pairs gives a 95% score interval of
48.875–57.625%, so direct play alone does not establish a clear winner.

The distilled puzzle advantage is 36 solves. The models uniquely solved 505 and
469 puzzles respectively; a paired normal 95% interval for the distilled-minus-CE
solve-rate difference is -0.286 to +1.104 percentage points. The puzzle gap is
also inconclusive by itself.

| Matchup | Wins | Draws | Losses | Score |
| --- | ---: | ---: | ---: | ---: |
| CE / sf1000 | 185 | 15 | 200 | 48.125% |
| CE / sf1500 | 77 | 11 | 312 | 20.625% |
| CE / sf2000 | 39 | 13 | 348 | 11.375% |
| Distilled / sf1000 | 218 | 7 | 175 | 55.375% |
| Distilled / sf1500 | 92 | 18 | 290 | 25.250% |
| Distilled / sf2000 | 43 | 14 | 343 | 12.500% |

Tournament protocol: the same 200 fixed openings, both colors, greedy legal-masked
play, no model search, canonical Fairy-Stockfish 14 at depth 1, 10 ms adjudication
at 1300 cp or mate, draw claims enabled, maximum 400 plies. Each model played
1,200 Stockfish games plus the shared 400-game direct match, for 2,800 new games.
All replayed legally with consistent results: 2,692 adjudications and 108 threefold
repetitions, no capped, unscored, or context-limit games.

BayesElo fits those games with the existing 2,400 pool/baseline games, anchoring
sf1500 at 1500. These are internal benchmark ratings, not human ratings. The
fresh distilled rating of 1281 differs from the earlier 1260 because this is a
new tournament and joint fit. Full puzzle identities and heldout position hashes
match the earlier experiment. No puzzles were skipped.

## Validation, artifacts, and cost

CPU ONNX outputs match reconstructed PyTorch outputs within 1.16e-5 at
1/8/32/64/128/512 tokens. Benchmarks batch the exact exported weights in PyTorch
on GPU. The new model also passed `challenge/starter/check.py --games 0` on real
game histories. Headless Chrome exited with SIGABRT at startup locally, so a
browser compatibility check was not completed. No automation browser remains open.
Four focused tests passed, including teacher-free training, resume, and INT4-aware
training with a nonexistent teacher checkpoint.

Local artifacts and reports: `output/small-ce-20260928/`, including `ce.onnx`,
`distilled.onnx`, `report.json`, `ratings.jsonl`, `protocol.json`, `training.json`,
`exports.json`, and full game/puzzle records. The CE ONNX SHA-256 is
`be32a31d7d7bc0c26b32e3581f6300187faed9d29ae9355c712ccdbda0404ccc`.

Full checkpoints remain on the workspace2 Modal volume, profile `second`, under
`/data/checkpoints/plain/student-ce-20260928/{float,qat}/`. Each has best and
resumable latest checkpoints. No large checkpoint was downloaded locally.

Measured function compute is approximately **$2.23**, including training ($1.06),
both exports, and evaluation of both models. Startup, builds, and storage are
excluded. Conservative reservations plus overhead total $10.29 against a $15 cap.
Rates were checked against [Modal pricing](https://modal.com/pricing) on September
28. All three apps created for this experiment stopped after completion.

Local reproduction scripts are `scratch/compare_small_training.py` (remote stages
`train`, `export`, `evaluate`, selected by `SMALL_STAGE`),
`scratch/finish_small_training.py` (budgeted export/evaluation and downloads), and
`scratch/report_small_training.py` (game replay and paired comparisons). Training
runs through `training/run_distillation.py` with a separate budget ledger. Use a
fresh `SMALL_RUN_DIR` and stage names for an intentional new run; the training
stage refuses an existing checkpoint directory.
