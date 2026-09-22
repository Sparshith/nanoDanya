# INT8 browser candidates: 5M and 15M training games

Both L12 models were exported with per-channel INT8 linear weights and dynamic
activation quantization. Embeddings remain FP32. The 15M-trained INT8 model is the
stronger candidate at nearly the same file size. Preserved FP32 strength is not
established: a fresh SF1000 control scored 76.75%, versus INT8's 69.25%.

| | 5M-game L12 | 15M-game L12 |
|---|---:|---:|
| Parameters | 105,512,448 | 107,590,656 |
| Vocabulary | 13,397 | 14,750 |
| INT8 ONNX file | 138,460,325 bytes | 143,662,610 bytes |
| SF500 score | 98.00% (192W/8D/0L) | 98.75% (195W/5D/0L) |
| SF1000 score | 60.00% (102W/36D/62L) | 69.25% (127W/23D/50L) |
| SF1500 score | 29.50% (42W/34D/124L) | 44.00% (71W/34D/95L) |
| Strict illegal top choices / 4,096 | 23 (0.562%) | 16 (0.391%) |
| Illegal excluding SAN under-disambiguation | 19 (0.464%) | 13 (0.317%) |
| Mean legal probability mass | 97.417% | 98.069% |
| Blunder rate, loss >=300 cp | 9.375% | 8.325% |
| Median cp loss | 23 | 21 |
| Stockfish best-move match | 42.041% | 43.701% |
| Puzzle first move | 48.727% | 53.455% |
| Puzzle full solve | 30.284% | 36.545% |

## Protocol and limits

Canonical `benchmark/run.py` on Modal, native ONNX Runtime 1.23.2 CPU with four
inference threads. Each Stockfish level has 200 games, colors balanced, temperature
0.8, legal masking, EOS suppressed, depth 1, 200-ply cap, seed 5000, four 50-game
shards. The time flag remains 0.02s but depth 1 takes precedence in these games.
The existing harness counts unfinished games as draws: 5M had 26 capped games and
one opponent vocabulary failure; 15M had 20 capped games and no vocabulary failures.
All 1,200 INT8 game records were legally replayed and outcomes audited.

Legality uses 4,096 held-out `datasets/games/5m` positions. Move quality uses the
same 4,096 positions from `datasets/eval/base` for both models, greedy legal moves,
and full-strength Fairy-Stockfish at 20ms per analysis. Both scored all positions
without failures. Puzzles are the same 8,800 IDs, 400 per 100-rating bin from
600 to 2800; no skips. Position-level and puzzle totals were verified.

The fresh 15M FP32 ONNX SF1000 control used the same current CPU harness and
settings: 147W/13D/40L, 76.75%, with three capped draws. Its 200 game records also
passed replay and score checks. The 7.5-point gap suggests possible quantization
loss but is not conclusive at this sample size. Historical FP32 scores were
97.5/62.5/29.25% for 5M and 97/78.25/42.5% for 15M; these are historical context,
not fresh matched controls.

These are native CPU strength tests, not browser GPU tests or public Elo ratings.
Earlier 5M native INT8 agreed with the browser on 42/45 reference choices, so
runtime differences matter. The 15M export rewrite and FP32 ONNX parity errors
were at most 0.0000175 on the export checks. Its browser runtime is not yet tested.

## Artifacts and reproduction

Source checkpoints: `/data/checkpoints/plain/games-{5m,15m}/l12_best.pt`.
Exports: `/data/browser/games-5m-20260921/` and
`/data/browser/games-15m-l12-20260921/` on `nanodanya-data`.
The 15M model is explicitly `plain/games-15m/l12`; the unsuffixed registry alias
selects the larger L16.

Local artifacts under `benchmark/results/`:

- `sf{500,1000,1500}_int8_{games5m,games15m_l12}_200_20260921.jsonl`
- `legality_int8_{games5m,games15m_l12}_4096_20260921.jsonl`
- `move_quality_int8_{games5m,games15m_l12}_4096_20260921.jsonl`
- `puzzles_int8_{games5m,games15m_l12}_400_20260921.{jsonl,png}`
- `sf1000_fp32_games15m_l12_200_20260921.jsonl`
- `int8_games15m_l12_{manifest,replay_audit}_20260921.json`

Use `MODAL_GPU=none MODAL_CPU=4 modal run modal_benchmark.py` with the usual mode
and the relevant `model_q8.onnx` path. See `benchmark/README.md` and
`browser/README.md` for benchmark and export commands. Nine focused backend,
legality-helper, and shard-persistence tests passed. No production model changed.
