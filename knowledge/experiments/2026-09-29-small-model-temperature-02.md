# Temperature 0.2 on the 9-layer, 200k-step model

Complete on workspace2 H100. Keep temperature 0 as the default: temperature 0.2
performed worse in this single sampling run.

| Temperature | Internal benchmark Elo | Score against matched Stockfish opponents | Puzzles solved |
| --- | ---: | ---: | ---: |
| 0 | 1517 ±12 | 54.83% | 2023/8800 (22.99%) |
| 0.2 | 1475 ±19 | 51.17% | 1991/8800 (22.63%) |

The same `d9-200k.onnx` artifact was used, without retraining or re-exporting.
SHA-256: `9d179f227cac980c06aa26a5f9824e975ee1c1ef4d21044e19e9b83955107ebe`.
Size remains 9,454,890 bytes. The new evaluation played 400 games each against
sf1000, sf1500, and sf2000, using the previous 200 openings with both colors.
The model samples at temperature 0.2 with seed 0; Stockfish behavior is unchanged.
Engine binary, depth 1, legal masking, 10 ms/1300 cp adjudication, 400-ply cap,
four workers, and batch size 16 match the previous evaluation.

The matched game score fell 3.67 percentage points. An opening-cluster paired
bootstrap gives a 95% interval of -7.13 to -0.04 points. Puzzle accuracy fell
0.36 points, with a paired 95% interval of -0.80 to +0.07 points. The puzzle
difference is inconclusive. This is one sampling seed; these intervals do not
measure seed variance. Ratings use a shared 22,800-game BayesElo fit anchored
to sf1500=1500, and are not human chess ratings.

| Opponent | Temp 0 W/D/L | Temp 0.2 W/D/L |
| --- | ---: | ---: |
| sf1000 | 311/8/81 | 297/12/91 |
| sf1500 | 191/33/176 | 174/22/204 |
| sf2000 | 121/29/250 | 114/24/262 |

All 1,200 new games replayed legally, with matching openings and colors.
The 8,800 puzzle identities, model hash, engine hash, and prior source hashes
matched. The existing puzzle evaluator hardcodes greedy selection, so the
isolated evaluation wrapper overrides its sampler to 0.2 and records that
temperature in the run configuration. It sampled 12,785 puzzle moves.
The tournament uses the existing `--temperature 0.2` option directly.

Measured function compute estimate: $0.4163 for this run, $46.6270 including
the earlier experiments. The 2,100-second timeout, startup allowance and
$0.25 overhead add a $2.9516 reservation. All retained reservations and
overhead now total $99.8052, below the $100 cap. These are resource-time
estimates, not a final Modal invoice. No automatic retries were used.

Modal app `ap-n7T5MjS4xR1ONZExIp3YnF` completed and was verified stopped with
zero tasks. The local orchestrator exited 0. Artifacts, protocol, source copy,
budget, report and shutdown evidence are under `output/small-temp02-20260929/`.
Remote results are under `/data/checkpoints/plain/small-temp02-20260929/` on
workspace2 volume `nanodanya-data`. Scripts: `scratch/small_temperature.py`,
`scratch/run_small_temperature.py`, and `scratch/report_small_temperature.py`.
