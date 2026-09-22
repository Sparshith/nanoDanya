# Champion temperature sweep against SF1500

Completed 2026-09-21 on Modal workspace `workspace2` (profile `second`).

Lower temperatures performed substantially better in this sweep. Temperature 0
scored 74.875%, versus 50.375% at 0.8, a 24.5 percentage-point increase.
Temperature 0.2 had the highest observed score, 77.375%, but its 2.5-point lead
over 0 is inconclusive (approximate independent-sample 95% difference interval
-3.02 to +8.02 points). Temperature 0 is a reasonable deterministic default;
0.2 retains some sampling with similarly strong results. These findings apply
to this champion and SF1500 protocol, not automatically to the browser L12 or
the newer fine-tuned model.

Checkpoint: `/data/checkpoints/plain/games-15m/l16_best.pt`, the original L16 champion. This is not the L12 browser export or the deployed own-game fine-tune.

Protocol: 400 games per temperature, 200 as each color, starting from the initial position. Fairy-Stockfish UCI_Elo 1500, 20 ms per move, no depth/node limit. Legal masking, EOS disabled, 200-ply cap scored as a draw. Eight 50-game shards, batch size 64, seed 29210000 plus shard offset. Standard PyTorch benchmark on A100; no model weights or serving defaults changed.

| Temperature | Wins | Draws | Losses | Score | Approx. 95% interval | Capped games |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 279 | 41 | 80 | 74.875% | 70.92–78.83% | 6 |
| 0.2 | 293 | 33 | 74 | 77.375% | 73.52–81.23% | 7 |
| 0.4 | 258 | 30 | 112 | 68.250% | 63.88–72.62% | 7 |
| 0.6 | 228 | 50 | 122 | 63.250% | 58.85–67.65% | 12 |
| 0.8 | 177 | 49 | 174 | 50.375% | 45.78–54.97% | 18 |
| 1.0 | 119 | 39 | 242 | 34.625% | 30.22–39.03% | 10 |

Score is (wins + half the draws) / games. Intervals use the sample variance of per-game scores (1, 0.5, 0) and a normal approximation. They are descriptive, assume independent games, and are not adjusted for selecting among six temperatures. The model seeds match across settings; the time-limited engine adds uncontrolled variation, so these are not paired identical-opponent trajectories.

All 2,400 games replayed successfully: legal SAN moves, matching results and terminations, complete game numbering and color balance. Temperature 0 produced 400 distinct full games, with 392 distinct first-ten-ply sequences. Its deterministic model policy did not collapse this SF1500 test to duplicate games.

Artifacts:

- `benchmark/results/temperature_sweep_20260921_summary.json`: audited aggregate results, color splits, diversity, and uncertainty.
- `benchmark/results/sf1500_champion_temp{00,02,04,06,08,10}_20260921_400.jsonl`: complete game records and run configuration.
- Corresponding `_shards/` directories: resumable worker results.
- `benchmark/results/temperature_sweep_20260921_source.json` and `_source.zip`: SHA-256 manifest and fixed source snapshot. The initial live-workspace launch was interrupted after a concurrent edit broke packaging; reported runs used this snapshot.

Modal runs:

- Temperature 0.0: https://modal.com/apps/workspace2/main/ap-gN7LiM7g5KQs9P29O715zH
- Temperature 0.2: https://modal.com/apps/workspace2/main/ap-JGnshf9cDelF6bTIzElpib
- Temperature 0.4: https://modal.com/apps/workspace2/main/ap-GxncjQF3kfn56US7pi77VF
- Temperature 0.6: https://modal.com/apps/workspace2/main/ap-r1cBEVhMVT7qTnyzbtb0fH
- Temperature 0.8: https://modal.com/apps/workspace2/main/ap-OfLhIO4h53OO3d4ir40OK5
- Temperature 1.0: https://modal.com/apps/workspace2/main/ap-aArgzXx1TDnn6rDTeiqTax
