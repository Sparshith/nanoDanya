# Bayesian Elo estimates for the temperature sweep

Computed 2026-09-22 from the six completed 400-game runs against the same SF1500 opponent. No additional games were played.

These are Bayesian **score-equivalent logistic Elo** estimates. Fixing the benchmark opponent at 1500 is a reference convention, not evidence that the opponent has a calibrated Lichess rating. This analysis estimates performance against that opponent under the recorded protocol.

| Temperature | Elo difference | Anchored Elo | 95% credible interval |
| --- | ---: | ---: | ---: |
| 0.0 | +188 | 1688 | 1653–1725 |
| 0.2 | +212 | 1712 | 1675–1750 |
| 0.4 | +132 | 1632 | 1598–1667 |
| 0.6 | +94 | 1594 | 1561–1627 |
| 0.8 | +3 | 1503 | 1471–1534 |
| 1.0 | -110 | 1390 | 1356–1424 |

Temperature 0.2 minus 0: posterior median 23.7 Elo, 95% interval -28.5 to 75.9. Probability the difference is positive: 81.37%. This favors 0.2 but does not establish a clear winner.

## Method

For each temperature and each color independently, use a multinomial W/D/L likelihood and a Dirichlet(0.5, 0.5, 0.5) prior. The posterior is Dirichlet(W + 0.5, D + 0.5, L + 0.5). Average the White and Black expected scores equally, then transform each posterior draw with `delta = 400 * log10(score / (1 - score))`. Add the fixed 1500 anchor. Report the posterior median and 2.5/97.5 percentiles. This estimates the Elo equivalent of expected score, rather than imposing a shared draw propensity across all temperatures.

This uses the [Dirichlet-multinomial posterior](https://users.stat.ufl.edu/~aa/cda2/bayes.pdf) and the standard [logistic score-to-Elo conversion used in Fishtest](https://github.com/official-stockfish/fishtest/blob/master/server/fishtest/stats/stat_util.py).

One million samples per temperature/color, NumPy 2.2.6, seed 20260922 plus the temperature index. Posterior sample means are checked against the analytic Dirichlet expected score, allowing six Monte Carlo standard errors. Games are assumed independent within each color and across temperatures; these games were not matched opening pairs. The intervals exclude opponent-calibration uncertainty, model misspecification, and transfer to other opponents.

## Sensitivity

Primary results retain the benchmark convention that unresolved 200-ply caps are draws. Treating every capped game as a loss or as a win moves the medians by at most about 16 Elo. Changing each prior component from 0.5 to 1 changes the medians by at most 2 Elo. Neither check overturns the broad temperature trend.

| Temperature | Caps all losses | Caps all wins | Uniform-prior estimate |
| --- | ---: | ---: | ---: |
| 0.0 | 1682 | 1695 | 1687 |
| 0.2 | 1703 | 1721 | 1710 |
| 0.4 | 1625 | 1639 | 1631 |
| 0.6 | 1583 | 1605 | 1593 |
| 0.8 | 1487 | 1518 | 1503 |
| 1.0 | 1381 | 1400 | 1391 |

Artifacts:

- Reproducible calculation: `output/temperature_elo/estimate.py` (run with `uv run python output/temperature_elo/estimate.py`).
- Posterior summaries, source hashes, prior sensitivity, cap sensitivity, rank probabilities: `benchmark/results/temperature_sweep_20260921_bayesian_elo.json`.
- Updated dashboard: https://app.artifactbin.dev/a/OyGiWE
