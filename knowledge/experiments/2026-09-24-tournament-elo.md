# Tournament Elo with a pinned sf1500 anchor

BayesElo rates the 15M-game L12 INT8 export (`model_q8.onnx`, CPU) at
**1766 +18/-17** on a scale where Fairy-Stockfish 14 at `UCI_Elo=1500`, depth 1,
is fixed at 1500. The scale is not a human or Lichess rating.

| Player | Elo | 95% CI | Games |
|---|---:|---:|---:|
| nanodanya-l12-int8 | 1766 | +18/-17 | 1200 |
| sf2000 (depth 1) | 1663 | +16/-16 | 1200 |
| sf1500 (depth 1, anchor) | 1500 | | 1200 |
| sf1000 (depth 1) | 1242 | +20/-21 | 1200 |

Raw scores for the model: 92.7% vs sf1000 (365W/12D/23L), 78.0% vs sf1500
(306W/12D/82L), 65.2% vs sf2000 (250W/22D/128L). Pool round robin:
sf1500 beat sf1000 313-83 (4 draws), sf2000 beat sf1000 363-31 (6),
sf2000 beat sf1500 276-111 (13).

## Protocol

Follows DeepMind searchless_chess `tournament.py`. There are 200 openings, sampled
with seed 1 from lichess-org/chess-openings@c67912be (CC0), 6 to 16 plies, all in
the challenge vocabulary (`benchmark/openings.tsv`). Each opening is played once
with each color. The model plays greedy (temperature 0) with a legal mask. Games end
on outcome with draw claims, or by adjudication when a 10 ms full-strength eval
exceeds 1300 cp or finds mate. Terminations: 2,331 of 2,400 adjudicated, 63
threefold, rest stalemate, fifty-move or material. There were no unscored, capped
or context-limit games. The fit uses BayesElo `mm` with default advantage and
drawelo, then `exactdist` and `offset 1500 sf1500`. The engine and the BayesElo
source are pinned by sha256 in `modal_benchmark.py`.

## Reading

- Depth 1 compresses the Stockfish levels. sf2000 is only +163 over sf1500, and
  sf1000 is -258. `UCI_Elo` labels are not Elo differences at depth 1.
- The same model scored 44.0% vs sf1500 under the 2026-09-21 protocol, but 78.0%
  here. That protocol used temperature 0.8 from the start position, with no
  adjudication and 200-ply caps counted as draws. Greedy play, openings and
  adjudication all differ, so the protocols are not comparable.
- The rated artifact is the CPU INT8 ONNX export. The challenge ZIP holds the
  WebGPU INT8 export, whose strength is not measured.

## Artifacts

Volume: `/data/tournament/games/{pool-sf-depth1_20260924_135642,baseline-l12-int8_20260924_135500}.jsonl`.
Local: `benchmark/results/modal_tournament_*_20260924_*.jsonl`, `modal_elo_20260924_135728.jsonl`.
Rerun: see "Tournament Elo" in `benchmark/README.md`.
