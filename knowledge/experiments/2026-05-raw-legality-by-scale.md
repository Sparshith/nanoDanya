# Raw Legality by Scale

Date: May 2026 (15M runs `created_at` 2026-05-31). Verified 2026-07-04 against
`benchmark/results/legality_*_on_actual_5m_4096.jsonl`.

Raw top-1 legality on 4096 held-out `actual_5m` validation positions, no legality
mask. Strict = exact SAN-token illegality. Real = strict minus under-disambiguated
SAN cases (see STATE.md measurement conventions). Phase columns are strict legal
rates.

| Model | Strict illegal | Real illegal | Opening | Middle | End |
| --- | ---: | ---: | ---: | ---: | ---: |
| `plain/games-500k` | 167/4096 = 4.077% | 158/4096 = 3.857% | 99.63% | 94.69% | 94.21% |
| `plain/puzzles-500k` | 163/4096 = 3.979% | 138/4096 = 3.369% | 99.72% | 94.44% | 95.74% |
| `plain/games-5m` | 22/4096 = 0.537% | 18/4096 = 0.439% | 99.91% | 99.38% | 98.98% |
| `plain/puzzles-5m` | 16/4096 = 0.391% | - | 99.91% | 99.59% | 99.15% |
| 15M games, L12, step 200k | 12/4096 = 0.293% | 10/4096 = 0.244% | 99.91% | 99.63% | 99.66% |

The 15M checkpoints ran from `/data/actual_15m/` and have since been registered:
`plain/games-15m` (L16, the registry champion) and `plain/games-15m/l12`, under
`checkpoints/plain/games-15m/`.

## Read

Scale drives raw legality down by an order of magnitude (3.9% real illegal at 500k
games to 0.44% at 5M), and the 15M model continues the trend. At 500k, puzzle-linked
and actual-game sources are roughly tied on legality. Errors concentrate in
middlegame and endgame; openings are near-perfect at every scale.

## Arch comparison at 15M (8192 positions, `actual_15m` val, 2026-05-31)

`legality_actual_15m_{L12_H6_E768,L16_H8_E1024}_best_8192.jsonl`: L16 is slightly
cleaner in the middlegame (0.243% vs 0.385% strict illegal), identical in the
endgame (0.970%). Small effect, consistent with the ~54.5% H2H edge.

Measurement command:

```bash
uv run python benchmark/run.py legality --model <alias> --output benchmark/<file>.jsonl
```

Known contrast preserved from earlier runs: `weighted/puzzles-5m` had much worse raw
legality than `plain/puzzles-5m`, part of why weighting is a settled negative result.
Special-token caveat: report legality pre-`<eos>` excluding `<bos>`; positions after a
first `<eos>` are benchmark artifacts.

Related: [API model comparison](2026-05-27-api-model-legality.md) on the same 4096
positions, and [king-safety SFT](2026-05-19-king-safety-sft.md) targeting the
in-check/late-position failure slices.
