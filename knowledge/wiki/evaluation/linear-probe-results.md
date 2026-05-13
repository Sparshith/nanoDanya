# Linear Probe Results

Run date: 2026-05-01

## Setup

- Position set: 100 validation games from `data/processed/val.bin`
- Positions: 7,960
- Labels: board state, side to move, check status, legal move ids, Stockfish depth-8 eval bucket
- Split: 80/20 by game
- Probe: linear readout from frozen hidden states
- Epochs: 20
- Combined metrics: `data/probes/metrics_all_val_100_sf8.csv`

## How To Read This

Read `pieces_occ` and `legal_f1` first.

- `pieces_occ`: can a hidden state reconstruct the occupied squares?
- `legal_f1`: can a hidden state recover the legal continuation set?

Baselines are dumb frequency guesses with no hidden states:

- `pieces_occ` baseline: `0.253`
- `legal_f1` baseline: `0.227`
- `eval_bucket` baseline: `0.427`

Rough interpretation:

- `0.70+`: real chess state is linearly readable.
- `0.78+`: strong internal board/legal-state representation.
- Near baseline: not meaningfully readable.

## Best Layer Per Model

| Model | pieces_occ | legal_f1 | eval_bucket |
|---|---:|---:|---:|
| `plain/games-500k/l4-gpu-legacy` | 0.666 L3 | 0.628 L3 | 0.427 L1 |
| `plain/games-500k/l8` | 0.717 L6 | 0.713 L6 | 0.401 L2 |
| `plain/games-500k` | 0.713 L9 | 0.710 L9 | 0.406 L9 |
| `weighted/middlegame-ft` | 0.715 L9 | 0.713 L9 | 0.392 L9 |
| `plain/puzzles-500k` | 0.718 L9 | 0.707 L11 | 0.395 L6 |
| `plain/puzzles-5m` | 0.777 L9 | 0.782 L9 | 0.406 L3 |
| `weighted/puzzles-5m` | 0.786 L9 | 0.784 L11 | 0.406 L6 |
| `eval-aware/weighted-ft` | 0.789 L9 | 0.783 L11 | 0.469 L11 |
| `eval-aware/v1` | 0.448 L9 | 0.222 L11 | 0.379 L3 |
| `eval-aware/v2` | 0.422 L0 | 0.169 L11 | 0.324 L9 |

## Main Read

Puzzle-derived data improves the model's internal board and legal-move
representation.

The clearest comparison:

| Model | pieces_occ | legal_f1 |
|---|---:|---:|
| `plain/games-500k` | 0.713 | 0.710 |
| `plain/puzzles-5m` | 0.777 | 0.782 |
| `weighted/puzzles-5m` | 0.786 | 0.784 |

`eval-aware/weighted-ft` keeps the strong board/legal representation and
is the only model that clearly beats the eval-bucket baseline:

| Model | eval_bucket | baseline |
|---|---:|---:|
| `eval-aware/weighted-ft` | 0.469 | 0.427 |

The eval-aware checkpoints look representationally weak under this probe. Their
legal-move F1 is near or below the frequency baseline:

| Model | legal_f1 | baseline |
|---|---:|---:|
| `eval-aware/v1` | 0.222 | 0.227 |
| `eval-aware/v2` | 0.169 | 0.227 |

## Interpretation

Best current story:

1. Next-token training on better chess data teaches a readable board/legal-state
   representation.
2. Puzzle-derived data improves that representation substantially.
3. Eval supervision helps eval readout when added on top of a good representation.
4. Eval-aware scratch-style checkpoints did not learn a clean linearly readable
   chess state here.

This supports treating data quality and next-token pressure as the primary levers
for learned chess structure. Auxiliary objectives are useful only when they do not
damage the underlying board/legal representation.
