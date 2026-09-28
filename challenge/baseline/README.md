# Baseline

The model to beat: 1260 Elo, 12.9% puzzles solved, 9.63 MB as an ONNX file. A 16M-parameter transformer (6 layers, width 384, tied embeddings) trained by distillation, with every matrix stored in 4 bits.

The baseline is the starter. Its architecture is `../starter/model.py`, its weights are `../starter/baseline.pt`, and `../starter/export.py --int4` produces the exact file the page serves. Training code lives in `training/`.

This folder holds one thing:

| File | What it is |
|---|---|
| `test_export.py` | Exports the baseline with the starter script and checks its logits against the PyTorch model and against the served file on 1, 8 and 64-token inputs. |

Run it with `uv run pytest challenge/baseline/test_export.py`.
