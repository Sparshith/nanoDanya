# Training

- `train.py`: next-move cross-entropy pretraining.
- `train_distill.py`: tied-embedding browser student training; `DISTILL_ALPHA=0`
  trains from recorded moves only and does not load a teacher. The default `0.5`
  mixes recorded-move CE with teacher KL. Both modes support INT4-aware training
  with `QAT=1` and a separate `CKPT_DIR` initialized through `INIT_FROM`.
- `train_involved_pieces.py`: next-move training plus move-involved piece locations.
- `train_exact_policy.py`: exact expected-reward updates over legal moves, with checkpoint selection and resume.
- `common.py` and `grpo.py`: shared model, checkpoint, reward-table, and exact-policy loss helpers.

One-time traces, probes, and their tests live in the Git-ignored `scratch/`
directory. Local experiment instructions are in `scratch/README.md`.

Run the reusable-code tests with:

```sh
PYTHONPATH=.:nanochat .venv/bin/python -m pytest tests
```

## Board input experiment

`train_board_input.py` supplies actual piece locations to the move predictor.
`prepare_board_inputs.py` expands validated identities into the board after each
input token, including captures and promoted piece kinds. The board encoder adds
a vector to each move embedding before the transformer; the output vocabulary
is unchanged. Its projection starts at zero to preserve the initial policy.

`BOARD_INPUT_ENABLED=0` trains the same architecture with constant blank boards;
`1` supplies actual boards. Both use next-move CE only. The board-aware checkpoint
loader replays observed SAN during ordinary and cached inference. It requires
complete standard-start history. New checkpoints store the encoder separately
under `board_input` and must use `model_loading.load_model` for board-aware play.

`MODAL_PROFILE=second modal run --detach modal_board_input.py` prepares inputs,
runs two matched 1,000-update H100 continuations, and benchmarks 4,000 games plus
fixed-position metrics and a real/blank board ablation on separate probe games.
