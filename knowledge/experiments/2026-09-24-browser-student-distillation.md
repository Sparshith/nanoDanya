# Under-10 MB browser student

Status: training in progress. The size target is verified with synthetic weights;
playing strength has not yet been measured.

User authorized a $100 Modal budget in the workspace2 workspace (`MODAL_PROFILE=second`).
All GPU jobs use `training/run_distillation.py`, which reserves the full resource
limit before launch and retains reservations after failures. The ledger is
`output/distill-20260924/budget.json`; $10 is held for setup and other overhead.
Reservations are conservative bounds, not billing receipts.

## Inputs and model

Teacher: `/data/checkpoints/plain/games-15m/l12_best.pt`, the 107.6M-parameter
L12 model underlying the approximately 110 MB browser export. Training uses its
original checkpoint, not its browser INT8 approximation. This is distinct from
the larger L16 model selected by the registry's unqualified `plain/games-15m` name.

Teacher and `datasets/games/15m` copied directly from the original Modal workspace
into workspace2. No large model or dataset was downloaded locally. Source file hashes
and sizes are recorded in `output/distill-20260924/assets.json` and on the volume
at `/data/experiments/distill-20260924/asset-copy.json`.

Student: causal decoder, 6 layers, width 384, 6 attention heads, 512-token context,
14,750 SAN tokens, tied input/output embedding. Total 16,280,832 parameters.
RMSNorm, Q/K normalization, RoPE, ReLU-squared MLP, logit softcap 15. Every matrix
uses symmetric signed INT4 with groups of 64 and float32 scales. Self-contained
ZIP includes weights, vocabulary, RoPE table, and browser inference code.
Synthetic export: 9,424,814 bytes, below the strict 10,000,000-byte target.

## Training recipe

Objective: equal weights on recorded next-token cross entropy and teacher KL,
with temperature 2 and the usual temperature-squared KL scaling. AdamW with
weight decay 0.1 and gradient clipping at 1.0. BF16 autocast on H100.

- Pilot: 100 updates, batch 32 x 256 tokens, LR 3e-4. Validation CE 5.3380,
  combined loss 7.8270. Measured function runtime 28.29 seconds (~$0.034 compute,
  excluding startup/build costs). Steady-state throughput approximately 200k tokens/s.
- Main: resume pilot; batch 32 x 256 with accumulation 2, total update target
  25,000, cosine LR 3e-4 to 3e-5, warmup through step 200. Roughly 409M tokens
  including pilot. Trainer time cap 3,300 seconds; Modal execution timeout 3,600.
- Quantization-aware tuning and final evaluation: pending.

Checkpoint selection uses fixed batches in the first half of `val.bin`.
Final paired evaluation starts at the next complete game after the midpoint,
using only the second half. No final evaluation scores guide checkpoint selection.

## Verification

Local tests passed for student/GPT forward parity, INT4 fake-quant gradients, and
export/dequantization agreement. Synthetic browser checker passed histories of
1, 8, 32 and 64 tokens. Browser logits and the dequantized Python model matched
to approximately 1e-6 on sampled scores. These checks establish implementation
compatibility, not chess strength. Final artifact and game results are pending.
