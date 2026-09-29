# Larger L12 model: epoch accounting and million-step cost pilot

The actual `checkpoints/plain/games-15m/l12_best.pt` checkpoint is saved at
**191,700 steps**, with `best_step=191700`. Earlier notes call this the 200k-step
model; that is not its exact saved step. This is the 107,590,656-parameter L12,
width 768 model underlying the approximately 1757-rated INT8 browser baseline,
not the registry-default L16.

The dataset has 1,045,627,425 training tokens and 116,242,624 validation tokens.
At batch 64, sequence length 512 and no accumulation, each update processes
32,768 tokens. One million updates represent 32.768 billion tokens, or **31.338
equivalent epochs**. Going from the saved checkpoint to one million adds 808,300
updates and 25.331 equivalent epochs. Sampling uses random windows, so these
are token-exposure equivalents, not exact passes through every game.

Batch 64 is the original weighted trainer's default (git commit `0db2372`,
`training/train_weighted.py`), but the historical launch override is unverified.
The checkpoint stores model configuration and tokenizer metadata, not batch
size or RNG state. Its exact lifetime token count therefore remains unknown.
At batch 32 and context 512, one million updates would instead be 15.669 epochs.

## H100 throughput pilot

Loaded the verified original checkpoint and saved AdamW state: learning rate
1e-4, betas 0.9/0.999, weight decay 0.01. Used batch 64 × 512, BF16 autocast,
compiled model, recorded next-move CE, gradient clipping 1.0 and no accumulation.
Ten warmup updates followed by 25 measured updates averaged **0.06878 seconds
per step**, approximately 476,420 tokens/second. The 35 temporary updates were
discarded; the original checkpoint was untouched.

That projects approximately **15.44 H100 hours and $65.87 training compute**
for the 808,300 additional updates. This is a short throughput sample, excluding
validation, checkpoint writes, exports, evaluation and startup. Rates were
verified against [Modal pricing](https://modal.com/pricing): H100 plus four CPU
cores and 16 GiB RAM totals $0.00118492/second.

An initial attempt stopped before training because it asserted the old 200k
step count. The corrected pilot inspected the actual step and completed.
Combined measured pilot compute was approximately $0.09; total estimated
compute including earlier experiments is now approximately $46.72.
Both pilot apps were verified stopped with zero tasks.

## Proposed continuation, not launched

Resume the L12 checkpoint with batch 64 × 512 and its saved optimizer settings.
Retain milestones at 400k, 600k, 800k and 1m; compare validation, the same 8,800
puzzles, and temperature-0 tournament strength after export. Preserve the
original model. One million is the total target, not one million additional
updates. The pilot does not establish that further training improves Elo.

The full projection exceeds the user's existing $100 total cap even before
evaluation. A $150 total cap would leave room for evaluation and runtime variance.
No full continuation has been launched. Details: `output/champion-million-pilot-20260929/`
contains `report.json`, `continuation-plan.json`, ledgers, logs and stopped-app
evidence. `scratch/champion_million_pilot.py` reproduces the speed measurement.

Prior completed jobs' unused timeout reservations were released for this pilot
using their resource receipts, while retaining every prior overhead allowance.
Original ledgers are unchanged; the reconciliation is saved alongside the pilot.
