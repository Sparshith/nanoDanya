# 10 MB student: corrected human–Leela recipe

Status: completed October 19, reviewed October 21. The new exported model regressed
on strength, puzzles and raw game reliability. Keep the existing d9-200k model.
Estimated compute $11.88 against the user-authorized $50 total experiment cap.

The existing d9-200k model forfeited 691/1,200 unmasked games (57.58%). Its
691 invalid choices among 21,751 decisions (3.18%) included 174 EOS predictions,
474 illegal moves and 43 notation errors. Per-decision error and per-game
forfeit rates have different denominators. The published 99.71% fixed-position
legality result belongs to the larger L12, not this 10 MB student.

## Results

Both stages completed: 200,000 float updates and 2,000 INT4-aware updates. Float
training supervised 2,956,025,379 human and 2,799,622,540 Leela tokens (5.756B total).
Both selected checkpoints are their final steps. Float mixed move CE decreased
from 1.88982 at 50k to 1.78345 at 100k, 1.71947 at 150k and 1.68608 at 200k.
Quantization initially raised CE to 2.00915; QAT recovered it to 1.74929. There is
no float game-strength evaluation, so this does not isolate quantization's role.

| Metric | Existing d9-200k | New human–Leela |
| --- | ---: | ---: |
| Raw puzzles, 8,800 each | 1,941 (22.06%) | 1,613 (18.33%) |
| Masked puzzles, 8,800 each | 2,023 (22.99%) | 1,694 (19.25%) |
| Raw forfeits, 1,200 engine games each | 563 (46.92%) | 609 (50.75%) |
| Raw score vs SF1000, 400 games each | 61.63% | 54.88% |
| Raw score vs SF1500, 400 games each | 34.13% | 30.63% |
| Raw score vs SF2000, 400 games each | 23.63% | 19.50% |
| Masked score vs SF1000, 400 games each | 80.38% | 75.50% |
| Masked score vs SF1500, 400 games each | 54.88% | 45.88% |
| Masked score vs SF2000, 400 games each | 31.75% | 24.50% |

New versus old scores 45.375% raw (400 games; paired-opening bootstrap 95% interval
41.0–49.75%) and 46.125% masked (400 games; 41.875–50.375%). Masked H2H alone is
inconclusive; the broader evaluation consistently favors the existing artifact.
Both masked models have zero forfeits, enforced by the harness.

The old model's freshly measured 46.92% raw forfeits, not September's 57.58%, are
the valid control: this evaluator preserves canonical check markers in histories.
The recipe changed data, context length, target masking and optimization together,
and used one seed. These results do not establish that Leela data itself hurts or
that small-model capacity is the cause. Next diagnostic: evaluate the existing
float checkpoint against its quantized export before committing to another retrain.
A human-only run under the same corrected recipe would isolate the data mixture.

The export is 9,454,897 bytes, SHA-256
`aa5f433fc724673fab307b2fea268e877c259540ab2789c35e0e07a8e096a0af`.
Export parity passed through 512 tokens (maximum logit error below 6e-6).
All 5,600 games passed legal replay; native ONNX invalid/near-tied decision checks,
forfeit FEN reconciliation and puzzle identity checks passed. Local `report.json`
records paired-opening intervals, puzzle totals, reliability and budget receipts.
Pipeline resource time was 9,940.72 seconds, approximately 2h46m. Estimated compute
including preflight/recovery is $11.8821, not a final provider invoice. No promotion.

## Training

- Fresh 9-layer, width-320, five-head tied-embedding StudentGPT, 14,750 SAN tokens.
  Same architecture as d9-200k; no teacher or RL.
- Existing 15M human corpus and cleaned T80 Leela corpus (52.15M training games,
  6.894B training tokens); tokenizer identities and file hashes verified.
- 50/50 domain-normalized CE. Ignore targets before the first visible BOS and
  artificial BOS targets, retain EOS. Full 512-token windows, not whole-game
  batching. Prior small model used 256-token windows, so this is a recipe upgrade,
  not an isolated data-source comparison or a matched-token ablation.
- 200,000 updates, batch 64, cosine 3e-4 to 3e-5, 2,000 warmup steps, AdamW,
  BF16 and compilation. Approximately 5.75B supervised tokens expected; report
  actual counts. Fixed dual-domain validation chooses the float checkpoint.
- 2,000 INT4-aware updates from float best, same data mixture and masks,
  cosine 3e-5 to 3e-6. Export must remain under 10,000,000 bytes.

## Evaluation

Both the existing d9-200k ONNX and new export are freshly evaluated. The tournament
history path now preserves canonical check markers, matching the corrected normal
game path. Historical September scores are not direct controls for this evaluator.

- 5,600 games: each model versus SF1000/SF1500/SF2000 and new versus old,
  400 games per pair, separately with and without legal masking.
- Greedy play, 200 fixed openings with both colors, depth-1 Fairy-Stockfish,
  10ms/1300cp adjudication and 400-ply caps. Internal protocol, not human Elo.
- 8,800 identical puzzles per model per mask mode. No test-driven selection.
- Native ONNX parity, invalid/near-tied prediction checks, legal replay, forfeit
  token/FEN reconciliation, puzzle identity checks and paired-opening intervals.

## Preflight and budget

32 local tests passed, including real float/QAT student interruption recovery,
equal domain loss weighting, fragment masking and canonical tournament history.
H100 preflight measured 0.02597 seconds/update, projecting 5,194 seconds for float
training before validation/checkpoint overhead. Random-weight INT4 export was
9,454,897 bytes; native ONNX parity passed through 512 tokens.

Initial preflight completed training/export but failed to import the Modal SDK
from the subprocess for volume commit. The enclosing runner preserved the results.
An opt-in parent/subprocess commit handshake was added and verified by a short CPU
recovery, without repeating GPU work. Checkpoint commits are now acknowledged
before training continues.

Full timeout reservations plus $5 overhead reserve total $42.07 of $50. Includes
the failed preflight reservation. H100/CPU/memory rates verified October 19;
resource-time estimates are not a final invoice. Main job has a 30,000-second
external timeout and a persistent internal deadline; no automatic job retries.

App: https://modal.com/apps/workspace2/main/ap-kDUi0prgokO9oUMjfI6jq1

Local: `output/small-leela-20261019/` (protocol, budget ledger, launcher log;
results downloaded and independently summarized automatically after completion).
Remote: `/data/experiments/small-leela-20261019/` on `nanodanya-data`.
Scripts: `scratch/small_leela_20261019.py`, `scratch/small_leela_eval.py`,
`scratch/run_small_leela.py`, `scratch/report_small_leela.py`.
No production promotion, commits or pushes are authorized by this run.
