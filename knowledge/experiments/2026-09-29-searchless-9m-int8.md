# Searchless Chess 9M compressed below 10 MB

DeepMind's pretrained action-value model exports to **9,190,933 bytes** with
INT8 weights and substantially outperforms our current small model. No training
or fine-tuning was performed. The model is saved on workspace2; browser integration
has not been implemented or tested.

| Model | File size | Combined internal Elo | Puzzles solved |
| --- | ---: | ---: | ---: |
| Our d9-200k INT4 | 9.45 MB | 1525 ±12 | 2023/8800 (22.99%) |
| Searchless 9M INT8 | 9.19 MB | 2355 (+44/-38) | 6834/8800 (77.66%) |
| Searchless 9M FP32 control | Outside the size limit | 2448 (+61/-51) | 6842/8800 (77.75%) |

The INT8 model directly beat d9-200k **382 wins, 18 draws, zero losses** over
200 openings with both colors, a 97.75% score. The baseline's previous 1517 Elo
becomes 1525 when all games are fitted together; its weights are unchanged.
The larger L12 baseline is 1761 ±22 in this combined fit.

The combined rating table includes INT8's direct games against d9-200k, which
FP32 did not play. For the compression comparison, a separate fit gives both
Searchless versions exactly the same five Stockfish opponents and 2,000 games:
**INT8 2412 (+54/-46), FP32 2452 (+61/-51)**. Their matched scores are 99.05%
and 99.30%, a difference of -0.25 percentage points (paired opening bootstrap
95% interval -0.65 to +0.15). These results do not establish a significant
playing-strength loss from quantization.

INT8 solves eight fewer puzzles, gaining 44 and losing 52 relative to FP32.
The paired accuracy difference is -0.091 percentage points, with a 95% interval
of -0.309 to +0.127. All 8,800 puzzle identities match the previous evaluation.
Overlap between this puzzle sample and upstream ChessBench training has not
been audited, so the game results are the stronger evidence of improvement.

## Model and validation

Source: [Google DeepMind Searchless Chess](https://github.com/google-deepmind/searchless_chess),
revision `90ae0e6b121673fc3079aaeffa047580bb600c0a`, published `9M` action-value
checkpoint at step 6,400,000, using `params`, not EMA parameters. The architecture
has 8 layers, width 256, 8 attention heads, and 8,954,240 parameters. The published
state-value and behavioral-cloning archives were also inspected; they use a
different Flax layout and were not converted or benchmarked in this experiment.

The input is 77 FEN tokens, a legal UCI move ID, and a dummy target. The network
predicts a distribution over 128 win-probability buckets for each legal move.
Inference chooses the highest expected value, retaining the upstream repetition
draw adjustment. SAN histories are replayed only to adapt the existing benchmark
interface; they are not the neural network's input. Legal moves have no SAN
vocabulary restriction in the Searchless adapter.

All matrices use symmetric INT8 storage with one FP32 scale per row. LayerNorm
and activations remain FP32. Runtime dequantizes weights, so 9.19 MB is the model
file size, not a RAM limit. The ONNX file is standalone and has no external weights.

On 66 positions, including castling, en passant and promotion cases:

- Original JAX versus PyTorch maximum log-probability error: 2.29e-5.
- Native ONNX CPU versus quantized PyTorch: 2.48e-5.
- Native ONNX CPU versus the reconstructed GPU model: 2.67e-5.
- CPU inference for a batch of 32 candidate moves: median 124.45 ms with four
  ONNX Runtime threads, excluding board encoding and move selection.

The export passes the ONNX checker and the strict 10,000,000-byte limit.
GPU tournaments reconstruct the exact exported weights in PyTorch. This is
validated native inference, not a browser or browser-latency test.

## Evaluation and cost

The experiment added 6,400 games: 2,800 initial games, followed by 2,000 stronger
engine calibration games and 1,600 model games against the stronger settings.
The combined BayesElo fit uses 28,000 games; the matched-opponent fit uses 27,600.
All 6,400 new games replay legally, match the prescribed openings, and are scored.

Temperature 0, 200 fixed openings with both colors, the same Fairy-Stockfish
binary, depth 1, 10 ms/1300 cp adjudication, and a 400-ply limit match the previous
small-model tournament. The anchor is sf1500=1500. The extra sf2500 and sf2800
settings calibrate to only 1766 and 1796 in this pool. Searchless largely
saturates even this expanded pool, so its absolute Elo is extrapolated above
the opponents. These are internal benchmark ratings, not human or Lichess Elo.

Estimated function compute is approximately **$2.15 for this experiment**, and
**$48.86 cumulatively** including the prior runs. Timeout reservations and retained
overhead remain below the user's $100 total cap. Figures are resource-time
estimates, not a final invoice. No automatic retries or training jobs were used.

## Artifacts

workspace2 Modal profile `second`, volume `nanodanya-data`:

- `/data/experiments/searchless-9m-20260929/9M-int8.onnx`
- `/data/experiments/searchless-9m-20260929/searchless-9m-int8.zip`
- Model SHA-256: `c473e42c09ccbe94d7a618d632984a46a22d5f14e43b014156e19fe5df0e1d8a`

The ZIP contains the model, source snapshots, and upstream attribution. Weights
retain the upstream CC-BY-4.0 license; upstream code is Apache-2.0.
Local evidence and budget ledgers are in `output/searchless-9m-20260929/`.
Scripts are `scratch/searchless_{prepare,model,export,evaluate,stronger,finalize}.py`
and `scratch/report_searchless.py`. Existing benchmark and training sources are
unchanged, verified against the initial source manifest. No deployment was changed.
