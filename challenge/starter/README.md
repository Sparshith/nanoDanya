# Starter

The baseline, ready to run, train, or replace. Six layers, width 384, 16 million weights, 1260 Elo. Your job is to make it better, or to make something better.

| File | What it is |
|---|---|
| `model.py` | The baseline in PyTorch, about 100 lines. In: token ids `[batch, tokens]`. Out: logits `[batch, tokens, vocabulary]`. Train it, change it, or replace it with anything that has the same in and out. |
| `baseline.pt` | The baseline's weights, 9 MB. `export.py` downloads it the first time. Gitignored. |
| `export.py` | Turns weights into `model.onnx`, the one file you drop on the page. Writes the vocabulary and context length into the file, then prints the size against the 10 MB limit. |
| `vocabulary.json` | The token list: `<bos>`, `<eos>`, then one chess move per token in standard notation (`e4`, `Nf3`, `O-O`, `Qxf7#`). A token's id is its position in the list. The baseline was trained with it. You may edit it or bring your own; it must contain `<bos>`. |
| `check.py` | The page's three checks, on your laptop, plus a quick match against the baseline. Run it after every export. Go to the page only for the file you want to submit. |
| `tokenize_game.py` | Example: turns a game's moves into token ids with this vocabulary. Use the same rule when you build training data. |
| `requirements.txt` | torch, onnx, onnxruntime, python-chess, pinned. |

## Minute one

```sh
pip install -r requirements.txt
python3 export.py --int4
python3 check.py
```

`export.py` writes `model.onnx`, 9.63 MB. `check.py` runs the page's three checks on it and plays 10 games against the baseline. It is the baseline, so expect draws. Drop the file on the challenge page when you want to see it play on a board.

## Train

Load the baseline and keep training, or start fresh:

```python
from model import Model, load_weights

config, state, _ = load_weights("baseline.pt")
model = Model(config)            # qat=True by default: rounds weights to 4 bits during training
model.load_state_dict(state)     # skip this line to start from scratch
# ... your data, your loop ...
torch.save(model.state_dict(), "mine.pt")
```

Training data is games turned into token ids the way `tokenize_game.py` does it. The model predicts the next id from the ids so far. That is the whole objective.

`Model(config)` has `qat=True`: during training the forward pass rounds every weight to 4 bits, so what you train is what `--int4` exports. Pass `qat=False` if you plan to export at fp32 or int8.

## Export and test

```sh
python3 export.py --checkpoint mine.pt --name "Mine v3" --int4
python3 check.py
```

Precision is your lever against the 10 MB limit:

| Flag | Bits per weight | Fits about |
|---|---|---|
| none | 32 | 2.5M weights |
| `--int8` | 8 | 10M weights |
| `--int4` | 4 | 20M weights |

`--int4` packs every 2-D weight matrix whose width is a multiple of 64 and adds a few standard ONNX ops that unpack it at load time. It works for the baseline architecture and for most others exported through PyTorch. If it prints TOO BIG, use a smaller precision or a smaller model.

`check.py` prints the size, the load, the four test positions, and the match score. Iterate here. When a file wins, drop it on the page, run the 20-game match there too, and click Submit.

## How the page talks to your model

1. It reads the vocabulary from the file's metadata.
2. It loads the file with ONNX Runtime Web, in the visitor's browser.
3. For every move, it turns the game so far into your token ids and runs the model once.
4. It looks at the legal moves on the board, finds each one in your vocabulary, and plays the legal move with the highest score.

Your model never has to know the rules. It only scores moves.

## Limits

- One ONNX file, at most 10,000,000 bytes. Standard ops only, opset 17 or newer.
- Context length from 64 to 4,096 tokens.
- No search, no chess engine, no hand-written chess rules at inference time.
- Training: anything goes.

Full contract: `../llms.txt`.
