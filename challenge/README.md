# nanoDanya challenge

Chess in 10 MB, no search. An entry is one ONNX file. The page checks it and plays it in the browser; the maintainers score it offline on the same file.

- `starter/`: copy this to enter. The baseline model in PyTorch, its weights (downloaded), the export script with fp32, int8 and int4 output, the default vocabulary, a tokenizer example.
- `baseline/`: the test that the starter's int4 export matches PyTorch and the served file.
- `llms.txt`: the full contract, written for a coding agent. The page's "Copy prompt for your agent" button points at it.

The page itself is at https://www.sparshith.com/nd-challenge/. Its source is not in this repository yet. You do not need it to enter.

## Entry contract v2

One ONNX file, at most 10,000,000 bytes, standard ops only, opset 17 or newer. One int64 input `[1, tokens]`, one float32 output `[1, vocabulary]` or `[1, tokens, vocabulary]`. Metadata: `vocabulary` (JSON array of token strings, must contain `<bos>`), `context_length` (64 to 4096, default 512), `name`. One token per SAN move, `<bos>` first. The page picks the highest-scoring legal move from the entry's own list. Full text in `llms.txt`.

## Submissions

After the check passes, "Submit model" opens a dialog with a copyable prompt for a coding agent. The prompt includes the file's SHA-256, size, and the full check report. It tells the agent to host that exact file, link the training and export source at a commit, and open an issue in `Sparshith/nanoDanya`.

Issues are the manual evaluation queue; creating one does not launch compute. Download the file, verify its hash and size, review the source, then run the 8,800-puzzle and 1,200-game evaluations through `inference/onnx_policy.py`, attach the results and BayesElo fit to the issue, and update the leaderboard with the file hash.

**Compatibility is not eligibility.** The browser check does not prove rule compliance, strength, or absence of search. Those need source review and the offline evaluation.
