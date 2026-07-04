# Architecture mental model

Distilled from the old ARCHITECTURE.md (deleted 2026-07-04); this keeps the parts
that stay true regardless of which experiment lines exist.

## Pipeline

```
raw games -> dataset builder -> tokenizer contract -> training objective
          -> checkpoint selection -> inference mask -> benchmark/report
```

Keep these boxes separate. The model never sees "chess"; it sees whatever move
sequences the data induces. Legality is learned from examples, not from a rules
engine inside the network.

## Token contract

The model predicts whole SAN move tokens (not UCI, not piece-square actions), with
`<bos>`/`<eos>` specials. `meta.pkl` in each dataset dir defines the vocab and is
part of checkpoint compatibility: same architecture shape does not imply same vocab
semantics (older vocabs keep `+`/`#` annotations, newer ones strip them). All SAN
normalization and legality matching is centralized in `chess_token_utils.py`
(`strip_san`, `normalized_legal_sans`, `token_is_legal_prediction`,
`token_is_playable`, `resolve_token_id`); `tests/` covers this plus move selection.
A mismatch between tokenizer semantics and runtime legality logic produces fake
regressions.

## Which knob moves which metric

- Raw top-1 legality (the model-cleanliness metric): training objective, data,
  checkpoint selection. Runtime cannot change it.
- Played legality: inference-time masking (`python-chess` board + legal-token mask).
  Should be ~0 by construction; it measures model + harness, never the model.
- Reported numbers: benchmark definition (`<eos>` handling, `<bos>` exclusion,
  under-disambiguated SAN). See "Measurement conventions" in STATE.md.

"Best" checkpoints are only best relative to the metric that selected them; a
best-combined-objective snapshot can be worse at clean next-move prediction
(this happened; see the eval-aware negative results in STATE.md).

## Experiment checklist

Every run should be able to answer:

1. Which dataset built this checkpoint?
2. Which tokenizer contract did it use?
3. Which training objective did it optimize?
4. Which metric chose the checkpoint as "best"?
5. Which benchmark configuration produced the reported number?

## Source-of-truth files

`model_registry.py`, `training/common.py`, `chess_token_utils.py`,
`benchmark/run.py`, `inference/serve.py`.
