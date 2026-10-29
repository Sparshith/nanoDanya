import pickle

import numpy as np
import pytest

from training.recipe import resolve_recipe


@pytest.fixture
def recipe(tmp_path):
    meta = dict(vocab_size=128, context_length=8,
        stoi={'<bos>': 0, '<eos>': 1, **{str(i): i for i in range(2, 128)}},
        itos={0: '<bos>', 1: '<eos>', **{i: str(i) for i in range(2, 128)}})
    datasets, next_id = {}, 2
    for name, lengths in [('archive', [2, 5, 6, 3, 6, 1, 4] * 2), ('engine', [3, 6, 2, 6, 1] * 2)]:
        folder = tmp_path / name
        folder.mkdir()
        tokens = []
        for length in lengths:
            tokens.extend([0, *range(next_id, next_id + length), 1])
            next_id += length
        np.array(tokens, dtype='<u2').tofile(folder / 'train.bin')
        np.array([0, 120, 121, 1], dtype='<u2').tofile(folder / 'val.bin')
        (folder / 'meta.pkl').write_bytes(pickle.dumps(meta))
        datasets[name] = str(folder)
    return resolve_recipe(dict(datasets=datasets,
        model_config=dict(sequence_len=8, vocab_size=128, n_layer=1, n_head=1, n_kv_head=1, n_embd=16),
        batch_size=4, microbatch_size=2, seed=123, device='cpu', compile=False,
        val_batches=1, eval_interval=2, checkpoint_interval=2,
        learning_rate=.001, min_learning_rate=.0001, warmup_steps=1))
