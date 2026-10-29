from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from training.recipe import BatchStream, load_data, resolve_recipe


@pytest.mark.parametrize('mode', ['single', 'coverage', 'hybrid'])
def test_coverage_preserves_each_target_once_and_its_context(recipe, mode):
    tokens, next_id = [], 2
    for length in [1, 7, 8, 9, 19, 2]:
        tokens.extend([0, *range(next_id, next_id + length), 1])
        next_id += length
    np.array(tokens, dtype='<u2').tofile(Path(recipe['datasets']['archive']) / 'train.bin')
    recipe['batch_size'] = 7
    if mode == 'single':
        recipe['datasets'].pop('engine')
    elif mode == 'hybrid':
        recipe['batch_size'] = 4
        recipe['sampling'] = dict(mode='mixture', windows_per_source={'archive': 2, 'engine': 2},
            coverage_source='archive')
    data, meta, _ = load_data(recipe)
    stream = BatchStream(data, meta, resolve_recipe(recipe))
    covered = ['archive'] if mode == 'hybrid' else list(data)
    observed = {name: Counter() for name in covered}
    saw_padding, saw_overlap = False, False
    while stream.step < stream.total_steps:
        x, y, ids = stream.take()
        for row, source in enumerate(ids.tolist()):
            if source == -1:
                assert torch.all(y[row] == -1)
                saw_padding = True
                continue
            name = list(data)[source]
            if name not in covered:
                continue
            corpus = data[name]['train']
            for col, target in enumerate(y[row].tolist()):
                if target == -1:
                    continue
                assert target != 0
                # Move IDs are unique; identify EOS by the move immediately before it.
                position = int(np.flatnonzero(corpus == (target if target != 1 else x[row, col].item()))[0])
                if target == 1:
                    position += 1
                assert corpus[position] == target
                assert x[row, :col + 1].tolist() == corpus[position - col - 1:position].tolist()
                observed[name][position] += 1
                saw_overlap |= x[row, 0].item() != 0
    for name in covered:
        assert observed[name] == Counter(np.flatnonzero(data[name]['train'] != 0))
    assert saw_overlap
    if mode != 'hybrid':
        assert saw_padding
    with pytest.raises(StopIteration):
        stream.take()


def test_mixture_obeys_quotas_and_masks_incomplete_game_history(recipe):
    recipe['sampling'] = dict(mode='mixture', windows_per_source={'archive': 1, 'engine': 3}, steps=8)
    data, meta, _ = load_data(recipe)
    stream = BatchStream(data, meta, resolve_recipe(recipe))
    saw_fragment, saw_move, saw_eos = False, False, False
    for _ in range(8):
        x, y, ids = stream.take()
        assert Counter(ids.tolist()) == {0: 1, 1: 3}
        for row, source in enumerate(ids.tolist()):
            corpus = data[list(data)[source]]['train']
            starts = [i for i in range(len(corpus) - x.shape[1])
                      if np.array_equal(corpus[i:i + x.shape[1]], x[row].numpy())]
            assert len(starts) == 1
            start, has_start = starts[0], False
            for col, token in enumerate(x[row].tolist()):
                has_start |= token == 0
                next_token = int(corpus[start + col + 1])
                expected = next_token if has_start and next_token != 0 else -1
                assert y[row, col].item() == expected
                saw_fragment |= not has_start
                saw_move |= expected > 1
                saw_eos |= expected == 1
    assert saw_fragment and saw_move and saw_eos
