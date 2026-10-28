import numpy as np
import pytest
import torch

from training.coverage_batches import CoverageSampler, coverage_batch, coverage_index


@pytest.mark.parametrize('lengths', [[1], [7, 2, 5, 1], [8, 9, 31, 3], [1] * 30])
def test_every_move_and_eos_is_supervised_exactly_once(lengths):
    pieces, next_id = [], 2
    for length in lengths:
        pieces.extend([0, *range(next_id, next_id + length), 1])
        next_id += length
    tokens = np.array(pieces)
    windows, summary = coverage_index(tokens, 0, 1, width=8, chunk_size=7)
    sampler = CoverageSampler(windows, 42)
    covered = np.zeros(len(tokens), dtype=int)
    while sampler.cursor < len(windows):
        selected = sampler.take(3)
        x, y = coverage_batch(tokens, windows, selected, 3, 8, 0, 1)
        for row, (start, stop, target) in enumerate(windows[selected]):
            valid = np.flatnonzero(y[row].numpy() != -1)
            positions = start + 1 + valid
            covered[positions] += 1
            np.testing.assert_array_equal(y[row, valid], tokens[positions])
            np.testing.assert_array_equal(x[row, valid], tokens[positions - 1])
            assert np.all(positions >= target)
            if tokens[start] != 0:
                assert target - start == 4
        assert torch.all(y[len(selected):] == -1)
    np.testing.assert_array_equal(covered, tokens != 0)
    assert summary['supervised_tokens'] == int(covered.sum())
    assert summary['excluded_games'] == 0
    assert sampler.take(3).size == 0


def test_resume_preserves_shuffle_and_never_repeats_windows():
    tokens = np.tile([0, 2, 3, 1], 41)
    windows, _ = coverage_index(tokens, 0, 1, width=8)
    original = CoverageSampler(windows, 7)
    first = original.take(9)
    restored = CoverageSampler(windows, 7, original.cursor)
    remaining = restored.take(len(windows))
    np.testing.assert_array_equal(original.take(len(windows)), remaining)
    np.testing.assert_array_equal(np.sort(np.concatenate([first, remaining])), np.arange(len(windows)))
