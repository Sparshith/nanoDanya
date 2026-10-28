import numpy as np
import torch
import torch.nn.functional as F

from training.context_batches import context_targets, sample_context_batch


def test_only_targets_with_their_game_start_receive_loss():
    blocks = np.array([[7, 8, 1, 0, 2, 3, 1, 0, 4], [0, 2, 3, 1, 0, 5, 6, 1, 0]])
    x, y = context_targets(blocks, 0)
    assert x.tolist() == blocks[:, :-1].tolist()
    assert y.tolist() == [[-1, -1, -1, 2, 3, 1, -1, 4], [2, 3, 1, -1, 5, 6, 1, -1]]
    assert blocks[0, 1] == 8


def test_long_fragment_without_game_start_has_no_supervision():
    _, y = context_targets(np.array([[2, 3, 4, 5, 6]]), 0)
    assert np.all(y == -1)


def test_masked_targets_produce_zero_gradient():
    _, y = context_targets(np.array([[7, 8, 1, 0, 2, 3, 1]]), 0)
    logits = torch.zeros(6, 10, requires_grad=True)
    F.cross_entropy(logits, torch.tensor(y[0]), ignore_index=-1).backward()
    assert torch.count_nonzero(logits.grad[:3]) == 0
    assert torch.count_nonzero(logits.grad[3:]) > 0


def test_sampler_resumes_identically_and_keeps_next_token_alignment():
    data = np.tile(np.array([0, 2, 3, 4, 1]), 100)
    rng = np.random.default_rng(42)
    saved = rng.bit_generator.state
    first = sample_context_batch(data, rng, 8, 16, 0, 'cpu')
    restored = np.random.default_rng()
    restored.bit_generator.state = saved
    second = sample_context_batch(data, restored, 8, 16, 0, 'cpu')
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    x, y = first
    for row in range(len(x)):
        for pos in range(x.shape[1] - 1):
            if y[row, pos] != -1:
                assert y[row, pos] == x[row, pos + 1]
                assert 0 in x[row, :pos + 1]
