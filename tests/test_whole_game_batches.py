import numpy as np
import torch

from training.whole_game_batches import GameSampler, collate_games, domain_weighted_loss, game_index, microbatches


def corpus(lengths):
    return np.concatenate([np.array([0] + [i + 2] * (n - 2) + [1], dtype=np.uint16)
                           for i, n in enumerate(lengths)])


def test_index_and_padding_preserve_complete_games_and_eos():
    tokens = corpus([3, 5, 513, 514])
    index = game_index(tokens, 0, 1, chunk_size=7)
    assert index['lengths'].tolist() == [3, 5, 513]
    assert index['excluded'] == 1
    x, y = collate_games(tokens, index, np.arange(3), 512, 1)
    for row, length in enumerate(index['lengths']):
        assert x[row, 0] == 0 and y[row, length - 2] == 1
        assert np.all(y[row, length - 1:] == -1)
        assert np.array_equal(x[row, 1:length - 1], y[row, :length - 2])
        assert np.count_nonzero(y[row] != -1) == length - 1


def test_shuffle_visits_every_game_and_restores_across_epochs():
    index = game_index(corpus([3, 7, 11, 19]), 0, 1)
    sampler = GameSampler(index, 42)
    first, count = sampler.take(sum(index['lengths'] - 1))
    assert sorted(first.tolist()) == [0, 1, 2, 3]
    assert count == sum(index['lengths'] - 1)
    restored = GameSampler(index, 42, sampler.state())
    for target in [13, 50, 17]:
        actual, count = sampler.take(target)
        expected, expected_count = restored.take(target)
        assert np.array_equal(actual, expected) and count == expected_count
        assert target <= count < target + max(index['lengths'] - 1)


def test_accumulated_microbatch_gradient_matches_equal_domain_objective():
    data = {d + '_train': corpus([3, 5, 7]) for d in ['human', 'leela']}
    indices = {d: game_index(data[d + '_train'], 0, 1) for d in ['human', 'leela']}
    selected = {'human': np.array([0, 1]), 'leela': np.array([0, 1, 2])}
    counts = {d: int((indices[d]['lengths'][ids] - 1).sum()) for d, ids in selected.items()}
    parameter = torch.zeros(6, requires_grad=True)
    for x, y, domains in microbatches(data, indices, selected, 1, buckets=(4, 8), max_tokens=8):
        loss = domain_weighted_loss(parameter.expand(*x.shape, 6), y, domains, counts)
        loss.backward()
    reference = torch.zeros(6, requires_grad=True)
    for d in ['human', 'leela']:
        _, y = collate_games(data[d + '_train'], indices[d], selected[d], 8, 1)
        target = torch.tensor(y)
        (.5 * torch.nn.functional.cross_entropy(reference.expand(target.numel(), 6),
            target.reshape(-1), ignore_index=-1)).backward()
    torch.testing.assert_close(parameter.grad, reference.grad)
