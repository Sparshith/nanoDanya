import math

import chess
import numpy as np
import pytest
import torch

from inference.board_input import BoardState
from probes.square_readout import game_average, report, score_positions, square_loss, square_targets


def features(n):
    result = torch.zeros(1, 1, 32, 2, dtype=torch.long)
    result[..., 0] = 64
    result[0, 0, :n, 0] = torch.arange(n)
    result[0, 0, :n, 1] = 1
    return result


@pytest.mark.parametrize('n', [2, 10, 32])
def test_occupied_and_empty_each_receive_half_regardless_of_piece_count(n):
    boards = features(n)
    logits = torch.zeros(1, 1, 64, 13, requires_grad=True)
    loss = square_loss(logits, torch.tensor([[0]]), boards, {'<bos>': 0, '<eos>': 1})
    assert loss.item() == pytest.approx(math.log(13))
    loss.backward()
    gradient = logits.grad.abs().sum(-1)[0, 0]
    assert gradient[:n].sum().item() == pytest.approx(gradient[n:].sum().item())
    assert gradient[0].item() / gradient[n].item() == pytest.approx((64-n) / n)


def test_balance_is_per_board_not_over_the_batch_and_masks_invalid_prefixes():
    boards = torch.cat((features(2), features(32)), dim=1)
    logits = torch.zeros(1, 2, 64, 13)
    logits[:, 0, :, 0] = 4
    tokens = torch.tensor([[0, 2]])
    stoi = {'<bos>': 0, '<eos>': 1}
    combined = square_loss(logits, tokens, boards, stoi)
    separate = sum(square_loss(logits[:, i:i+1], torch.tensor([[0]]), boards[:, i:i+1], stoi) for i in range(2)) / 2
    torch.testing.assert_close(combined, separate)
    boards = features(10).expand(1, 4, 32, 2)
    logits = torch.zeros(1, 4, 64, 13, requires_grad=True)
    square_loss(logits, torch.tensor([[2, 0, 2, 1]]), boards, stoi).backward()
    assert logits.grad[:, 0].count_nonzero() == logits.grad[:, 3].count_nonzero() == 0
    assert logits.grad[:, 1:3].count_nonzero() > 0


def test_empty_everywhere_is_expensive_even_on_sparse_boards():
    logits = torch.zeros(1, 1, 64, 13)
    logits[..., 0] = 12
    sparse = square_loss(logits, torch.tensor([[0]]), features(2), {'<bos>': 0, '<eos>': 1})
    dense = square_loss(logits, torch.tensor([[0]]), features(32), {'<bos>': 0, '<eos>': 1})
    assert sparse > 6
    torch.testing.assert_close(sparse, dense)


@pytest.mark.parametrize('moves', [[], ['e4', 'e5', 'Nf3', 'Nc6', 'Bc4', 'Nf6', 'O-O'],
                                    ['e4', 'a6', 'e5', 'd5', 'exd6']])
def test_square_targets_match_independent_chess_replay(moves):
    state, board = BoardState(), chess.Board()
    features = state.append('<bos>')
    for move in moves:
        features = state.append(move)
        board.push_san(move)
    actual = square_targets(torch.tensor(features.astype(np.int64))[None, None])[0, 0].numpy()
    expected = np.zeros(64, dtype=int)
    for square, piece in board.piece_map().items():
        expected[square] = piece.piece_type + (0 if piece.color else 6)
    np.testing.assert_array_equal(actual, expected)


def test_promotion_uses_current_kind_not_original_identity():
    boards = features(0)
    boards[0, 0, 8] = torch.tensor([56, 5])
    boards[0, 0, 24] = torch.tensor([0, 2])
    target = square_targets(boards)
    assert target[0, 0, 56] == 5 and target[0, 0, 0] == 8
    assert (target != 0).sum() == 2


def test_metrics_have_no_credit_for_empty_squares_and_count_relocation_once():
    actual = np.zeros((1, 64), dtype=int)
    actual[0, 0], actual[0, 60] = 4, 12
    predicted = np.zeros_like(actual)
    empty = score_positions(actual, predicted)[0]
    assert empty['matched'] == 0 and empty['exact_boards'] == 0 and empty['errors'] == 2
    predicted[0, 1], predicted[0, 60] = 4, 12
    moved = score_positions(actual, predicted)[0]
    assert moved['errors'] == 1 and moved['missing'] == moved['extra'] == 1
    perfect = score_positions(actual, actual)[0]
    assert perfect['exact_boards'] == 1 and perfect['errors'] == 0


def test_game_average_does_not_give_long_games_more_weight():
    actual = np.zeros((1, 64), dtype=int)
    actual[0, 0] = 4
    good = score_positions(actual, actual)[0]
    bad = score_positions(actual, np.zeros_like(actual))[0]
    short = report([{'ply': 1, 'counts': good}])
    long = report([{'ply': ply, 'counts': bad} for ply in range(1, 20)])
    summary = game_average([short, long])
    assert summary['moves_1_10']['exact_board_rate'] == 0.5
    assert summary['moves_1_10']['positions'] == 20
    assert summary['moves_31_plus']['games'] == 0
