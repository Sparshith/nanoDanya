import numpy as np
import torch

from probes.full_board_readout import decode, full_loss, report, score_position


def board(*pieces):
    result = np.zeros((32, 2), dtype=np.int64)
    result[:, 0] = 64
    for identity, square, kind in pieces:
        result[identity] = square, kind
    return result


def test_same_color_same_kind_identities_can_swap_but_colors_cannot():
    actual = board((0, 0, 4), (7, 7, 4), (16, 56, 4))
    swapped = board((0, 7, 4), (7, 0, 4), (16, 56, 4))
    assert score_position(actual, swapped)['exact_boards'] == 1
    swapped[0, 0], swapped[16, 0] = 56, 7
    assert score_position(actual, swapped)['errors'] == 2


def test_missing_extra_overlap_and_misplacement_are_counted_without_empty_credit():
    actual = board((0, 0, 4), (8, 8, 1))
    misplaced = board((0, 0, 4), (8, 0, 1))
    row = score_position(actual, misplaced)
    assert (row['missing'], row['extra'], row['errors'], row['overlap_positions']) == (1, 1, 1, 1)
    missing = score_position(actual, board((0, 0, 4)))
    assert (missing['missing'], missing['extra'], missing['errors']) == (1, 0, 1)
    # An extra duplicate rook on a correct square must not disappear in a set.
    ghost = score_position(actual, board((0, 0, 4), (7, 0, 4), (8, 8, 1)))
    assert (ghost['missing'], ghost['extra'], ghost['matched'], ghost['overlap_positions']) == (0, 1, 2, 1)
    empty = score_position(actual, board())
    assert empty['matched'] == 0 and empty['errors'] == 2


def test_promotions_material_and_kings():
    actual = board((4, 4, 6), (8, 56, 5), (9, 9, 1), (20, 60, 6))
    wrong_queen = board((4, 4, 6), (8, 56, 1), (9, 9, 1), (20, 60, 6))
    row = score_position(actual, wrong_queen)
    assert (row['actual_material'], row['predicted_material'], row['matched_material']) == (10, 2, 1)
    assert row['errors'] == 1 and row['both_kings_correct'] == 1
    wrong_king = actual.copy()
    wrong_king[4, 0] = 5
    row = score_position(actual, wrong_king)
    assert row['both_kings_correct'] == 0 and row['exact_boards'] == 0
    assert row['actual_material'] == row['matched_material'] == row['predicted_material']
    missing_queen = actual.copy()
    missing_queen[8] = 64, 0
    missing_pawn = actual.copy()
    missing_pawn[9] = 64, 0
    q = report([{'ply': 1, 'counts': score_position(actual, missing_queen)}])['aggregate']
    p = report([{'ply': 1, 'counts': score_position(actual, missing_pawn)}])['aggregate']
    assert q['piece_f1'] == p['piece_f1']
    assert q['material_f1'] < p['material_f1']


def test_move_bands_exclude_initial_position_from_primary():
    actual = board((4, 4, 6), (20, 60, 6))
    correct = score_position(actual, actual)
    records = [{'ply': ply, 'counts': correct} for ply in (0, 1, 20, 21, 60, 61)]
    result = report(records)
    assert result['aggregate']['counts']['positions'] == 5
    assert [r['counts']['positions'] for r in result['by_stage'].values()] == [1, 2, 2, 1]
    assert result['aggregate']['material_f1'] is None
    assert result['aggregate']['exact_board_rate'] == 1


def test_full_head_learns_captured_status_without_padding_or_truth_filtering():
    tokens = torch.tensor([[2, 0, 2, 1]])
    boards = torch.tensor(np.stack([board((4, 4, 6))] * 4)).unsqueeze(0)
    logits = torch.zeros(1, 4, 32, 71, requires_grad=True)
    loss = full_loss(logits, tokens, boards, {'<bos>': 0, '<eos>': 1})
    loss.backward()
    assert logits.grad[:, 0].count_nonzero() == logits.grad[:, 3].count_nonzero() == 0
    assert logits.grad[0, 1, 0, 64] < 0  # Captured identities are explicitly learned.
    assert logits.grad[0, 1, 0, 65:].count_nonzero() == 0
    assert logits.grad[0, 1, 4, 70] < 0
    with torch.no_grad():
        logits[..., 64] = 10
        logits[..., 70] = 10
    decoded = decode(logits)
    assert (decoded[..., 0] == 64).all() and (decoded[..., 1] == 0).all()
