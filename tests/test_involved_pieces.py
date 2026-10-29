import chess
import numpy as np
import pytest

from data_prep.prepare_involved_pieces import (
    CAPTURED, CASTLING_ROOK, INITIAL_SQUARES, MOVER, label_move, prepare_split,
)


def replay(moves):
    board = chess.Board()
    identities = dict(zip(INITIAL_SQUARES, range(32)))
    labels = []
    for san in moves.split():
        labels.append(label_move(board, identities, board.parse_san(san)))
    return board, identities, labels


def identity(square):
    return INITIAL_SQUARES.index(square)


def test_capture_targets_current_locations_and_preserves_piece_identity():
    board, identities, labels = replay("e4 d5 exd5 Qxd5 Nc3 Qe5+")
    assert labels[2] == [(identity(chess.E2), chess.E4, MOVER),
                         (identity(chess.D7), chess.D5, CAPTURED)]
    assert labels[3] == [(identity(chess.D8), chess.D8, MOVER),
                         (identity(chess.E2), chess.D5, CAPTURED)]
    assert labels[5] == [(identity(chess.D8), chess.D5, MOVER)]
    assert identities[chess.E5] == identity(chess.D8)
    assert set(identities) == set(board.piece_map())


def test_en_passant_labels_captured_pawn_square_not_destination():
    board, identities, labels = replay("e4 a6 e5 d5 exd6")
    assert labels[-1] == [(identity(chess.E2), chess.E5, MOVER),
                          (identity(chess.D7), chess.D5, CAPTURED)]
    assert chess.D5 not in identities
    assert identities[chess.D6] == identity(chess.E2)
    assert set(identities) == set(board.piece_map())
    _, _, labels = replay("a3 e5 a4 e4 d4 exd3")
    assert labels[-1][1] == (identity(chess.D2), chess.D4, CAPTURED)


@pytest.mark.parametrize("fen, san, king, rook, destination", [
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "O-O", chess.E1, chess.H1, chess.F1),
    ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "O-O-O", chess.E8, chess.A8, chess.D8),
])
def test_castling_supervises_both_pre_move_locations(fen, san, king, rook, destination):
    board = chess.Board(fen)
    identities = {sq: identity(sq) for sq in board.piece_map()}
    labels = label_move(board, identities, board.parse_san(san))
    assert labels == [(identity(king), king, MOVER), (identity(rook), rook, CASTLING_ROOK)]
    assert identities[destination] == identity(rook)
    assert set(identities) == set(board.piece_map())


def test_promotion_keeps_pawn_identity_on_later_moves():
    board = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")
    identities = {chess.A7: identity(chess.A2), chess.H1: identity(chess.E1), chess.H8: identity(chess.E8)}
    first = label_move(board, identities, board.parse_san("a8=Q+"))
    assert first == [(identity(chess.A2), chess.A7, MOVER)]
    label_move(board, identities, board.parse_san("Kg7"))
    assert label_move(board, identities, board.parse_san("Qa7+")) == [(identity(chess.A2), chess.A8, MOVER)]


def test_invalid_game_fails_instead_of_supervising_wrong_board(tmp_path):
    path = tmp_path / "bad.bin"
    np.array([0, 2, 2, 1], dtype=np.uint16).tofile(path)
    with pytest.raises(ValueError, match="Invalid SAN"):
        prepare_split(path, tmp_path / "labels.npy", {"stoi": {"<bos>": 0, "<eos>": 1},
                                                      "itos": ["<bos>", "<eos>", "e4"]})
