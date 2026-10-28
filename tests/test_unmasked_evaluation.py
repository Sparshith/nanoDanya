from types import SimpleNamespace

import chess
import pytest
import torch

from benchmark.run import PuzzleState, build_parser, command_elo, command_tournament, solve_puzzles
from chess_inference import parse_san_prediction


@pytest.mark.parametrize("token", ["e5", "<bos>", "<eos>", "<unk>", "--", "e2e4", "Ng1f3", "bad"])
def test_invalid_top_token_cannot_fall_back_to_a_legal_move(token):
    assert parse_san_prediction(chess.Board(), token) is None


def test_checkmate_suffix_is_optional():
    board = chess.Board()
    for san in "f3 e5 g4".split():
        board.push_san(san)
    assert parse_san_prediction(board, "Qh4") == board.parse_san("Qh4#")


def test_illegal_puzzle_prediction_fails_instead_of_crashing_or_falling_back():
    tokens = ["<bos>", "<eos>", "e4"]
    stoi = {token: i for i, token in enumerate(tokens)}
    model = SimpleNamespace(last_logits=lambda prefix: torch.tensor([0.0, 10.0, 9.0]))
    for mask, solved in [(False, False), (True, True)]:
        state = PuzzleState(1000, 1000, "p", "g", [0], chess.Board(), ["unused", "e2e4"])
        solve_puzzles([state], model, stoi, tokens, device="cpu", batch_size=1, max_len=64, legal_mask=mask)
        assert state.solved == solved
        assert state.first_correct == solved
        assert not state.alive


def test_tournament_forfeits_both_colors_and_records_protocol(monkeypatch, tmp_path):
    import json
    import benchmark.run as bench

    tokens = ["<bos>", "<eos>", "e4"]
    stoi = {token: i for i, token in enumerate(tokens)}
    model = SimpleNamespace(last_logits=lambda prefix: torch.tensor([0.0, 10.0, 9.0]))
    config = SimpleNamespace(sequence_len=64)
    judge = SimpleNamespace(
        analyse=lambda board, limit: {"score": chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE)},
        quit=lambda: None,
    )
    monkeypatch.setattr(bench, "load_tournament_model", lambda ref, device: (model, config, stoi, tokens))
    monkeypatch.setattr(bench, "load_openings", lambda path: [("A00", []), ("A00", ["e4"])])
    monkeypatch.setattr(chess.engine.SimpleEngine, "popen_uci", lambda *args, **kwargs: judge)
    path = tmp_path / "games.jsonl"
    args = build_parser().parse_args(["tournament", "--players", "a=a", "b=b", "--openings", "2",
                                     "--no-legal-mask", "--device", "cpu", "--output", str(path)])
    command_tournament(args)
    games = [json.loads(line) for line in path.read_text().splitlines() if json.loads(line)["type"] == "game"]
    assert [g["result"] for g in games] == ["0-1", "0-1", "1-0", "1-0"]
    assert all(g["termination"].endswith("_illegal") and not g["legal_mask"] for g in games)
    assert [g["num_plies"] for g in games] == [0, 0, 1, 1]


def test_elo_rejects_mixing_old_masked_games_with_unmasked_games(tmp_path):
    import json

    rows = [{"type": "game", "result": "1-0"}, {"type": "game", "result": "0-1", "legal_mask": False}]
    (tmp_path / "games.jsonl").write_text("\n".join(map(json.dumps, rows)))
    with pytest.raises(ValueError, match="separate Elo pools"):
        command_elo(SimpleNamespace(games_dir=str(tmp_path)))
