import json
from argparse import Namespace

import chess
import chess.engine

from benchmark.run import (
    TournamentGame,
    check_tournament_game,
    command_elo,
    load_openings,
    push_tournament_move,
)


class Judge:
    def __init__(self, cp):
        self.cp = cp

    def analyse(self, board, limit):
        return {"score": chess.engine.PovScore(chess.engine.Cp(self.cp), chess.WHITE)}


def game(**prefixes):
    return TournamentGame(0, "a", "b", "C00", [], chess.Board(), prefixes)


def test_openings_are_legal():
    openings = load_openings("benchmark/openings.tsv")
    assert len(openings) == 200
    for _, moves in openings:
        board = chess.Board()
        for san in moves:
            board.push_san(san)


def test_moves_extend_model_prefixes_and_oov_is_unscored():
    models = {"a": (None, None, {"<bos>": 0, "e4": 1}, None)}
    g = game(a=[0])
    push_tournament_move(g, chess.Move.from_uci("e2e4"), models)
    assert g.prefixes["a"] == [0, 1] and not g.finished
    push_tournament_move(g, chess.Move.from_uci("c7c5"), models)
    assert (g.finished, g.result, g.termination) == (True, "*", "a_oov")


def test_adjudication_awards_the_side_ahead():
    args = Namespace(adjudicate_time=0.01, adjudicate_cp=1300, max_plies=400)
    g = game()
    check_tournament_game(g, Judge(-1500), args, {})
    assert (g.result, g.termination) == ("0-1", "adjudicated")
    g = game()
    check_tournament_game(g, Judge(200), args, {})
    assert not g.finished


def test_elo_parses_bayeselo_table_and_skips_unscored(tmp_path):
    games = tmp_path / "games"
    games.mkdir()
    rows = [
        {"type": "game", "white": "m", "black": "sf1500", "result": "1-0", "pgn": '[Result "1-0"]\n\n1-0'},
        {"type": "game", "white": "sf1500", "black": "m", "result": "*", "pgn": ""},
    ]
    (games / "t.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    fake = tmp_path / "bayeselo"
    fake.write_text(
        "#!/bin/sh\ncat > /dev/null\n"
        "echo 'Rank Name   Elo    +    - games score oppo. draws'\n"
        "echo '   1 m     1600  120  110     1  100%  1500    0%'\n"
        "echo '   2 sf1500 1500 110  120     1    0%  1600    0%'\n"
    )
    fake.chmod(0o755)
    output = tmp_path / "elo.jsonl"
    command_elo(Namespace(games_dir=str(games), anchor="sf1500", anchor_elo=1500, bayeselo=str(fake), output=str(output)))
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert records[0]["config"]["unscored"] == 1
    assert [(r["player"], r["elo"], r["ci_plus"], r["score"]) for r in records[1:]] == [("m", 1600, 120, 1.0), ("sf1500", 1500, 110, 0.0)]
