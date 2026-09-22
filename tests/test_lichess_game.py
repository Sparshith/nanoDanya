from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lichess_bot.game import play_game


def full(moves="", status="started", white="nanoDanya"):
    return {"type": "gameFull", "white": {"name": white},
            "state": {"moves": moves, "status": status}}


def state(moves="", status="started"):
    return {"type": "gameState", "moves": moves, "status": status}


def run(monkeypatch, streams, submissions, choices):
    closed = []

    def stream_game(game_id):
        assert len(closed) == bots.stream_game_state.call_count - 1
        events = streams.pop(0)
        try:
            yield from events
        finally:
            closed.append(game_id)

    bots = SimpleNamespace(stream_game_state=Mock(side_effect=stream_game),
                           make_move=Mock(side_effect=submissions))
    choose = Mock(side_effect=choices)
    sleep = Mock()
    monkeypatch.setattr("lichess_bot.game.time.sleep", sleep)
    play_game(SimpleNamespace(bots=bots), "game", "nanoDanya", choose)
    assert len(closed) == bots.stream_game_state.call_count
    return bots, choose, sleep


def test_failed_submission_refreshes_and_retries_same_move(monkeypatch):
    bots, choose, sleep = run(
        monkeypatch,
        [[full()], [full(), state("e2e4"), state("e2e4", "resign")]],
        [ConnectionError("response disconnected"), None], ["e4"],
    )
    assert [call.args for call in bots.make_move.call_args_list] == [("game", "e2e4")] * 2
    choose.assert_called_once_with([])
    sleep.assert_called_once_with(5)


@pytest.mark.parametrize("opponent_moved", [False, True])
def test_lost_response_to_accepted_move_never_replays_stale_move(monkeypatch, opponent_moved):
    moves = "e2e4 e7e5" if opponent_moved else "e2e4"
    bots, choose, _ = run(
        monkeypatch, [[full()], [full(moves), state(moves, "resign")]],
        [ConnectionError("response lost"), None], ["e4", "Nf3"],
    )
    expected = [("game", "e2e4")]
    if opponent_moved:
        expected.append(("game", "g1f3"))
        assert choose.call_args.args == (["e4", "e5"],)
    assert [call.args for call in bots.make_move.call_args_list] == expected


def test_terminal_snapshot_after_failure_stops_retrying(monkeypatch):
    bots, _, _ = run(monkeypatch, [[full()], [full(status="outoftime")]],
                     [ConnectionError("lost")], ["e4"])
    assert bots.make_move.call_count == 1


def test_terminal_result_is_returned_for_trial_accounting():
    event = full(status="resign")
    event["state"]["winner"] = "white"
    def stream(game_id):
        yield event
    client = SimpleNamespace(bots=SimpleNamespace(stream_game_state=stream))
    assert play_game(client, "game", "nanoDanya", Mock()) == event["state"]


def test_duplicate_state_does_not_submit_twice_and_black_waits(monkeypatch):
    bots, choose, _ = run(
        monkeypatch,
        [[full(white="opponent"), state("e2e4"), state("e2e4"),
          state("e2e4 e7e5", "resign")]], [None], ["e5"],
    )
    bots.make_move.assert_called_once_with("game", "e7e5")
    choose.assert_called_once_with(["e4"])


def test_closed_stream_reconnects_to_latest_position(monkeypatch):
    bots, choose, sleep = run(
        monkeypatch, [[full("e2e4")], [full("e2e4 e7e5"), state(status="resign")]],
        [None], ["Nf3"],
    )
    bots.make_move.assert_called_once_with("game", "g1f3")
    choose.assert_called_once_with(["e4", "e5"])
    sleep.assert_called_once_with(5)


def test_rate_limit_backs_off_before_refreshing(monkeypatch):
    class RateLimited(Exception):
        status_code = 429

    bots, _, sleep = run(
        monkeypatch, [[full()], [full(), state(status="resign")]],
        [RateLimited(), None], ["e4"],
    )
    assert bots.make_move.call_count == 2
    sleep.assert_called_once_with(65)
