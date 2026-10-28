from datetime import datetime
import json

from lichess_bot.schedule import TIMEZONE, can_start, challenge_cooldown, finish_trial_game, games_last_day, games_today, limits, record_game, trial_counts


def at(value):
    return datetime.fromisoformat(value).replace(tzinfo=TIMEZONE).timestamp()


def test_default_limits():
    assert limits(at("2026-09-20T12:00:00")) == (5, 7200, 300)


def test_fifth_game_blocks_more_starts_after_restart():
    now = at("2026-09-19T12:00:00")
    state = {"games": {str(i): now - 8000 - i for i in range(4)}}
    assert can_start(state, now)
    record_game(state, "last", now)
    reloaded = json.loads(json.dumps(state))
    assert games_today(reloaded, now + 7300) == 5
    assert not can_start(reloaded, now + 7300)


def test_reconnected_game_is_counted_once_and_keeps_original_day():
    yesterday = at("2026-09-18T23:59:00")
    now = at("2026-09-19T00:01:00")
    state = {"games": {}}
    record_game(state, "ongoing", yesterday)
    record_game(state, "ongoing", now)
    assert state["games"]["ongoing"] == yesterday
    assert games_today(state, now) == 0
    record_game(state, "new", now)
    record_game(state, "new", now + 10)
    assert games_today(state, now) == 1


def test_one_minute_after_game_finish_even_for_a_long_game():
    now = at("2026-09-19T12:00:00")
    state = {"games": {"long": now - 8000}, "last_game_finish": now}
    assert not can_start(state, now + 59)
    assert can_start(state, now + 60)


def test_normal_two_hour_gap_and_five_game_cap_resume():
    now = at("2026-09-20T12:00:00")
    state = {"games": {"first": now - 7199}}
    assert not can_start(state, now)
    assert can_start(state, now + 1)
    state = {"games": {str(i): now - 8000 - i for i in range(5)}}
    assert not can_start(state, now)


def test_yesterday_does_not_consume_todays_budget():
    yesterday = at("2026-09-19T20:00:00")
    now = at("2026-09-20T06:00:00")
    state = {"games": {str(i): yesterday - i for i in range(50)}}
    assert games_today(state, now) == 0
    assert can_start(state, now)


def test_opponent_limit_skips_only_that_opponent_until_server_deadline():
    cause = {"error": "ExampleBot played 100 games", "ratelimit": {"seconds": 15000}}
    assert challenge_cooldown(cause, "ExampleBot", 1000) == ("examplebot", 16000)


def test_our_or_unknown_limit_blocks_all_challenges():
    cause = {"error": "nanoDanya played 100 games", "ratelimit": {"seconds": 5000}}
    assert challenge_cooldown(cause, "ExampleBot", 1000) == ("*", 6000)
    assert challenge_cooldown({}, "ExampleBot", 1000) == ("*", 1060)


def trial_state(now):
    return {"games": {}, "trial": {"started_at": now - 1000, "target": 100,
                                   "max_concurrent": 10, "games": {}}}


def test_trial_reserves_all_ten_slots_and_releases_finished_game():
    now = at("2026-09-22T12:00:00")
    state = trial_state(now)
    for index in range(10):
        record_game(state, str(index), now - 100)
    assert trial_counts(state) == (0, 10)
    assert not can_start(state, now)
    finish_trial_game(state, "0", {"status": "mate", "winner": "white"})
    state["last_game_finish"] = now
    assert trial_counts(state) == (1, 9)
    assert can_start(state, now)


def test_trial_counts_completed_plus_ongoing_to_prevent_overshoot():
    now = at("2026-09-22T12:00:00")
    state = trial_state(now)
    for index in range(100):
        record_game(state, str(index), now - 100)
        if index < 99:
            finish_trial_game(state, str(index), {"status": "resign"})
    assert trial_counts(state) == (99, 1)
    assert not can_start(state, now)
    finish_trial_game(state, "99", {"status": "draw"})
    reloaded = json.loads(json.dumps(state))
    assert trial_counts(reloaded) == (100, 0)
    assert not can_start(reloaded, now + 86400 * 3)


def test_aborts_do_not_consume_trial_budget_and_exports_reconcile():
    now = at("2026-09-22T12:00:00")
    state = trial_state(now)
    for index in range(100):
        record_game(state, str(index), now - 100)
        finish_trial_game(state, str(index), {"status": "mate" if index < 99 else "aborted"})
    assert trial_counts(state) == (99, 0)
    assert can_start(state, now)
    record_game(state, "next", now)
    assert not can_start(state, now + 10)
    finish_trial_game(state, "next", {"status": "outoftime", "winner": "black"})
    record_game(state, "next", now + 60)
    assert trial_counts(state) == (100, 0)


def test_pretrial_game_does_not_enter_trial_on_reconnection():
    now = at("2026-09-22T12:00:00")
    state = trial_state(now)
    record_game(state, "old", now - 2000)
    record_game(state, "old", now)
    assert trial_counts(state) == (0, 0)
    assert limits(now, state) == (100, 10, 10)


def test_bullet_and_rapid_budgets_are_independent():
    now = at("2026-09-22T12:00:00")
    state = trial_state(now)
    state["bullet_trial"] = {"started_at": now - 1000, "target": 100,
                             "max_concurrent": 1, "games": {}}
    record_game(state, "rapid", now - 100)
    record_game(state, "bullet", now - 100, "bullet")
    assert trial_counts(state) == (0, 1)
    assert trial_counts(state, "bullet") == (0, 1)
    assert can_start(state, now)
    assert not can_start(state, now, "bullet")
    finish_trial_game(state, "bullet", {"status": "resign", "winner": "white"})
    assert trial_counts(state) == (0, 1)
    assert trial_counts(state, "bullet") == (1, 0)
    assert can_start(state, now, "bullet")


def test_bullet_cap_survives_restart_without_stopping_rapid():
    now = at("2026-09-22T12:00:00")
    state = trial_state(now)
    state["bullet_trial"] = {"started_at": now - 1000, "target": 100,
                             "max_concurrent": 1, "games": {}}
    for index in range(100):
        record_game(state, str(index), now - 100, "bullet")
        finish_trial_game(state, str(index), {"status": "mate"})
    state = json.loads(json.dumps(state))
    assert trial_counts(state) == (0, 0)
    assert not can_start(state, now + 86400, "bullet")
    assert can_start(state, now + 86400)


def test_unconfigured_bullet_does_not_start():
    now = at("2026-09-22T12:00:00")
    assert not can_start(trial_state(now), now, "bullet")
    assert not can_start({"games": {}}, now, "bullet")


def daily_state():
    return {"games": {}, "daily": {"max_games": 100, "gap_seconds": 60, "max_concurrent": 5}}


def test_daily_limit_survives_restart_and_does_not_reset_at_midnight():
    now = at("2026-10-19T23:59:00")
    state = daily_state()
    for i in range(100):
        record_game(state, str(i), now - 60 - i)
        finish_trial_game(state, str(i), {"status": "aborted"})
    state = json.loads(json.dumps(state))
    assert games_today(state, now + 120) == 0
    assert games_last_day(state, now + 120) == 100
    assert not can_start(state, now + 120)
    assert can_start(state, now + 86400)


def test_daily_mode_enforces_rapid_spacing_and_global_cooldowns():
    now = at("2026-10-19T12:00:00")
    state = daily_state()
    assert limits(now, state) == (100, 60, 10)
    assert can_start(state, now)
    assert not can_start(state, now, "bullet")
    assert not can_start(state, now, "classical")
    record_game(state, "game", now)
    assert not can_start(state, now + 59)
    assert can_start(state, now + 60)
    state["challenge_cooldowns"] = {"*": now + 5000}
    assert not can_start(state, now + 60)
    assert can_start(state, now + 5000)


def test_evaluation_tracks_only_new_model_games_once():
    now = at("2026-10-19T12:00:00")
    state = daily_state()
    state["evaluation"] = {"started_at": now, "games": {}}
    record_game(state, "old", now - 1)
    record_game(state, "old", now + 1)
    record_game(state, "new", now + 2)
    record_game(state, "new", now + 3)
    finish_trial_game(state, "new", {"status": "mate", "winner": "white"})
    assert state["evaluation"]["games"] == {"new": {"started_at": now + 2, "status": "mate", "winner": "white", "speed": "rapid"}}
