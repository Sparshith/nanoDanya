from datetime import datetime
from zoneinfo import ZoneInfo


TIMEZONE = ZoneInfo("Asia/Kolkata")


def day_key(now):
    return datetime.fromtimestamp(now, TIMEZONE).date().isoformat()


def limits(now, state=None):
    if state and state.get("daily"):
        return state["daily"]["max_games"], state["daily"]["gap_seconds"], 10
    if state and (state.get("trial") or state.get("bullet_trial")):
        return (state.get("trial") or state["bullet_trial"])["target"], 10, 10
    return 5, 7200, 300


def games_today(state, now):
    return sum(day_key(start) == day_key(now) for start in state["games"].values())


def games_last_day(state, now):
    return sum(start > now - 86400 for start in state["games"].values())


def trial_key(speed):
    return "trial" if speed == "rapid" else "bullet_trial" if speed == "bullet" else None


def record_game(state, game_id, started_at, speed="rapid"):
    started_at = state["games"].setdefault(game_id, started_at)
    trial = state.get(trial_key(speed))
    if trial and started_at >= trial["started_at"]:
        trial["games"].setdefault(game_id, {"started_at": started_at, "status": "started"})
    evaluation = state.get("evaluation")
    if evaluation and started_at >= evaluation["started_at"]:
        evaluation["games"].setdefault(game_id, {"started_at": started_at, "status": "started", "speed": speed})


def finish_trial_game(state, game_id, result):
    for key in ("trial", "bullet_trial", "evaluation"):
        trial = state.get(key)
        if trial and game_id in trial["games"]:
            trial["games"][game_id].update({key: result[key] for key in ("status", "winner") if key in result})


def trial_counts(state, speed="rapid"):
    games = state[trial_key(speed)]["games"].values()
    completed = pending = 0
    for game in games:
        if game["status"] in ("created", "started"):
            pending += 1
        elif game["status"] not in ("aborted", "noStart"):
            completed += 1
    return completed, pending


def can_start(state, now, speed="rapid"):
    cap, gap, _ = limits(now, state)
    if now < state.get("challenge_cooldowns", {}).get("*", 0):
        return False
    if state.get("daily"):
        return (speed == "rapid" and games_last_day(state, now) < cap
                and now - max(state["games"].values(), default=0) >= gap)
    if state.get("trial") or state.get("bullet_trial"):
        trial = state.get(trial_key(speed))
        if not trial:
            return False
        completed, pending = trial_counts(state, speed)
        return (completed + pending < trial["target"]
                and pending < trial["max_concurrent"]
                and now - max(state["games"].values(), default=0) >= gap)
    if speed == "bullet":
        return False
    count = games_today(state, now)
    return (count < cap
            and now - max(state["games"].values(), default=0) >= gap
            and now - state.get("last_game_finish", 0) >= 60)


def challenge_cooldown(cause, opponent, now):
    seconds = (cause.get("ratelimit") or {}).get("seconds", 60)
    owner = opponent.lower() if cause.get("error", "").lower().startswith(opponent.lower() + " ") else "*"
    return owner, now + max(60, seconds)
