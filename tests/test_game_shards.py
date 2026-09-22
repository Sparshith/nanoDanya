import copy

import pytest

from benchmark.game_shards import load_shards, save_shard


def test_resume_only_runs_missing_shards_and_rejects_changed_seed(tmp_path):
    jobs = [("games", {"game_offset": offset, "games": 2, "seed": 100 + offset}) for offset in (0, 2)]
    cached, pending = load_shards(tmp_path, jobs)
    assert cached == [] and pending == jobs
    result = {"request": {"subcommand": "games", "flags": jobs[0][1]},
              "records": [{"type": "game", "game_num": i} for i in (0, 1)]}
    save_shard(tmp_path, result)
    cached, pending = load_shards(tmp_path, jobs)
    assert cached == [result] and pending == jobs[1:]
    changed = copy.deepcopy(jobs)
    changed[0][1]["seed"] += 1
    with pytest.raises(ValueError, match="different benchmark request"):
        load_shards(tmp_path, changed)


def test_duplicate_or_missing_games_are_never_saved(tmp_path):
    result = {"request": {"subcommand": "games", "flags": {"game_offset": 0, "games": 2}},
              "records": [{"type": "game", "game_num": 0}] * 2}
    with pytest.raises(ValueError, match="Incomplete or duplicate"):
        save_shard(tmp_path, result)
    assert list(tmp_path.iterdir()) == []
