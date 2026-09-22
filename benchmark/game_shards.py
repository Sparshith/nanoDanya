"""Persist completed game batches so interrupted collections can resume."""

import json
from pathlib import Path


def validate_shard(result, request):
    if result["request"] != request:
        raise ValueError("Cached shard uses a different benchmark request")
    flags = request["flags"]
    games = [row for row in result["records"] if row["type"] == "game"]
    expected = set(range(flags["game_offset"], flags["game_offset"] + flags["games"]))
    if len(games) != len(expected) or {row["game_num"] for row in games} != expected:
        raise ValueError("Incomplete or duplicate games in shard")


def load_shards(directory, jobs):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    cached, pending = [], []
    for subcommand, flags in jobs:
        path = directory / f"{flags['game_offset']:06d}.json"
        if not path.exists():
            pending.append((subcommand, flags))
            continue
        result = json.loads(path.read_text())
        validate_shard(result, {"subcommand": subcommand, "flags": flags})
        cached.append(result)
    return cached, pending


def save_shard(directory, result):
    validate_shard(result, result["request"])
    offset = result["request"]["flags"]["game_offset"]
    path = Path(directory) / f"{offset:06d}.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result) + "\n")
    temporary.replace(path)
