import modal


app = modal.App("nanodanya-benchmark")

ignore = modal.FilePatternMatcher.from_file(".modalignore")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy",
        "torch==2.5.1+cu124",
        "tokenizers",
        "tiktoken",
        "python-chess",
        "pyarrow",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_dir(".", "/root/project", ignore=ignore)
)

volume = modal.Volume.from_name("nanodanya-data")


def _run_batched_h2h(
    model_a: str,
    model_b: str,
    games: int = 50,
    temperature: float = 0.8,
    batch_size: int = 64,
    max_plies: int = 200,
    game_offset: int = 0,
    kv_cache: bool = False,
):
    import json
    import os
    import subprocess
    import time
    from pathlib import Path

    project_root = Path("/root/project")
    output = Path("/tmp/h2h_games.jsonl")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"{project_root}:{project_root / 'nanochat'}"

    started = time.time()
    cmd = [
        "python",
        "benchmark/run.py",
        "games",
        "--game-mode",
        "h2h",
        "--model",
        model_a,
        "--opponent-model",
        model_b,
        "--games",
        str(games),
        "--game-offset",
        str(game_offset),
        "--temperature",
        str(temperature),
        "--batch-size",
        str(batch_size),
        "--max-plies",
        str(max_plies),
        "--output",
        str(output),
    ]
    if kv_cache:
        cmd.append("--kv-cache")

    subprocess.run(
        cmd,
        check=True,
        cwd=project_root,
        env=env,
    )
    elapsed = time.time() - started
    summary = None
    games_seen = []
    with output.open() as f:
        for line in f:
            record = json.loads(line)
            if record.get("type") == "games_summary":
                summary = record
            elif record.get("type") == "game":
                games_seen.append(record)
    if summary is None:
        raise RuntimeError("benchmark did not write a games_summary record")
    summary["elapsed"] = elapsed
    summary["game_offset"] = game_offset
    summary["kv_cache"] = kv_cache
    summary["model_a"] = model_a
    summary["model_b"] = model_b
    summary["game_records"] = games_seen
    return summary


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 2,
    volumes={"/data": volume},
)
def benchmark_h2h(
    model_a: str,
    model_b: str,
    games: int = 50,
    temperature: float = 0.8,
    batch_size: int = 64,
    max_plies: int = 200,
    game_offset: int = 0,
    kv_cache: bool = False,
):
    return _run_batched_h2h(
        model_a=model_a,
        model_b=model_b,
        games=games,
        temperature=temperature,
        batch_size=batch_size,
        max_plies=max_plies,
        game_offset=game_offset,
        kv_cache=kv_cache,
    )


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 2,
    volumes={"/data": volume},
)
def benchmark_h2h_shard(spec: dict):
    return _run_batched_h2h(**spec)


def _split_games(total_games: int, shards: int) -> list[tuple[int, int]]:
    shards = max(1, min(shards, total_games))
    base = total_games // shards
    extra = total_games % shards
    offsets = []
    cursor = 0
    for shard_index in range(shards):
        count = base + (1 if shard_index < extra else 0)
        offsets.append((cursor, count))
        cursor += count
    return offsets


def _print_combined_summary(model_a: str, model_b: str, summaries: list[dict], wall_time: float) -> None:
    outcomes = {"win": 0, "draw": 0, "loss": 0}
    terminations = {}
    game_records = []
    for summary in summaries:
        for key, value in summary["outcomes"].items():
            outcomes[key] = outcomes.get(key, 0) + value
        for key, value in summary["terminations"].items():
            terminations[key] = terminations.get(key, 0) + value
        game_records.extend(summary.get("game_records", []))

    games = sum(outcomes.values())
    a_wins = outcomes.get("win", 0)
    draws = outcomes.get("draw", 0)
    b_wins = outcomes.get("loss", 0)
    a_score = a_wins + 0.5 * draws
    b_score = b_wins + 0.5 * draws
    avg_plies = (
        sum(record["num_plies"] for record in game_records) / len(game_records)
        if game_records
        else None
    )
    worker_time = sum(summary["elapsed"] for summary in summaries)

    print(f"\nModel A ({model_a}): {a_wins}W {draws}D {b_wins}L  score={a_score}/{games}")
    print(f"Model B ({model_b}): {b_wins}W {draws}D {a_wins}L  score={b_score}/{games}")
    print(f"Wall time: {wall_time:.1f}s ({wall_time / games:.2f}s/game)")
    print(f"Worker time: {worker_time:.1f}s ({worker_time / games:.2f}s/game)")
    if avg_plies is not None:
        print(f"Avg plies: {avg_plies:.1f}")
    print(f"Terminations: {dict(sorted(terminations.items()))}")


@app.local_entrypoint()
def main(
    model_a: str = "/data/actual_3m/chess_actual_3m_uniform_L12_H6_E768.pt",
    model_b: str = "puzzle-plain/reference",
    games: int = 50,
    temperature: float = 0.8,
    batch_size: int = 64,
    max_plies: int = 200,
    shards: int = 1,
    kv_cache: bool = False,
):
    import time

    started = time.time()
    splits = _split_games(games, shards)
    print(
        f"Running {games} H2H games with {len(splits)} shard(s), "
        f"batch_size={batch_size}, max_plies={max_plies}, temp={temperature}, "
        f"kv_cache={kv_cache}"
    )
    if len(splits) == 1:
        summaries = [
            benchmark_h2h.remote(
                model_a,
                model_b,
                games,
                temperature,
                batch_size,
                max_plies,
                0,
                kv_cache,
            )
        ]
    else:
        jobs = [
            {
                "model_a": model_a,
                "model_b": model_b,
                "games": count,
                "temperature": temperature,
                "batch_size": batch_size,
                "max_plies": max_plies,
                "game_offset": offset,
                "kv_cache": kv_cache,
            }
            for offset, count in splits
        ]
        summaries = list(benchmark_h2h_shard.map(jobs))
    _print_combined_summary(model_a, model_b, summaries, time.time() - started)
