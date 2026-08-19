import os

import modal


app = modal.App("nanodanya-benchmark")

# A100 requires a payment method on file (credits alone no longer unlock it).
# Set MODAL_GPU (e.g. A10G, L4, L40S) to run on a credit-eligible GPU.
GPU = os.environ.get("MODAL_GPU", "A100")

ignore = modal.FilePatternMatcher.from_file(".modalignore")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ca-certificates", "curl")
    .run_commands(
        "curl -L --fail -o /usr/local/bin/fairy-stockfish "
        "https://github.com/fairy-stockfish/Fairy-Stockfish/releases/latest/download/fairy-stockfish-largeboard_x86-64",
        "chmod +x /usr/local/bin/fairy-stockfish",
    )
    .pip_install(
        "numpy",
        "torch==2.5.1+cu124",
        "python-chess",
        "requests",
        "matplotlib",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_dir(".", "/root/project", ignore=ignore)
)

volume = modal.Volume.from_name("nanodanya-data")


@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60 * 2,
    volumes={"/data": volume},
)
def bench(subcommand: str, flags: dict):
    import base64
    import json
    import os
    import subprocess
    import time
    from pathlib import Path

    project_root = Path("/root/project")
    output = Path("/tmp/bench.jsonl")
    plot = Path("/tmp/bench.png")

    cmd = ["python", "benchmark/run.py", subcommand, "--output", str(output)]
    if subcommand == "puzzles":
        cmd += ["--plot", str(plot)]
    for key, value in flags.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd += [flag, str(value)]

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"{project_root}:{project_root / 'nanochat'}"
    started = time.time()
    subprocess.run(cmd, check=True, cwd=project_root, env=env)

    records = [json.loads(line) for line in output.read_text().splitlines()]
    return {
        "records": records,
        "elapsed": time.time() - started,
        "png_b64": base64.b64encode(plot.read_bytes()).decode() if plot.exists() else None,
    }


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


def _combine_games(model_a: str, model_b: str, results: list[dict], wall_time: float) -> tuple[dict, list[dict]]:
    outcomes = {"win": 0, "draw": 0, "loss": 0}
    terminations = {}
    game_records = []
    for result in results:
        shard_summary = next(r for r in result["records"] if r["type"] == "games_summary")
        for key, value in shard_summary["outcomes"].items():
            outcomes[key] = outcomes.get(key, 0) + value
        for key, value in shard_summary["terminations"].items():
            terminations[key] = terminations.get(key, 0) + value
        game_records.extend(r for r in result["records"] if r["type"] == "game")

    games = sum(outcomes.values())
    a_score = outcomes["win"] + 0.5 * outcomes["draw"]
    b_score = outcomes["loss"] + 0.5 * outcomes["draw"]
    avg_plies = (
        sum(record["num_plies"] for record in game_records) / len(game_records)
        if game_records
        else None
    )
    worker_time = sum(result["elapsed"] for result in results)

    print(f"\nModel A ({model_a}): {outcomes['win']}W {outcomes['draw']}D {outcomes['loss']}L  score={a_score}/{games}")
    print(f"Model B ({model_b}): {outcomes['loss']}W {outcomes['draw']}D {outcomes['win']}L  score={b_score}/{games}")
    print(f"Wall time: {wall_time:.1f}s ({wall_time / games:.2f}s/game)")
    print(f"Worker time: {worker_time:.1f}s ({worker_time / games:.2f}s/game)")
    if avg_plies is not None:
        print(f"Avg plies: {avg_plies:.1f}")
    print(f"Terminations: {dict(sorted(terminations.items()))}")

    summary = {
        "type": "games_summary",
        "schema_version": 1,
        "games": games,
        "score": a_score / games if games else None,
        "model_a": model_a,
        "model_b": model_b,
        "model_a_score": a_score,
        "model_b_score": b_score,
        "outcomes": outcomes,
        "terminations": dict(sorted(terminations.items())),
        "avg_num_plies": avg_plies,
        "wall_time": wall_time,
        "worker_time": worker_time,
    }
    game_records.sort(key=lambda record: record.get("game_num", 0))
    return summary, game_records


@app.local_entrypoint()
def main(
    mode: str = "h2h",
    model: str = "",
    model_a: str = "plain/games-5m",
    model_b: str = "plain/puzzles-5m",
    data_dir: str = "",
    split: str = "val",
    max_positions: int = 4096,
    max_ply: int = 140,
    max_games: int = 0,
    allow_eos: bool = False,
    top_illegal: int = 20,
    games: int = 50,
    temperature: float = 0.8,
    batch_size: int = 0,
    max_plies: int = 200,
    shards: int = 1,
    kv_cache: bool = False,
    seed: int = 0,
    stockfish_elo: int = 500,
    stockfish: str = "/usr/local/bin/fairy-stockfish",
    stockfish_time: float = 0.02,
    stockfish_depth: int = 0,
    stockfish_nodes: int = 0,
    rating_min: int = 600,
    rating_max: int = 2800,
    bin_width: int = 100,
    per_bin: int = 400,
    scan_cap: int = 0,
    write_positions: bool = False,
    output: str = "",
):
    import base64
    import json
    import time
    from pathlib import Path

    def out_path(prefix: str) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = Path(output) if output else Path(f"benchmark/results/modal_{prefix}_{stamp}.jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_records(path: Path, records: list[dict]) -> None:
        with path.open("w") as f:
            for record in records:
                f.write(json.dumps(record, sort_keys=True) + "\n")

    def print_summaries(records: list[dict]) -> None:
        for record in records:
            if record.get("type", "").endswith("_summary"):
                print(json.dumps(record, indent=2, sort_keys=True))

    if mode == "legality":
        result = bench.remote("legality", {
            "model": model or "plain/games-5m",
            "data_dir": data_dir or "/data/datasets/games/5m",
            "split": split,
            "max_positions": max_positions,
            "batch_size": batch_size or 128,
            "max_ply": max_ply,
            "max_games": max_games,
            "allow_eos": allow_eos,
            "top_illegal": top_illegal,
        })
        path = out_path("legality")
        write_records(path, result["records"])
        print_summaries(result["records"])
        print(f"wrote {path}")
        return

    if mode == "move-quality":
        result = bench.remote("move-quality", {
            "model": model or "plain/games-5m",
            "data_dir": data_dir or "/data/datasets/eval/base",
            "split": split,
            "max_positions": max_positions,
            "max_ply": max_ply,
            "max_games": max_games,
            "batch_size": batch_size or 64,
            "stockfish": stockfish,
            "stockfish_time": stockfish_time,
        })
        path = out_path("move_quality")
        write_records(path, result["records"])
        print_summaries(result["records"])
        print(f"wrote {path}")
        return

    if mode == "puzzles":
        result = bench.remote("puzzles", {
            "model": model or "/data/checkpoints/plain/games-15m/l16_best.pt",
            "ndjson": "/data/puzzles_raw/puzzle_games_ndjson.txt",
            "metadata": "/data/puzzles_raw/puzzle_metadata.txt",
            "rating_min": rating_min,
            "rating_max": rating_max,
            "bin_width": bin_width,
            "per_bin": per_bin,
            "scan_cap": scan_cap,
            "batch_size": batch_size or 256,
            "write_positions": write_positions,
        })
        path = out_path("puzzles")
        write_records(path, result["records"])
        if result["png_b64"]:
            png_path = path.with_suffix(".png")
            png_path.write_bytes(base64.b64decode(result["png_b64"]))
            print(f"wrote plot {png_path}")
        print_summaries(result["records"])
        print(f"wrote {path}")
        return

    if mode not in ("h2h", "stockfish"):
        raise ValueError(f"unsupported mode: {mode}")

    if mode == "h2h":
        label_a, label_b = model_a, model_b
        base = {
            "game_mode": "h2h",
            "model": model_a,
            "opponent_model": model_b,
            "temperature": temperature,
            "batch_size": batch_size or 64,
            "max_plies": max_plies,
            "kv_cache": kv_cache,
        }
    else:
        label_a, label_b = model or "plain/games-5m", f"sf{stockfish_elo}"
        base = {
            "game_mode": "stockfish",
            "model": label_a,
            "stockfish": stockfish,
            "stockfish_elo": stockfish_elo,
            "stockfish_time": stockfish_time,
            "stockfish_depth": stockfish_depth,
            "stockfish_nodes": stockfish_nodes,
            "temperature": temperature,
            "batch_size": batch_size or 64,
            "max_plies": max_plies,
        }

    splits = _split_games(games, shards)
    print(
        f"Running {games} games vs {label_b} with {len(splits)} shard(s), "
        f"batch_size={base['batch_size']}, max_plies={max_plies}, temp={temperature}, seed={seed}"
    )
    started = time.time()
    jobs = [
        ("games", base | {"games": count, "game_offset": offset, "seed": seed + offset})
        for offset, count in splits
    ]
    results = list(bench.starmap(jobs))
    wall_time = time.time() - started

    summary, game_records = _combine_games(label_a, label_b, results, wall_time)
    if mode == "stockfish":
        summary |= {key: base[key] for key in ("stockfish_elo", "stockfish", "stockfish_time", "stockfish_depth", "stockfish_nodes")}

    run_record = {
        "type": "run",
        "schema_version": 1,
        "command": f"modal_{mode}",
        "config": base | {"games": games, "shards": shards, "seed": seed},
    }
    path = out_path(mode)
    write_records(path, [run_record, *game_records, summary])
    print(f"wrote full game records to {path}")
