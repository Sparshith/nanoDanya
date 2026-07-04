import modal


app = modal.App("nanodanya-benchmark")

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
        "tokenizers",
        "tiktoken",
        "python-chess",
        "pyarrow",
        "matplotlib",
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
    seed: int = 0,
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
        "--seed",
        str(seed),
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
    summary["seed"] = seed
    summary["model_a"] = model_a
    summary["model_b"] = model_b
    summary["game_records"] = games_seen
    return summary


def _run_batched_stockfish(
    model: str,
    stockfish_elo: int = 500,
    games: int = 50,
    temperature: float = 0.8,
    batch_size: int = 64,
    max_plies: int = 200,
    game_offset: int = 0,
    seed: int = 0,
    stockfish: str = "/usr/local/bin/fairy-stockfish",
    stockfish_time: float = 0.02,
    stockfish_depth: int = 0,
    stockfish_nodes: int = 0,
):
    import json
    import os
    import subprocess
    import time
    from pathlib import Path

    project_root = Path("/root/project")
    output = Path("/tmp/stockfish_games.jsonl")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"{project_root}:{project_root / 'nanochat'}"

    started = time.time()
    cmd = [
        "python",
        "benchmark/run.py",
        "games",
        "--game-mode",
        "stockfish",
        "--model",
        model,
        "--stockfish",
        stockfish,
        "--stockfish-elo",
        str(stockfish_elo),
        "--stockfish-time",
        str(stockfish_time),
        "--stockfish-depth",
        str(stockfish_depth),
        "--stockfish-nodes",
        str(stockfish_nodes),
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
        "--seed",
        str(seed),
        "--output",
        str(output),
    ]

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
    summary["seed"] = seed
    summary["model_a"] = model
    summary["model_b"] = f"sf{stockfish_elo}"
    summary["stockfish_elo"] = stockfish_elo
    summary["stockfish"] = stockfish
    summary["stockfish_time"] = stockfish_time
    summary["stockfish_depth"] = stockfish_depth
    summary["stockfish_nodes"] = stockfish_nodes
    summary["game_records"] = games_seen
    return summary


def _run_legality(
    model: str,
    data_dir: str = "/data/datasets/games/5m",
    split: str = "val",
    max_positions: int = 4096,
    batch_size: int = 128,
    max_ply: int = 140,
    max_games: int = 0,
    allow_eos: bool = False,
    top_illegal: int = 20,
    device: str = "cuda",
):
    import os
    import subprocess
    from pathlib import Path

    project_root = Path("/root/project")
    output = Path("/tmp/legality.jsonl")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"{project_root}:{project_root / 'nanochat'}"

    cmd = [
        "python",
        "benchmark/run.py",
        "legality",
        "--model",
        model,
        "--data-dir",
        data_dir,
        "--split",
        split,
        "--max-positions",
        str(max_positions),
        "--batch-size",
        str(batch_size),
        "--max-ply",
        str(max_ply),
        "--max-games",
        str(max_games),
        "--top-illegal",
        str(top_illegal),
        "--device",
        device,
        "--output",
        str(output),
    ]
    if allow_eos:
        cmd.append("--allow-eos")

    subprocess.run(cmd, check=True, cwd=project_root, env=env)
    return output.read_text()


def _run_move_quality(
    model: str,
    data_dir: str = "/data/datasets/eval/base",
    split: str = "val",
    max_positions: int = 600,
    max_ply: int = 140,
    max_games: int = 0,
    batch_size: int = 64,
    stockfish: str = "/usr/local/bin/fairy-stockfish",
    stockfish_time: float = 0.05,
    device: str = "cuda",
):
    import os
    import subprocess
    from pathlib import Path

    project_root = Path("/root/project")
    output = Path("/tmp/move_quality.jsonl")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"{project_root}:{project_root / 'nanochat'}"

    cmd = [
        "python",
        "benchmark/run.py",
        "move-quality",
        "--model",
        model,
        "--data-dir",
        data_dir,
        "--split",
        split,
        "--max-positions",
        str(max_positions),
        "--max-ply",
        str(max_ply),
        "--max-games",
        str(max_games),
        "--batch-size",
        str(batch_size),
        "--stockfish",
        stockfish,
        "--stockfish-time",
        str(stockfish_time),
        "--device",
        device,
        "--output",
        str(output),
    ]

    subprocess.run(cmd, check=True, cwd=project_root, env=env)
    return output.read_text()


def _inspect_legality_failures(
    model: str,
    data_dir: str = "/data/datasets/games/5m",
    split: str = "val",
    phase: str = "opening",
    max_positions: int = 4096,
    batch_size: int = 128,
    max_ply: int = 140,
    max_games: int = 0,
    allow_eos: bool = False,
    device: str = "cuda",
):
    import json
    import sys
    from pathlib import Path

    import chess

    project_root = Path("/root/project")
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    nanochat_root = project_root / "nanochat"
    if str(nanochat_root) not in sys.path:
        sys.path.insert(0, str(nanochat_root))

    from benchmark.run import (
        last_logits_for_prefixes,
        load_model,
        load_token_games,
        phase_for_ply,
        raw_metrics_from_logits,
        resolve_device,
    )
    from chess_token_utils import normalized_legal_sans, resolve_token_id

    device = resolve_device(device)
    loaded_model, _, stoi, itos = load_model(model, device)
    games, _ = load_token_games(Path(data_dir), split, max_games)

    positions = []
    bos_id = stoi["<bos>"]
    for game_index, game in enumerate(games):
        board = chess.Board()
        prefix_ids = [bos_id]
        moves = []
        ply = 0
        for token in game[1:]:
            if token == "<eos>":
                break
            target_id = resolve_token_id(stoi, token)
            if target_id is None:
                break
            ply += 1
            if max_ply <= 0 or ply <= max_ply:
                positions.append(
                    {
                        "board": board.copy(stack=False),
                        "prefix_ids": prefix_ids.copy(),
                        "ply": ply,
                        "game_index": game_index,
                        "moves": moves.copy(),
                    }
                )
                if max_positions > 0 and len(positions) >= max_positions:
                    break
            try:
                board.push_san(token)
            except ValueError:
                break
            prefix_ids.append(target_id)
            moves.append(token)
        if max_positions > 0 and len(positions) >= max_positions:
            break

    failures = []
    prefixes = [pos["prefix_ids"] for pos in positions]
    for idx, logits in last_logits_for_prefixes(
        loaded_model,
        prefixes,
        pad_id=stoi["<eos>"],
        device=device,
        batch_size=batch_size,
    ):
        pos = positions[idx]
        if phase and phase_for_ply(pos["ply"]) != phase:
            continue
        metrics = raw_metrics_from_logits(
            logits,
            pos["board"],
            stoi,
            itos,
            allow_eos=allow_eos,
        )
        if metrics["raw_top1_legal"]:
            continue
        legal_sans = sorted(normalized_legal_sans(pos["board"]))
        failures.append(
            {
                "model": model,
                "data_dir": data_dir,
                "split": split,
                "phase": phase_for_ply(pos["ply"]),
                "ply": pos["ply"],
                "game_index": pos["game_index"],
                "moves": pos["moves"],
                "fen": pos["board"].fen(),
                "raw_top1": metrics["raw_top1"],
                "raw_top1_under_disambiguated": metrics["raw_top1_under_disambiguated"],
                "raw_top1_under_disambiguated_matches": metrics["raw_top1_under_disambiguated_matches"],
                "raw_top1_prob": metrics["raw_top1_prob"],
                "legal_mass": metrics["legal_mass"],
                "legal_sans": legal_sans,
            }
        )

    return json.dumps(
        {
            "type": "legality_failures",
            "model": model,
            "data_dir": data_dir,
            "split": split,
            "phase": phase,
            "max_positions": max_positions,
            "failures": failures,
        },
        sort_keys=True,
    )


def _run_puzzles(
    model: str,
    ndjson: str = "/data/puzzles_raw/puzzle_games_ndjson.txt",
    metadata: str = "/data/puzzles_raw/puzzle_metadata.txt",
    rating_min: int = 600,
    rating_max: int = 2800,
    bin_width: int = 100,
    per_bin: int = 400,
    scan_cap: int = 0,
    batch_size: int = 256,
    write_positions: bool = False,
    device: str = "cuda",
):
    import base64
    import json
    import os
    import subprocess
    from pathlib import Path

    project_root = Path("/root/project")
    output = Path("/tmp/puzzles.jsonl")
    plot = Path("/tmp/puzzles.png")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = f"{project_root}:{project_root / 'nanochat'}"

    cmd = [
        "python",
        "benchmark/run.py",
        "puzzles",
        "--model",
        model,
        "--ndjson",
        ndjson,
        "--metadata",
        metadata,
        "--rating-min",
        str(rating_min),
        "--rating-max",
        str(rating_max),
        "--bin-width",
        str(bin_width),
        "--per-bin",
        str(per_bin),
        "--scan-cap",
        str(scan_cap),
        "--batch-size",
        str(batch_size),
        "--device",
        device,
        "--output",
        str(output),
        "--plot",
        str(plot),
    ]
    if write_positions:
        cmd.append("--write-positions")

    subprocess.run(cmd, check=True, cwd=project_root, env=env)

    summary = None
    positions = []
    with output.open() as f:
        for line in f:
            record = json.loads(line)
            if record.get("type") == "puzzles_summary":
                summary = record
            elif record.get("type") == "puzzle_position":
                positions.append(record)
    if summary is None:
        raise RuntimeError("benchmark did not write a puzzles_summary record")
    png_b64 = base64.b64encode(plot.read_bytes()).decode() if plot.exists() else None
    return {"summary": summary, "positions": positions, "png_b64": png_b64}


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 2,
    volumes={"/data": volume},
)
def benchmark_puzzles(
    model: str = "/data/checkpoints/plain/games-15m/l16_best.pt",
    ndjson: str = "/data/puzzles_raw/puzzle_games_ndjson.txt",
    metadata: str = "/data/puzzles_raw/puzzle_metadata.txt",
    rating_min: int = 600,
    rating_max: int = 2800,
    bin_width: int = 100,
    per_bin: int = 400,
    scan_cap: int = 0,
    batch_size: int = 256,
    write_positions: bool = False,
):
    return _run_puzzles(
        model=model,
        ndjson=ndjson,
        metadata=metadata,
        rating_min=rating_min,
        rating_max=rating_max,
        bin_width=bin_width,
        per_bin=per_bin,
        scan_cap=scan_cap,
        batch_size=batch_size,
        write_positions=write_positions,
        device="cuda",
    )


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
    seed: int = 0,
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
        seed=seed,
    )


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 2,
    volumes={"/data": volume},
)
def benchmark_h2h_shard(spec: dict):
    return _run_batched_h2h(**spec)


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 2,
    volumes={"/data": volume},
)
def benchmark_stockfish(
    model: str,
    stockfish_elo: int = 500,
    games: int = 50,
    temperature: float = 0.8,
    batch_size: int = 64,
    max_plies: int = 200,
    game_offset: int = 0,
    seed: int = 0,
    stockfish: str = "/usr/local/bin/fairy-stockfish",
    stockfish_time: float = 0.02,
    stockfish_depth: int = 0,
    stockfish_nodes: int = 0,
):
    return _run_batched_stockfish(
        model=model,
        stockfish_elo=stockfish_elo,
        games=games,
        temperature=temperature,
        batch_size=batch_size,
        max_plies=max_plies,
        game_offset=game_offset,
        seed=seed,
        stockfish=stockfish,
        stockfish_time=stockfish_time,
        stockfish_depth=stockfish_depth,
        stockfish_nodes=stockfish_nodes,
    )


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60 * 2,
    volumes={"/data": volume},
)
def benchmark_stockfish_shard(spec: dict):
    return _run_batched_stockfish(**spec)


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 30,
    volumes={"/data": volume},
)
def benchmark_legality(
    model: str = "plain/games-5m",
    data_dir: str = "/data/datasets/games/5m",
    split: str = "val",
    max_positions: int = 4096,
    batch_size: int = 128,
    max_ply: int = 140,
    max_games: int = 0,
    allow_eos: bool = False,
    top_illegal: int = 20,
    device: str = "cuda",
):
    return _run_legality(
        model=model,
        data_dir=data_dir,
        split=split,
        max_positions=max_positions,
        batch_size=batch_size,
        max_ply=max_ply,
        max_games=max_games,
        allow_eos=allow_eos,
        top_illegal=top_illegal,
        device=device,
    )


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60,
    volumes={"/data": volume},
)
def benchmark_move_quality(
    model: str,
    data_dir: str = "/data/datasets/eval/base",
    split: str = "val",
    max_positions: int = 600,
    max_ply: int = 140,
    max_games: int = 0,
    batch_size: int = 64,
    stockfish_time: float = 0.05,
):
    return _run_move_quality(
        model=model,
        data_dir=data_dir,
        split=split,
        max_positions=max_positions,
        max_ply=max_ply,
        max_games=max_games,
        batch_size=batch_size,
        stockfish_time=stockfish_time,
        device="cuda",
    )


@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 30,
    volumes={"/data": volume},
)
def inspect_legality_failures(
    model: str = "plain/games-5m",
    data_dir: str = "/data/datasets/games/5m",
    split: str = "val",
    phase: str = "opening",
    max_positions: int = 4096,
    batch_size: int = 128,
    max_ply: int = 140,
    max_games: int = 0,
    allow_eos: bool = False,
    device: str = "cuda",
):
    return _inspect_legality_failures(
        model=model,
        data_dir=data_dir,
        split=split,
        phase=phase,
        max_positions=max_positions,
        batch_size=batch_size,
        max_ply=max_ply,
        max_games=max_games,
        allow_eos=allow_eos,
        device=device,
    )


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


def _print_combined_summary(model_a: str, model_b: str, summaries: list[dict], wall_time: float) -> tuple[dict, list[dict]]:
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
    model: str = "plain/games-5m",
    model_a: str = "plain/games-3m",
    model_b: str = "plain/puzzles-5m",
    data_dir: str = "/data/datasets/games/5m",
    split: str = "val",
    max_positions: int = 4096,
    max_ply: int = 140,
    max_games: int = 0,
    allow_eos: bool = False,
    top_illegal: int = 20,
    write_failures: int = 0,
    games: int = 50,
    temperature: float = 0.8,
    batch_size: int = 64,
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

    if mode == "puzzles":
        puzzle_model = model if model != "plain/games-5m" else "/data/checkpoints/plain/games-15m/l16_best.pt"
        result = benchmark_puzzles.remote(
            model=puzzle_model,
            rating_min=rating_min,
            rating_max=rating_max,
            bin_width=bin_width,
            per_bin=per_bin,
            scan_cap=scan_cap,
            batch_size=batch_size if batch_size > 64 else 256,
            write_positions=write_positions,
        )
        summary = result["summary"]
        stamp = time.strftime("%Y%m%d_%H%M%S")
        if not output:
            output = f"benchmark/results/modal_puzzles_{stamp}.jsonl"
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        records = [summary, *result["positions"]]
        with out_path.open("w") as f:
            for record in records:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        if result["png_b64"]:
            png_path = out_path.with_suffix(".png")
            png_path.write_bytes(base64.b64decode(result["png_b64"]))
            print(f"wrote plot {png_path}")
        print(json.dumps(summary, indent=2, sort_keys=True))
        print(f"model: {puzzle_model}")
        print(f"first-move acc {summary['first_move_acc']:.3f} | full-solve acc {summary['full_solve_acc']:.3f} "
              f"over {summary['puzzles_scored']} puzzles")
        print(f"wrote {out_path}")
        return

    if mode == "legality":
        text = benchmark_legality.remote(
            model=model,
            data_dir=data_dir,
            split=split,
            max_positions=max_positions,
            batch_size=batch_size,
            max_ply=max_ply,
            max_games=max_games,
            allow_eos=allow_eos,
            top_illegal=top_illegal,
            device="cuda",
        )
        if not output:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            output = f"benchmark/results/modal_legality_{stamp}.jsonl"
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
        print(text)
        print(f"wrote {out_path}")
        return

    if mode == "move-quality":
        mq_data_dir = "/data/datasets/eval/base" if data_dir == "/data/datasets/games/5m" else data_dir
        text = benchmark_move_quality.remote(
            model=model,
            data_dir=mq_data_dir,
            split=split,
            max_positions=max_positions,
            max_ply=max_ply,
            max_games=max_games,
            batch_size=batch_size,
            stockfish_time=stockfish_time,
        )
        if not output:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            output = f"benchmark/results/modal_move_quality_{stamp}.jsonl"
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text)
        print(text)
        print(f"wrote {out_path}")
        return

    if mode == "stockfish":
        started = time.time()
        splits = _split_games(games, shards)
        opponent = f"sf{stockfish_elo}"
        print(
            f"Running {games} games against {opponent} with {len(splits)} shard(s), "
            f"batch_size={batch_size}, max_plies={max_plies}, temp={temperature}, "
            f"seed={seed}"
        )
        if len(splits) == 1:
            summaries = [
                benchmark_stockfish.remote(
                    model,
                    stockfish_elo,
                    games,
                    temperature,
                    batch_size,
                    max_plies,
                    0,
                    seed,
                    stockfish,
                    stockfish_time,
                    stockfish_depth,
                    stockfish_nodes,
                )
            ]
        else:
            jobs = [
                {
                    "model": model,
                    "stockfish_elo": stockfish_elo,
                    "games": count,
                    "temperature": temperature,
                    "batch_size": batch_size,
                    "max_plies": max_plies,
                    "game_offset": offset,
                    "seed": seed + offset,
                    "stockfish": stockfish,
                    "stockfish_time": stockfish_time,
                    "stockfish_depth": stockfish_depth,
                    "stockfish_nodes": stockfish_nodes,
                }
                for offset, count in splits
            ]
            summaries = list(benchmark_stockfish_shard.map(jobs))
        wall_time = time.time() - started
        summary, game_records = _print_combined_summary(model, opponent, summaries, wall_time)
        summary["stockfish_elo"] = stockfish_elo
        summary["stockfish"] = stockfish
        summary["stockfish_time"] = stockfish_time
        summary["stockfish_depth"] = stockfish_depth
        summary["stockfish_nodes"] = stockfish_nodes

        if not output:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            output = f"benchmark/results/modal_stockfish_{stamp}.jsonl"
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        run_record = {
            "type": "run",
            "schema_version": 1,
            "command": "modal_stockfish",
            "config": {
                "model": model,
                "opponent": opponent,
                "games": games,
                "temperature": temperature,
                "batch_size": batch_size,
                "max_plies": max_plies,
                "shards": shards,
                "seed": seed,
                "stockfish_elo": stockfish_elo,
                "stockfish": stockfish,
                "stockfish_time": stockfish_time,
                "stockfish_depth": stockfish_depth,
                "stockfish_nodes": stockfish_nodes,
            },
        }
        with out_path.open("w") as f:
            for record in [run_record, *game_records, summary]:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"wrote full game records to {out_path}")
        return

    if mode != "h2h":
        raise ValueError(f"unsupported mode: {mode}")

    started = time.time()
    splits = _split_games(games, shards)
    print(
        f"Running {games} H2H games with {len(splits)} shard(s), "
        f"batch_size={batch_size}, max_plies={max_plies}, temp={temperature}, "
        f"kv_cache={kv_cache}, seed={seed}"
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
                seed,
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
                "seed": seed + offset,
            }
            for offset, count in splits
        ]
        summaries = list(benchmark_h2h_shard.map(jobs))
    wall_time = time.time() - started
    summary, game_records = _print_combined_summary(model_a, model_b, summaries, wall_time)

    if not output:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        output = f"benchmark/results/modal_h2h_{stamp}.jsonl"
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run_record = {
        "type": "run",
        "schema_version": 1,
        "command": "modal_h2h",
        "config": {
            "model_a": model_a,
            "model_b": model_b,
            "games": games,
            "temperature": temperature,
            "batch_size": batch_size,
            "max_plies": max_plies,
            "shards": shards,
            "kv_cache": kv_cache,
            "seed": seed,
        },
    }
    with out_path.open("w") as f:
        for record in [run_record, *game_records, summary]:
            f.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"wrote full game records to {out_path}")
