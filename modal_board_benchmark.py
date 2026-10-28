"""Matched chess benchmarks for the joint board-supervision pilot."""
import json
from pathlib import Path

import modal

app = modal.App("nanodanya-joint-board-benchmark")
volume = modal.Volume.from_name("nanodanya-data")
RUN = "/data/experiments/involved-pieces-20260921"
MODELS = {"champion": "/data/checkpoints/plain/games-15m/l16_best.pt",
          "control": f"{RUN}/joint-board-control/final.pt", "joint": f"{RUN}/joint-board/final.pt"}
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("stockfish")
    .pip_install("torch==2.5.1+cu124", "numpy==2.2.6", "python-chess==1.999", "requests==2.32.3",
                 extra_index_url="https://download.pytorch.org/whl/cu124")
    .env({"PYTHONPATH": "/root/project:/root/project/nanochat", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4"})
    .add_local_dir("benchmark", "/root/project/benchmark", ignore=["results", "__pycache__", "*.jsonl", "*.json", "*.html"])
    .add_local_dir("inference", "/root/project/inference", ignore=["__pycache__"])
    .add_local_dir("nanochat/nanochat", "/root/project/nanochat/nanochat")
    .add_local_dir("probes", "/root/project/probes", ignore=["*.html", "__pycache__"])
    .add_local_dir("training", "/root/project/training", ignore=["__pycache__"])
    .add_local_dir("data_prep", "/root/project/data_prep", ignore=["__pycache__"])
    .add_local_file("chess_token_utils.py", "/root/project/chess_token_utils.py")
    .add_local_file("chess_inference.py", "/root/project/chess_inference.py")
    .add_local_file("model_loading.py", "/root/project/model_loading.py")
    .add_local_file("model_registry.py", "/root/project/model_registry.py")
)


def run_evaluation(name: str, subcommand: str, flags: dict, expected_hashes: dict | None = None,
                   result_root: str | None = None):
    import os
    import random
    import sys
    import time

    import torch

    os.chdir("/root/project")
    sys.path[:0] = ["/root/project", "/root/project/nanochat"]
    from benchmark.run import build_parser
    from data_prep.prepare_involved_pieces import file_hash

    volume.reload()
    output = Path(result_root) if result_root else Path(RUN) / "benchmarks/joint-board"
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / f"{name}.json"
    hashes = {key: file_hash(flags[key]) for key in ("model", "opponent_model") if key in flags}
    if expected_hashes is not None and hashes != expected_hashes:
        raise ValueError(f"Checkpoint differs from the original experiment: {name}")
    request = {"subcommand": subcommand, "flags": flags, "model_hashes": hashes}
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["request"] != request:
            raise ValueError(f"Cached benchmark request differs: {name}")
        return saved
    argv = [subcommand, "--output", str(output / f"{name}.jsonl")]
    for key, value in flags.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        else:
            argv.extend([flag, str(value)])
    args = build_parser().parse_args(argv)
    seed = getattr(args, "seed", 0)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    started = time.monotonic()
    args.func(args)
    records = [json.loads(line) for line in Path(args.output).read_text().splitlines()]
    result = {"name": name, "request": request, "records": records, "seconds": time.monotonic() - started}
    if subcommand == "move-quality":
        result["engine_sha256"] = file_hash(args.stockfish)
    receipt.write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.function(image=image, gpu="H100", cpu=4, memory=32768, max_containers=4,
              volumes={"/data": volume}, timeout=7200)
def evaluate(name: str, subcommand: str, flags: dict, expected_hashes: dict | None = None):
    return run_evaluation(name, subcommand, flags, expected_hashes)


def position_jobs(arms):
    jobs = []
    for arm in arms:
        base = {"model": MODELS[arm], "data_dir": f"{RUN}/dataset", "split": "probe",
                "batch_size": 64, "max_games": 200, "max_ply": 140}
        jobs.append((f"{arm}-legality", "legality", base | {"max_positions": 4096}))
        jobs.append((f"{arm}-termination", "termination", base | {"max_positions": 4096}))
        jobs.append((f"{arm}-move-quality", "move-quality", base | {
            "max_positions": 1024, "stockfish": "/usr/games/stockfish", "stockfish_depth": 10,
            "write_positions": True}))
    return jobs


@app.function(image=image, gpu="H100", cpu=4, memory=32768, volumes={"/data": volume}, timeout=1200)
def trace():
    import os
    import sys
    import torch

    os.chdir("/root/project")
    sys.path[:0] = ["/root/project", "/root/project/nanochat"]
    from probes.involved_trace import inspect_head
    from chess_inference import legal_token_ids, choose_move_from_logits
    from model_loading import load_model
    from probes.inspect_game import game_position

    volume.reload()
    torch.set_num_threads(4)
    pgn = (Path(RUN) / "game.pgn").read_text()
    result = {}
    for arm, checkpoint in MODELS.items():
        model, _, stoi, itos = load_model(checkpoint, "cuda")
        position = game_position(pgn, stoi, 71)
        import chess
        board = chess.Board(position["fen"][0])
        with torch.inference_mode():
            logits = model(torch.tensor(position["prefixes"], device="cuda"))[0, -1]
        legal = legal_token_ids(stoi, board, allow_eos=False)
        probabilities = logits[legal].softmax(-1)
        fatal = set()
        for move in board.legal_moves:
            after = board.copy()
            after.push(move)
            for reply in list(after.legal_moves):
                after.push(reply)
                mate = after.is_checkmate()
                after.pop()
                if mate:
                    fatal.add(board.san(move))
                    break
        from chess_token_utils import strip_san
        fatal_normalized = {strip_san(san) for san in fatal}
        from chess_inference import token_for_id
        fatal_mass = sum(float(probabilities[i]) for i, token in enumerate(legal)
                         if strip_san(token_for_id(itos, token)) in fatal_normalized)
        _, greedy = choose_move_from_logits(logits, board, stoi, itos, temperature=0,
                                            allow_eos=False, legal_mask=True)
        del model
        head_checkpoint = f"{RUN}/all-pieces/final.pt" if arm == "champion" else checkpoint
        result[arm] = {"before_g5": {"greedy_move": greedy, "temperature": 1.0,
                                      "mate_in_one_blunder_mass": fatal_mass,
                                      "fatal_moves": sorted(fatal)},
                       "head": inspect_head(head_checkpoint, pgn, all_positions=True)}
    path = Path(RUN) / "benchmarks/joint-board/trace.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.function(image=image, cpu=0.25, memory=2048, timeout=21600)
def suite(baseline_only: bool = False):
    jobs = position_jobs(["champion"] if baseline_only else MODELS)
    if not baseline_only:
        for a, b in (("joint", "control"), ("joint", "champion"), ("control", "champion")):
            for offset in range(0, 400, 50):
                jobs.append((f"{a}-vs-{b}-{offset:03d}", "games", {
                    "game_mode": "h2h", "model": MODELS[a], "opponent_model": MODELS[b],
                    "games": 50, "game_offset": offset, "seed": 20260921 + offset,
                    "batch_size": 50, "temperature": 0.8, "max_plies": 200, "kv_cache": True}))
    results = list(evaluate.starmap(jobs, order_outputs=False))
    return {"benchmarks": results, "trace": None if baseline_only else trace.remote()}


@app.function(image=image, cpu=0.25, memory=2048, timeout=21600)
def confirmation_suite():
    hashes = {
        "model": "e231e07541dc828fbfc751565e6c1b112fc14d178a7062653b2f26b9b6baa454",
        "opponent_model": "e2298703c6719cf9e7709af1b9c0cd0ea602ef991086042eebbb5404f5e2ba10",
    }
    jobs = [(f"confirmation-joint-vs-control-{offset:04d}", "games", {
        "game_mode": "h2h", "model": MODELS["joint"], "opponent_model": MODELS["control"],
        "games": 50, "game_offset": offset, "seed": 20260922 + offset,
        "batch_size": 50, "temperature": 0.8, "max_plies": 200, "kv_cache": True,
    }, hashes) for offset in range(400, 4400, 50)]
    results = []
    for row in evaluate.starmap(jobs, order_outputs=False):
        results.append(row)
        print(json.dumps({"confirmation_completed_games": len(results) * 50, "target_games": 4000}))
    return {"benchmarks": results, "trace": None, "protocol": {
        "games": 4000, "comparison": "joint versus next-move-only control",
        "game_offset": 400, "seed_base": 20260922, "stopping_rule": "fixed 4000 fresh games",
        "primary_analysis": "fresh games only; earlier 400 are exploratory",
    }}


@app.local_entrypoint()
def main(baseline_only: bool = False, confirmation: bool = False):
    if baseline_only and confirmation:
        raise ValueError("Choose baseline-only or confirmation")
    result = confirmation_suite.remote() if confirmation else suite.remote(baseline_only)
    root = Path("benchmark/results/joint-board-confirmation-20260922" if confirmation
                else "benchmark/results/joint-board-20260921")
    root.mkdir(parents=True, exist_ok=True)
    for row in result["benchmarks"]:
        (root / f"{row['name']}.json").write_text(json.dumps(row, indent=2) + "\n")
    name = "baseline.json" if baseline_only else "result.json"
    (root / name).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({row["name"]: [r for r in row["records"] if r["type"].endswith("_summary")]
                      for row in result["benchmarks"]}, indent=2))
