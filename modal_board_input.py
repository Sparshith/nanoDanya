"""Oracle piece-location inputs: matched training, ablation, and chess games."""
import json
from pathlib import Path

import modal

from modal_board_benchmark import image, run_evaluation

app = modal.App("nanodanya-board-input")
volume = modal.Volume.from_name("nanodanya-data")
SOURCE = "/data/experiments/involved-pieces-20260921"
RUN = "/data/experiments/board-input-20260922"
CHAMPION = "/data/checkpoints/plain/games-15m/l16_best.pt"
MODELS = {arm: f"{RUN}/{arm}/final.pt" for arm in ("control", "oracle")}
image = image.add_local_file("modal_board_benchmark.py", "/root/project/modal_board_benchmark.py")


def setup():
    import os
    import sys
    os.chdir("/root/project")
    sys.path[:0] = ["/root/project", "/root/project/nanochat"]


@app.function(image=image, cpu=4, memory=8192, volumes={"/data": volume}, timeout=3600)
def prepare():
    setup()
    from data_prep.prepare_board_inputs import prepare as prepare_inputs
    from data_prep.prepare_involved_pieces import file_hash

    volume.reload()
    assert file_hash(CHAMPION) == "7e72148ed260564d671a67f5e0bc9bc4d82fd11695a34d9f4708858033955b0e"
    output = Path(RUN) / "boards"
    if (output / "manifest.json").exists():
        return json.loads((output / "manifest.json").read_text())
    result = prepare_inputs(Path(SOURCE) / "dataset", Path(SOURCE) / "labels", output)
    volume.commit()
    return result


@app.function(image=image, gpu="H100", cpu=8, memory=32768, volumes={"/data": volume}, timeout=7200)
def train(arm):
    setup()
    import os
    import subprocess
    import time
    from data_prep.prepare_involved_pieces import file_hash

    assert arm in MODELS
    volume.reload()
    output = Path(RUN) / arm
    env = dict(os.environ, DATASET_DIR=f"{SOURCE}/dataset", BOARD_INPUT_DIR=f"{RUN}/boards",
               CKPT_DIR=str(output), INIT_CHECKPOINT=CHAMPION,
               BOARD_INPUT_ENABLED="1" if arm == "oracle" else "0", MAX_ITERS="1000",
               BATCH_SIZE="32", LR="3e-6", BOARD_ENCODER_LR="1e-4", SEED="42",
               VAL_BATCHES="10", EVAL_INTERVAL="100", OMP_NUM_THREADS="8")
    started = time.monotonic()
    try:
        subprocess.run(["python", "training/train_board_input.py"], env=env, check=True)
        result = {"arm": arm, "seconds": time.monotonic() - started, "checkpoint": MODELS[arm],
                  "sha256": file_hash(MODELS[arm]),
                  "baseline": json.loads((output / "baseline.json").read_text()),
                  "metrics": [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]}
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        volume.commit()


@app.function(image=image, gpu="H100", cpu=4, memory=32768, max_containers=4,
              volumes={"/data": volume}, timeout=7200)
def benchmark(name, subcommand, flags, hashes):
    return run_evaluation(name, subcommand, flags, hashes, f"{RUN}/benchmarks")


@app.function(image=image, gpu="H100", cpu=4, memory=32768, volumes={"/data": volume}, timeout=1800)
def diagnostics():
    setup()
    import chess
    import torch
    import torch.nn.functional as F
    from benchmark.run import load_token_games
    from chess_inference import choose_move_from_logits, legal_token_ids, token_for_id
    from chess_token_utils import resolve_token_id, strip_san
    from data_prep.prepare_involved_pieces import file_hash
    from model_loading import load_model
    from probes.inspect_game import game_position

    volume.reload()
    torch.set_num_threads(4)
    games, _ = load_token_games(Path(SOURCE) / "dataset", "probe", 200)
    pgn = (Path(SOURCE) / "game.pgn").read_text()
    result = {}
    with torch.inference_mode():
        for arm, checkpoint in {**MODELS, "champion": CHAMPION}.items():
            player, _, stoi, itos = load_model(checkpoint, "cuda")
            item = {"checkpoint": checkpoint, "sha256": file_hash(checkpoint)}
            if arm != "champion":
                sequences = [[stoi[t] if t.startswith("<") else resolve_token_id(stoi, t)
                              for t in game[:141]] for game in games]
                total, actual_loss, blank_loss, changed = 0, 0.0, 0.0, 0
                for start in range(0, len(sequences), 16):
                    rows = sequences[start:start + 16]
                    length = max(len(row) - 1 for row in rows)
                    x = torch.full((len(rows), length), stoi["<eos>"], device="cuda", dtype=torch.long)
                    y = torch.full_like(x, -1)
                    for i, row in enumerate(rows):
                        x[i, :len(row) - 1] = torch.tensor(row[:-1], device="cuda")
                        y[i, :len(row) - 1] = torch.tensor(row[1:], device="cuda")
                    valid = (y >= 0) & (y != stoi["<bos>"]) & (y != stoi["<eos>"])
                    player.blank_board = False
                    actual = player(x).float()
                    player.blank_board = True
                    blank = player(x).float()
                    actual_loss += float(F.cross_entropy(actual[valid], y[valid], reduction="sum"))
                    blank_loss += float(F.cross_entropy(blank[valid], y[valid], reduction="sum"))
                    changed += int(((actual.argmax(-1) != blank.argmax(-1)) & valid).sum())
                    total += int(valid.sum())
                item["probe_ablation"] = {"games": len(games), "moves": total,
                    "real_board_move_ce": actual_loss / total, "blank_board_move_ce": blank_loss / total,
                    "raw_greedy_changes": changed, "raw_greedy_change_rate": changed / total}
                player.blank_board = False
            position = game_position(pgn, stoi, 71)
            board = chess.Board(position["fen"][0])
            fatal = set()
            for move in list(board.legal_moves):
                after = board.copy()
                after.push(move)
                for reply in list(after.legal_moves):
                    after.push(reply)
                    mate = after.is_checkmate()
                    after.pop()
                    if mate:
                        fatal.add(strip_san(board.san(move)))
                        break
            conditions = ("actual", "blank") if arm != "champion" else ("actual",)
            item["before_g5"] = {"fen": board.fen(), "fatal_moves": sorted(fatal), "temperature": 1.0}
            for condition in conditions:
                if arm != "champion":
                    player.blank_board = condition == "blank"
                logits = player(torch.tensor(position["prefixes"], device="cuda"))[0, -1].float()
                legal = legal_token_ids(stoi, board, allow_eos=False)
                probs = logits[legal].softmax(-1)
                mass = sum(float(probs[i]) for i, token in enumerate(legal)
                           if strip_san(token_for_id(itos, token)) in fatal)
                _, greedy = choose_move_from_logits(logits, board, stoi, itos, temperature=0,
                                                     allow_eos=False, legal_mask=True)
                item["before_g5"][condition] = {"greedy_move": greedy, "mate_in_one_blunder_mass": mass}
            result[arm] = item
            del player
    (Path(RUN) / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.function(image=image, cpu=0.25, memory=2048, timeout=21600)
def pipeline(benchmark_only=False):
    training = []
    if not benchmark_only:
        prepare.remote()
        training = [train.remote(arm) for arm in MODELS]
    jobs = []
    hashes = {row["arm"]: row["sha256"] for row in training}
    if benchmark_only:
        hashes = checkpoint_hashes.remote()
    for arm in MODELS:
        base = {"model": MODELS[arm], "data_dir": f"{SOURCE}/dataset", "split": "probe",
                "batch_size": 64, "max_games": 200, "max_ply": 140}
        digest = {"model": hashes[arm]}
        for command in ("legality", "termination"):
            jobs.append((f"{arm}-{command}", command, base | {"max_positions": 4096}, digest))
        jobs.append((f"{arm}-move-quality", "move-quality", base | {
            "max_positions": 1024, "stockfish": "/usr/games/stockfish", "stockfish_depth": 10,
            "write_positions": True}, digest))
    for offset in range(0, 4000, 50):
        jobs.append((f"oracle-vs-control-{offset:04d}", "games", {
            "game_mode": "h2h", "model": MODELS["oracle"], "opponent_model": MODELS["control"],
            "games": 50, "game_offset": offset, "seed": 20270000 + offset,
            "batch_size": 50, "temperature": 0.8, "max_plies": 200, "kv_cache": True,
        }, {"model": hashes["oracle"], "opponent_model": hashes["control"]}))
    benchmarks = []
    for row in benchmark.starmap(jobs, order_outputs=False):
        benchmarks.append(row)
        print(json.dumps({"finished_benchmark": row["name"], "completed_jobs": len(benchmarks), "total_jobs": len(jobs)}))
    return {"training": training, "benchmarks": benchmarks, "diagnostics": diagnostics.remote()}


@app.function(image=image, cpu=0.25, memory=2048, volumes={"/data": volume}, timeout=120)
def checkpoint_hashes():
    setup()
    from data_prep.prepare_involved_pieces import file_hash
    volume.reload()
    return {arm: file_hash(path) for arm, path in MODELS.items()}


@app.function(image=image, gpu="H100", cpu=4, memory=32768, volumes={"/data": volume}, timeout=600)
def temperature_trace(temperature: float):
    setup()
    import chess
    import torch
    from chess_inference import choose_move_from_logits, legal_token_ids, token_for_id
    from chess_token_utils import strip_san
    from data_prep.prepare_involved_pieces import file_hash
    from model_loading import load_model
    from probes.inspect_game import game_position

    volume.reload()
    torch.set_num_threads(4)
    pgn = (Path(SOURCE) / "game.pgn").read_text()
    result = {}
    with torch.inference_mode():
        for arm, path in {**MODELS, "champion": CHAMPION}.items():
            player, _, stoi, itos = load_model(path, "cuda")
            position = game_position(pgn, stoi, 71)
            board = chess.Board(position["fen"][0])
            logits = player(torch.tensor(position["prefixes"], device="cuda"))[0, -1].float()
            legal = legal_token_ids(stoi, board, allow_eos=False)
            _, greedy = choose_move_from_logits(logits, board, stoi, itos, temperature=0,
                                                 allow_eos=False, legal_mask=True)
            distributions = {}
            for value in sorted({temperature, 0.8, 1.0}):
                probs = (logits[legal] / value).softmax(-1)
                distributions[str(value)] = {
                    "g5_probability": sum(float(probs[i]) for i, token in enumerate(legal)
                                          if strip_san(token_for_id(itos, token)) == "g5"),
                    "top5": [{"move": token_for_id(itos, legal[i]), "probability": float(probs[i])}
                             for i in probs.argsort(descending=True)[:5].tolist()]}
            result[arm] = {"checkpoint": path, "sha256": file_hash(path), "fen": board.fen(),
                           "greedy_move": greedy, "temperatures": distributions}
            del player
    (Path(RUN) / f"temperature-t{temperature:g}.json").write_text(json.dumps(result, indent=2) + "\n")
    volume.commit()
    return result


@app.function(image=image, cpu=0.25, memory=2048, timeout=21600)
def temperature_suite(temperature: float):
    import math
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    hashes = {
        "model": "b07f4d6ac0a7330faad2937954e9dee6f4dca4a0251191cda073196da34828be",
        "opponent_model": "d30440939a8e64634169cefecc6f7254b3e646c893b6b1f2f378fd6a0d9402e8",
    }
    jobs = [(f"t{temperature:g}-oracle-vs-control-{offset:04d}", "games", {
        "game_mode": "h2h", "model": MODELS["oracle"], "opponent_model": MODELS["control"],
        "games": 50, "game_offset": offset, "seed": 20270000 + offset,
        "batch_size": 50, "temperature": temperature, "max_plies": 200, "kv_cache": True,
    }, hashes) for offset in range(4000, 8000, 50)]
    results = []
    for row in benchmark.starmap(jobs, order_outputs=False):
        results.append(row)
        print(json.dumps({"temperature": temperature, "completed_games": len(results) * 50, "target_games": 4000}))
    return {"benchmarks": results, "temperature_trace": temperature_trace.remote(temperature),
            "protocol": {"temperature": temperature, "games": 4000, "game_offset": 4000,
                         "seed_base": 20270000, "stopping_rule": "fixed 4000 fresh games"}}


@app.local_entrypoint()
def main(benchmark_only: bool = False, temperature_check: bool = False, temperature: float = 0.2):
    result = temperature_suite.remote(temperature) if temperature_check else pipeline.remote(benchmark_only)
    suffix = f"-t{temperature:g}" if temperature_check else ""
    root = Path(f"benchmark/results/board-input-20260922{suffix}")
    root.mkdir(parents=True, exist_ok=True)
    for row in result["benchmarks"]:
        (root / f"{row['name']}.json").write_text(json.dumps(row, indent=2) + "\n")
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "benchmarks"}, indent=2))
