"""Grade positions from archived champion games, split by complete game."""

from collections import Counter
import hashlib
import json
import math
from multiprocessing import Pool
import os
from pathlib import Path
import random

import chess
import chess.engine
import torch

from chess_token_utils import resolve_token_id


def game_splits(games, seed):
    ids = sorted(game["game_id"] for game in games)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source games")
    random.Random(seed).shuffle(ids)
    train_end, val_end = int(len(ids) * 0.8), int(len(ids) * 0.9)
    return {gid: "train" if i < train_end else "val" if i < val_end else "test"
            for i, gid in enumerate(ids)}


def candidate_positions(game, stoi, sequence_len, seed, per_game=32):
    board = chess.Board()
    prefix = [stoi["<bos>"]]
    own_color = game["model_color"] == "white"
    records = []
    for ply, san in enumerate(game["moves"]):
        move = board.parse_san(san)
        canonical = board.san(move)
        if board.turn == own_color and ply >= 16 and len(prefix) <= sequence_len and not board.is_game_over():
            records.append({"game_id": game["game_id"], "ply": ply,
                            "history": " ".join(game["moves"][:ply]), "fen": board.fen(),
                            "prefix": list(prefix), "played": canonical})
        token_id = resolve_token_id(stoi, canonical)
        if token_id is None:
            raise ValueError(f"Unresolvable history token: {canonical}")
        prefix.append(token_id)
        board.push(move)
    rng = random.Random(f"{seed}:{game['game_id']}")
    return sorted(rng.sample(records, min(per_game, len(records))), key=lambda row: row["ply"])


def winprob(cp):
    return 1 / (1 + math.exp(-0.00368208 * max(-1000, min(1000, cp))))


def init_engine(path, nodes):
    global ENGINE, NODES
    ENGINE = chess.engine.SimpleEngine.popen_uci(path)
    ENGINE.configure({"Threads": 1, "Hash": 32})
    NODES = nodes


def grade(record):
    board = chess.Board()
    for san in record["history"].split():
        board.push_san(san)
    assert board.fen() == record["fen"]
    ENGINE.configure({"Clear Hash": None})
    infos = ENGINE.analyse(board, chess.engine.Limit(nodes=NODES), multipv=board.legal_moves.count())
    scores = {board.san(info["pv"][0]): info["score"].pov(board.turn).score(mate_score=10000)
              for info in infos if info.get("pv") and info.get("score") is not None}
    legal = {board.san(move) for move in board.legal_moves}
    if set(scores) != legal:
        raise ValueError(f"Incomplete engine table: {record['game_id']} ply {record['ply']}")
    for move in list(board.legal_moves):
        san = board.san(move)
        board.push(move)
        outcome = board.outcome()
        if outcome is not None:
            scores[san] = 0 if outcome.winner is None else (10000 if outcome.winner != board.turn else -10000)
        board.pop()
    best = max(scores.values())
    loss = max(0, best - scores[record["played"]])
    return dict(record, cp=scores, moves={san: winprob(cp) for san, cp in scores.items()},
                best_cp=best, played_cp_loss=loss, priority=best >= 200 or (best >= -200 and loss >= 300),
                min_depth=min(info.get("depth", 0) for info in infos))


def completed_records(output):
    completed, counts, priority, depths = set(), Counter(), Counter(), []
    for split in ("train", "val", "test"):
        path = output / f"{split}.jsonl"
        if not path.exists():
            continue
        with path.open("rb+") as handle:
            while True:
                start = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.endswith(b"\n"):
                    handle.truncate(start)
                    break
                row = json.loads(line)
                key = (row["game_id"], row["ply"])
                if row["split"] != split or key in completed:
                    raise ValueError("Invalid split or duplicate position in partial data")
                completed.add(key)
                counts[split] += 1
                priority[split] += int(row["priority"])
                depths.append(row["min_depth"])
    return completed, counts, priority, depths


def main():
    source = Path(os.environ["OWN_GAMES_INPUT"])
    output = Path(os.environ["OWN_REWARDS_OUTPUT"])
    resume = os.getenv("OWN_REWARDS_RESUME", "0") == "1"
    output.mkdir(parents=True, exist_ok=resume)
    seed = int(os.getenv("SEED", "20260918"))
    checkpoint = os.getenv("CKPT_INIT", "/data/checkpoints/plain/games-15m/l16_best.pt")
    engine = os.getenv("ENGINE_PATH", "/usr/local/bin/fairy-stockfish")
    nodes = int(os.getenv("ENGINE_NODES", "50000"))
    workers = int(os.getenv("ENGINE_WORKERS", "4"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    stoi = state["meta"]["tokenizer"]["stoi"]
    context = state["meta"]["model_config"]["sequence_len"]
    del state
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    run = next(row for row in rows if row["type"] == "run")
    if run["config"]["model"] != "plain/games-15m" or run["config"]["game_mode"] != "stockfish":
        raise ValueError("Expected archived champion-versus-Stockfish games")
    games = [row for row in rows if row["type"] == "game"]
    splits = game_splits(games, seed)
    records, seen = [], {}
    for game in sorted(games, key=lambda row: row["game_id"]):
        for record in candidate_positions(game, stoi, context, seed):
            key = " ".join(record["fen"].split()[:4])
            seen.setdefault(key, set()).add(splits[record["game_id"]])
            records.append(dict(record, split=splits[record["game_id"]]))
    records = [row for row in records if len(seen[" ".join(row["fen"].split()[:4])]) == 1]
    protocol = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "source_run": run, "checkpoint": checkpoint, "seed": seed, "game_splits": splits,
                "positions": dict(Counter(row["split"] for row in records)), "per_game_max": 32,
                "engine_nodes": nodes, "engine_sha256": hashlib.sha256(Path(engine).read_bytes()).hexdigest(),
                "reward": "sigmoid(0.00368208 * clamp(mover_cp, -1000, 1000))",
                "sampling": "half uniform, half winning or playable blunder positions during training",
                "selection": "random own-turn positions from ply 16; cross-split identical board states removed"}
    if resume:
        if json.loads((output / "protocol.json").read_text()) != protocol:
            raise ValueError("Preparation resume protocol differs from the saved run")
    else:
        (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({"games": dict(Counter(splits.values())), "positions": protocol["positions"]}), flush=True)
    completed, counts, priority, depths = completed_records(output) if resume else (set(), Counter(), Counter(), [])
    expected = {(row["game_id"], row["ply"]) for row in records}
    if not completed <= expected:
        raise ValueError("Partial data contains unknown positions")
    pending = [row for row in records if (row["game_id"], row["ply"]) not in completed]
    handles = {split: (output / f"{split}.jsonl").open("a" if resume else "w") for split in ("train", "val", "test")}
    try:
        with Pool(workers, initializer=init_engine, initargs=(engine, nodes)) as pool:
            for i, record in enumerate(pool.imap(grade, pending, chunksize=8), len(completed) + 1):
                handles[record["split"]].write(json.dumps(record) + "\n")
                counts[record["split"]] += 1
                priority[record["split"]] += int(record["priority"])
                depths.append(record["min_depth"])
                if i % 500 == 0:
                    for handle in handles.values():
                        handle.flush()
                    print(f"graded {i}/{len(records)}", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    manifest = dict(protocol, positions=dict(counts), priority=dict(priority), mean_min_depth=sum(depths) / len(depths),
                    files={split: hashlib.sha256((output / f"{split}.jsonl").read_bytes()).hexdigest() for split in handles})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"positions": dict(counts), "priority": dict(priority), "mean_min_depth": manifest["mean_min_depth"]}), flush=True)


if __name__ == "__main__":
    main()
