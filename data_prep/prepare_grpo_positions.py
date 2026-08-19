"""Materialize GRPO training positions from puzzles_raw onto the volume.

Each record: the token-id prefix (game replayed to the puzzle position, lead-in
included), the set of correct next-token ids (solution first move plus any mating
move), rating, fen, and the full solution line for later full-line reward.

Skips the first SKIP_GAMES games in file order: the puzzle benchmark samples its
positions from the head of the file, so that zone is reserved for eval.

Run: MODAL_GPU=A10G uv run modal run modal_train.py --datasets puzzles_raw \
  --script data_prep/prepare_grpo_positions.py
"""

import hashlib
import json
import os
import re
import sys
from itertools import islice
from multiprocessing import Pool
from pathlib import Path

import chess
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chess_token_utils import resolve_token_id

CKPT = os.getenv("CKPT", "/data/checkpoints/plain/games-15m/l16_best.pt")
NDJSON = Path(os.getenv("NDJSON", "/data/puzzles_raw/puzzle_games_ndjson.txt"))
METADATA = Path(os.getenv("METADATA", "/data/puzzles_raw/puzzle_metadata.txt"))
OUT_DIR = Path(os.getenv("OUT_DIR", "/data/datasets/puzzles/grpo"))
SKIP_GAMES = int(os.getenv("SKIP_GAMES", "100000"))
MAX_POSITIONS = int(os.getenv("MAX_POSITIONS", "500000"))
VAL_MOD = 50  # 1 in 50 positions to val, by puzzle_id hash
WORKERS = min(16, os.cpu_count() or 1)

PUZZLE_ID_RE = re.compile(rb'"id":"([^"]+)"')


def build_record(line: bytes) -> dict | None:
    match = PUZZLE_ID_RE.search(line)
    if not match:
        return None
    puzzles = META.get(match.group(1).decode())
    if not puzzles:
        return None
    p = puzzles[0]
    moves = json.loads(line)["moves"].split()
    sol = p["moves"].split()
    move_num = p["move_num"]
    if move_num - 1 > len(moves):
        return None

    board = chess.Board()
    prefix = [STOI["<bos>"]]
    for san in moves[: move_num - 1]:
        tid = resolve_token_id(STOI, san)
        if tid is None:
            return None
        try:
            board.push_san(san)
        except ValueError:
            return None
        prefix.append(tid)

    if board.fen() != p["fen"]:
        return None

    try:
        lead = board.parse_uci(sol[0])
    except ValueError:
        return None
    lead_tid = resolve_token_id(STOI, board.san(lead))
    if lead_tid is None:
        return None
    prefix.append(lead_tid)
    board.push(lead)
    if len(prefix) >= MAX_LEN:
        return None

    expected = board.parse_uci(sol[1])
    expected_tid = resolve_token_id(STOI, board.san(expected))
    if expected_tid is None:
        return None
    correct = {expected_tid}
    for mv in board.legal_moves:
        if mv == expected:
            continue
        probe = board.copy(stack=False)
        probe.push(mv)
        if probe.is_checkmate():
            tid = resolve_token_id(STOI, board.san(mv))
            if tid is not None:
                correct.add(tid)

    return {
        "puzzle_id": p["puzzle_id"],
        "rating": p["rating"],
        "fen": p["fen"],
        "sol": p["moves"],
        "prefix": prefix,
        "correct": sorted(correct),
    }


print(f"loading tokenizer from {CKPT} ...", flush=True)
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
STOI = ckpt["meta"]["tokenizer"]["stoi"]
MAX_LEN = ckpt["meta"]["model_config"]["sequence_len"]
del ckpt

print(f"loading metadata {METADATA} ...", flush=True)
META = json.load(METADATA.open())
print(f"metadata games: {len(META):,}", flush=True)

OUT_DIR.mkdir(parents=True, exist_ok=True)
train_path = OUT_DIR / "train.jsonl"
val_path = OUT_DIR / "val.jsonl"

kept = skipped = 0
n_train = n_val = 0
rating_hist: dict[int, int] = {}
prefix_len_sum = 0

with NDJSON.open("rb") as f, train_path.open("w") as train_f, val_path.open("w") as val_f, Pool(WORKERS) as pool:
    lines = islice(f, SKIP_GAMES, SKIP_GAMES + int(MAX_POSITIONS * 1.1))
    for rec in pool.imap_unordered(build_record, lines, chunksize=256):
        if rec is None:
            skipped += 1
            continue
        kept += 1
        rating_hist[rec["rating"] // 200 * 200] = rating_hist.get(rec["rating"] // 200 * 200, 0) + 1
        prefix_len_sum += len(rec["prefix"])
        row = json.dumps(rec, separators=(",", ":")) + "\n"
        if int(hashlib.md5(rec["puzzle_id"].encode()).hexdigest(), 16) % VAL_MOD == 0:
            val_f.write(row)
            n_val += 1
        else:
            train_f.write(row)
            n_train += 1
        if kept % 50000 == 0:
            print(f"  kept {kept:,} positions ({skipped:,} skipped)", flush=True)
        if kept >= MAX_POSITIONS:
            break

meta = {
    "source_ndjson": str(NDJSON),
    "tokenizer_ckpt": CKPT,
    "skip_games": SKIP_GAMES,
    "n_train": n_train,
    "n_val": n_val,
    "skipped": skipped,
    "val_mod": VAL_MOD,
    "max_len": MAX_LEN,
    "avg_prefix_len": round(prefix_len_sum / kept, 1) if kept else None,
    "rating_hist": {str(k): rating_hist[k] for k in sorted(rating_hist)},
}
(OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

print(f"\nkept {kept:,} positions ({skipped:,} skipped): train {n_train:,}, val {n_val:,}")
print(f"avg prefix length: {meta['avg_prefix_len']} tokens (cap {MAX_LEN})")
for b in sorted(rating_hist):
    print(f"  {b:>5}-{b + 200:<5} {rating_hist[b]:>8,}")
print(f"wrote {train_path} ({train_path.stat().st_size / 1e6:.0f} MB), "
      f"{val_path} ({val_path.stat().st_size / 1e6:.0f} MB), meta.json")
