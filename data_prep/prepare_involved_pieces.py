"""Pre-move locations of move-involved pieces, aligned with SAN target tokens."""
import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path

import chess
import numpy as np


INITIAL_SQUARES = tuple(sorted(chess.Board().piece_map()))
MOVER, CAPTURED, CASTLING_ROOK = range(3)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_move(board, identities, move):
    """Return (identity, current square, role), then advance board and identities."""
    labels = [(identities[move.from_square], move.from_square, MOVER)]
    captured_square = move.to_square
    if board.is_en_passant(move):
        captured_square += -8 if board.turn == chess.WHITE else 8
    if board.is_capture(move):
        labels.append((identities.pop(captured_square), captured_square, CAPTURED))
    if board.is_castling(move):
        rank = chess.square_rank(move.from_square)
        kingside = board.is_kingside_castling(move)
        rook_from = chess.square(7 if kingside else 0, rank)
        rook_to = chess.square(5 if kingside else 3, rank)
        labels.append((identities[rook_from], rook_from, CASTLING_ROOK))
        identities[rook_to] = identities.pop(rook_from)
    identities[move.to_square] = identities.pop(move.from_square)
    board.push(move)
    return labels


def prepare_split(data_path, output_path, meta):
    tokens = np.memmap(data_path, dtype=np.uint16, mode="r")
    labels = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.int8,
                                      shape=(len(tokens), 2, 3))
    labels[:] = -1
    bos, eos = meta["stoi"]["<bos>"], meta["stoi"]["<eos>"]
    itos = meta["itos"]
    board = None
    games, moves = 0, 0
    roles = [0, 0, 0]
    for index, token_id in enumerate(tokens):
        token_id = int(token_id)
        if token_id == bos:
            board = chess.Board()
            identities = dict(zip(INITIAL_SQUARES, range(32)))
            games += 1
        elif token_id == eos:
            board = None
        else:
            if board is None:
                raise ValueError(f"Move without game start at {data_path}:{index}")
            san = itos[token_id] if not isinstance(itos, dict) or token_id in itos else itos[str(token_id)]
            try:
                move = board.parse_san(san)
                if not move:
                    raise ValueError("Null moves are not supported")
                involved = label_move(board, identities, move)
            except ValueError as exc:
                raise ValueError(f"Invalid SAN {san!r} at {data_path}:{index}") from exc
            labels[index, :len(involved)] = involved
            for _, _, role in involved:
                roles[role] += 1
            moves += 1
        if (index + 1) % 1_000_000 == 0:
            print(f"{data_path.name}: {index + 1:,}/{len(tokens):,} tokens", flush=True)
    labels.flush()
    return {"tokens": len(tokens), "games": games, "moves": moves,
            "mover": roles[MOVER], "captured": roles[CAPTURED], "castling_rook": roles[CASTLING_ROOK]}


def prepare(data_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=False)
    meta = pickle.loads((data_dir / "meta.pkl").read_bytes())
    manifest = {"version": 1, "identity_start_squares": list(INITIAL_SQUARES),
                "meta_sha256": file_hash(data_dir / "meta.pkl"), "splits": {}}
    for split in ("train", "val"):
        source = data_dir / f"{split}.bin"
        output = out_dir / f"{split}.npy"
        counts = prepare_split(source, output, meta)
        manifest["splits"][split] = {"source_sha256": file_hash(source),
                                      "labels_sha256": file_hash(output), "counts": counts}
        print(f"{split}: {counts}", flush=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_data = Path(__file__).resolve().parents[1] / "data" / os.getenv("DATASET_DIR", "processed")
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--out-dir", type=Path, default=os.getenv("PIECE_LABELS_DIR"))
    args = parser.parse_args()
    if args.out_dir is None:
        parser.error("provide --out-dir or PIECE_LABELS_DIR")
    prepare(args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()
