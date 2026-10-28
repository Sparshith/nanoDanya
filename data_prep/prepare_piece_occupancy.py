"""Living identities for collision loss; full validation locations for scoring."""
import json
import pickle
import re

import chess
import numpy as np

from data_prep.prepare_involved_pieces import CAPTURED, CASTLING_ROOK, INITIAL_SQUARES, file_hash, label_move


def prepare(data_dir, labels_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = pickle.loads((data_dir / "meta.pkl").read_bytes())
    original = json.loads((labels_dir / "manifest.json").read_text())
    manifest = {"version": 1, "piece_labels": original, "splits": {}}
    bos, eos = meta["stoi"]["<bos>"], meta["stoi"]["<eos>"]
    for split in ("train", "val"):
        tokens = np.memmap(data_dir / f"{split}.bin", dtype=np.uint16, mode="r")
        involved = np.load(labels_dir / f"{split}.npy", mmap_mode="r")
        assert file_hash(data_dir / f"{split}.bin") == original["splits"][split]["source_sha256"]
        assert file_hash(labels_dir / f"{split}.npy") == original["splits"][split]["labels_sha256"]
        output = out_dir / f"{split}_alive.npy"
        alive = np.lib.format.open_memmap(output, mode="w+", dtype=np.uint32, shape=(len(tokens),))
        alive[:] = 0
        current = 0
        for i, token in enumerate(tokens):
            if token == bos:
                current = (1 << 32) - 1
            elif token == eos:
                current = 0
            else:
                alive[i] = current
                if involved[i, 1, 2] == CAPTURED:
                    current &= ~(1 << int(involved[i, 1, 0]))
        alive.flush()
        manifest["splits"][split] = {"alive_sha256": file_hash(output)}
        print(f"{split}: living masks ready", flush=True)
    output = out_dir / "val_locations.npy"
    locations = np.lib.format.open_memmap(output, mode="w+", dtype=np.int8, shape=(len(tokens), 32))
    locations[:] = -1
    itos = meta["itos"]
    for i, token in enumerate(tokens):
        token = int(token)
        if token == bos:
            board = chess.Board()
            identities = dict(zip(INITIAL_SQUARES, range(32)))
        elif token != eos:
            for square, identity in identities.items():
                locations[i, identity] = square
            san = itos[token] if not isinstance(itos, dict) or token in itos else itos[str(token)]
            label_move(board, identities, board.parse_san(san))
    locations.flush()
    manifest["splits"]["val"]["locations_sha256"] = file_hash(output)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare_training_locations(data_dir, labels_dir, out_dir):
    """Expand validated move labels into pre-move locations for every identity."""
    meta = pickle.loads((data_dir / "meta.pkl").read_bytes())
    manifest = json.loads((labels_dir / "manifest.json").read_text())
    tokens = np.asarray(np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r"))
    involved = np.asarray(np.load(labels_dir / "train.npy", mmap_mode="r"))
    assert file_hash(data_dir / "train.bin") == manifest["splits"]["train"]["source_sha256"]
    assert file_hash(labels_dir / "train.npy") == manifest["splits"]["train"]["labels_sha256"]
    destinations = {}
    for san, token in meta["stoi"].items():
        match = re.search(r"([a-h][1-8])(?:=?[NBRQ])?[+#]?$", san)
        if match:
            destinations[token] = chess.parse_square(match[1])
    output = out_dir / "train_locations.npy"
    mapped = np.lib.format.open_memmap(output, mode="w+", dtype=np.int8, shape=(len(tokens), 32))
    locations = np.asarray(mapped)
    locations[:] = -1
    bos, eos = meta["stoi"]["<bos>"], meta["stoi"]["<eos>"]
    for i, token in enumerate(tokens):
        if token == bos:
            current = np.array(INITIAL_SQUARES, dtype=np.int8)
        elif token != eos:
            locations[i] = current
            mover, source, _ = involved[i, 0]
            assert current[mover] == source
            other, square, role = involved[i, 1]
            if role == CASTLING_ROOK:
                assert current[other] == square
                rank = (int(source) // 8) * 8
                kingside = int(square) % 8 == 7
                current[mover] = rank + (6 if kingside else 2)
                current[other] = rank + (5 if kingside else 3)
            else:
                if role == CAPTURED:
                    assert current[other] == square
                    current[other] = -1
                current[mover] = destinations[int(token)]
        if (i + 1) % 1_000_000 == 0:
            print(f"All-piece locations: {i + 1:,}/{len(tokens):,}", flush=True)
    mapped.flush()
    result = {"piece_labels": manifest, "locations_sha256": file_hash(output), "tokens": len(tokens)}
    (out_dir / "train_locations.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
