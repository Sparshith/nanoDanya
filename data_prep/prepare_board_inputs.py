"""Expand validated move identities into causal, post-input board features."""
import json
import pickle
import re
from pathlib import Path

import chess
import numpy as np

from chess_inference import token_for_id
from data_prep.prepare_involved_pieces import CAPTURED, CASTLING_ROOK, INITIAL_SQUARES, file_hash
from inference.board_input import BoardState


def expand(tokens, involved, meta, output):
    bos, eos = meta["stoi"]["<bos>"], meta["stoi"]["<eos>"]
    destinations, promotions = {}, {}
    for san, token in meta["stoi"].items():
        match = re.search(r"([a-h][1-8])(?:=?([NBRQ]))?[+#]?$", san)
        if match:
            destinations[token] = chess.parse_square(match[1])
            if match[2]:
                promotions[token] = chess.Piece.from_symbol(match[2]).piece_type
    initial = np.array([(sq, chess.Board().piece_type_at(sq)) for sq in INITIAL_SQUARES], dtype=np.uint8)
    current = None
    for i, token in enumerate(tokens):
        if token == bos:
            current = initial.copy()
        elif token == eos:
            current = np.zeros((32, 2), dtype=np.uint8)
            current[:, 0] = 64
        else:
            mover, source, _ = involved[i, 0]
            if current is None or mover < 0 or current[mover, 0] != source:
                raise ValueError(f"Invalid mover source at token {i}")
            other, square, role = involved[i, 1]
            if role == CASTLING_ROOK:
                assert current[other, 0] == square
                rank = (int(source) // 8) * 8
                kingside = int(square) % 8 == 7
                current[mover, 0] = rank + (6 if kingside else 2)
                current[other, 0] = rank + (5 if kingside else 3)
            else:
                if role == CAPTURED:
                    assert current[other, 0] == square
                    current[other] = 64, 0
                current[mover, 0] = destinations[int(token)]
                if int(token) in promotions:
                    current[mover, 1] = promotions[int(token)]
        output[i] = current


def verify_replay(tokens, features, meta, max_games=None):
    starts = np.flatnonzero(tokens == meta["stoi"]["<bos>"])
    selected = np.arange(len(starts))
    if max_games is not None and len(starts) > max_games:
        selected = np.random.default_rng(42).choice(len(starts), max_games, replace=False)
    checked = 0
    for index in selected:
        start = starts[index]
        end = starts[index + 1] if index + 1 < len(starts) else len(tokens)
        state = BoardState()
        for i in range(start, end):
            expected = state.append(token_for_id(meta["itos"], int(tokens[i])))
            if not np.array_equal(features[i], expected):
                raise ValueError(f"Board features disagree with chess replay at token {i}")
            checked += 1
    return {"games": len(selected), "tokens": checked}


def prepare(data_dir, labels_dir, out_dir):
    data_dir, labels_dir, out_dir = map(Path, (data_dir, labels_dir, out_dir))
    meta = pickle.loads((data_dir / "meta.pkl").read_bytes())
    labels_manifest = json.loads((labels_dir / "manifest.json").read_text())
    if labels_manifest["meta_sha256"] != file_hash(data_dir / "meta.pkl"):
        raise ValueError("Tokenizer differs from piece-label metadata")
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"version": 1, "source_labels": labels_manifest,
                "features": "32 stable identities: square 0..63 or captured 64; current kind 1..6 or absent 0",
                "alignment": "board after current input token; BOS initial board; EOS blank", "splits": {}}
    for split in ("train", "val"):
        source, labels = data_dir / f"{split}.bin", labels_dir / f"{split}.npy"
        provenance = labels_manifest["splits"][split]
        assert file_hash(source) == provenance["source_sha256"]
        assert file_hash(labels) == provenance["labels_sha256"]
        tokens = np.memmap(source, dtype=np.uint16, mode="r")
        involved = np.load(labels, mmap_mode="r")
        path = out_dir / f"{split}.npy"
        features = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=(len(tokens), 32, 2))
        expand(tokens, involved, meta, features)
        features.flush()
        audit = verify_replay(tokens, features, meta, 500 if split == "train" else None)
        manifest["splits"][split] = {"features_sha256": file_hash(path), "replay_check": audit}
        print(json.dumps({"board_input_split": split, **manifest["splits"][split]}), flush=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
