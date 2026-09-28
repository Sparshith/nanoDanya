# Runs the challenge page's checks on your laptop, then plays your model against the baseline.
#   python3 check.py                         checks model.onnx and plays 10 games
#   python3 check.py mine.onnx --games 0     checks only
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import chess
import numpy as np
import onnxruntime

HERE = Path(__file__).parent
LIMIT = 10_000_000
TEST_GAME = ("d4 Nf6 c4 g6 Nc3 Bg7 e3 O-O Nf3 d6 Bd3 Nbd7 O-O e5 dxe5 dxe5 e4 Re8 Bg5 c6 b3 Qc7 Bc2 Nf8 h3 Ne6 Bxf6 Bxf6 "
             "Qe2 Nd4 Qd3 Be6 Rad1 Rad8 Nxd4 exd4 Ne2 c5 f4 Bg7 f5 Bc8 Nf4 Bh6 Nd5 Qd6 fxg6 hxg6 Nf6+ Kg7 Nxe8+ Rxe8 Rf3 Re5 "
             "Rdf1 Be6 Kh1 Rh5 Qe2 Be3 Rxe3 dxe3 Qxe3").split()
OPENINGS = [
    ("Italian Game: Evans Gambit", "e4 e5 Nf3 Nc6 Bc4 Bc5 b4 Bxb4 c3 Bd6"),
    ("King's Gambit Accepted", "e4 e5 f4 exf4 Nf3 Nf6"),
    ("Sicilian: Smith-Morra Gambit", "e4 c5 d4 cxd4 c3 dxc3 Nxc3 Nc6 Nf3 e6 Bc4 Bb4"),
    ("Catalan: Open Defense", "d4 Nf6 c4 e6 g3 d5 Bg2 dxc4 Qa4 Nbd7 Qxc4"),
    ("Blackmar-Diemer Gambit", "d4 d5 e4 dxe4 Nc3 e5 dxe5"),
    ("Italian Game: McDonnell Defense", "e4 e5 Nf3 Nc6 Bc4 Bc5 b4 Bxb4 c3 Bc5 d4 exd4 O-O d6 cxd4 Bb6"),
    ("English: Symmetrical, Anti-Benoni", "c4 e6 Nf3 Nf6 Nc3 c5 d4 Nc6 g3 cxd4 Nxd4 Qb6"),
    ("Four Knights: Double Spanish", "e4 e5 Nf3 Nc6 Nc3 Nf6 Bb5 Bb4"),
    ("Italian Game: Giuoco Pianissimo", "e4 e5 Nf3 Nc6 Bc4 Bc5 O-O Nf6 d3 d6 c3 a6 Re1"),
    ("Italian Game: Alapin-Steinitz", "e4 e5 Nf3 Nc6 Bc4 Bc5 b4 Bxb4 c3 Ba5 O-O d6 d4 Bg4"),
]
strip = lambda san: re.sub(r"[+#]+$", "", san)


class Entry:
    def __init__(self, path):
        self.path = Path(path)
        self.session = onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        meta = self.session.get_modelmeta().custom_metadata_map
        if "vocabulary" not in meta:
            raise ValueError('no "vocabulary" in the ONNX metadata; export with export.py')
        self.vocabulary = json.loads(meta["vocabulary"])
        self.name = meta.get("name") or self.path.stem
        self.context = int(meta.get("context_length", 512))
        if not 64 <= self.context <= 4096:
            raise ValueError(f"context_length must be 64 to 4096, got {self.context}")
        self.ids = {token: i for i, token in enumerate(self.vocabulary)}
        if "<bos>" not in self.ids:
            raise ValueError("vocabulary has no <bos>")
        self.by_san = {}
        for i, token in enumerate(self.vocabulary):
            self.by_san.setdefault(strip(token), []).append(i)
        inputs = self.session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(f"model must have one input, has {len(inputs)}")
        self.input = inputs[0].name

    def encode(self, san):
        for i in (self.ids.get(san), self.ids.get(strip(san)), *self.by_san.get(strip(san), []), self.ids.get("<unk>")):
            if i is not None:
                return i

    def tokens(self, sans):
        out = [self.ids["<bos>"]]
        for san in sans:
            i = self.encode(san)
            if i is None:
                raise ValueError(f"{san} is not in this model's vocabulary")
            out.append(i)
        return out

    def logits(self, tokens):
        out = self.session.run(None, {self.input: np.asarray([tokens], dtype=np.int64)})[0]
        return out.reshape(-1, out.shape[-1])[-1]

    def move(self, board, sans):
        scores = self.logits(self.tokens(sans))
        best, best_score = None, -np.inf
        for move in board.legal_moves:
            for i in self.by_san.get(strip(board.san(move)), []):
                if scores[i] > best_score:
                    best, best_score = move, scores[i]
        return best


def check(path):
    size = Path(path).stat().st_size
    print(f"size      {size:,} bytes ({size / 1e6:.2f} MB) {'OK' if size <= LIMIT else 'FAIL: over 10 MB'}")
    if size > LIMIT:
        return None
    try:
        entry = Entry(path)
    except Exception as error:
        print(f"loads     FAIL: {error}")
        return None
    print(f"loads     OK: {entry.name}, {len(entry.vocabulary):,} tokens, context {entry.context}")
    for length in (0, 7, 31, 63):
        tokens = entry.tokens(TEST_GAME[:length])
        start = time.perf_counter()
        scores = entry.logits(tokens)
        ms = (time.perf_counter() - start) * 1000
        if scores.shape != (len(entry.vocabulary),) or not np.isfinite(scores).all():
            print(f"logits    FAIL at {len(tokens)} tokens: shape {scores.shape}, finite {bool(np.isfinite(scores).all())}")
            return None
        print(f"logits    OK at {len(tokens):>2} tokens, {ms:.0f} ms")
    return entry


def play(mine, baseline, games):
    score, moves, elapsed = 0.0, 0, 0.0
    for i in range(games):
        name, opening = OPENINGS[(i // 2) % len(OPENINGS)]
        white, black = (mine, baseline) if i % 2 == 0 else (baseline, mine)
        board, sans = chess.Board(), []
        for san in opening.split():
            sans.append(san)
            board.push_san(san)
        while not board.is_game_over(claim_draw=True) and len(sans) < 300:
            side = white if board.turn == chess.WHITE else black
            if len(sans) + 1 >= side.context:
                break
            start = time.perf_counter()
            move = side.move(board, sans)
            elapsed += time.perf_counter() - start
            moves += 1
            sans.append(board.san(move))
            board.push(move)
        result = board.result(claim_draw=True)
        points = 0.5 if result in ("1/2-1/2", "*") else float((result == "1-0") == (white is mine))
        score += points
        print(f"game {i + 1:>2}  {'W' if points == 1 else 'D' if points == 0.5 else 'L'}  {name}, {len(sans)} plies, {result}")
    print(f"\n{mine.name} vs {baseline.name}: {score:g} - {games - score:g} in {games} games, {elapsed / max(moves, 1) * 1000:.0f} ms per move")


parser = argparse.ArgumentParser()
parser.add_argument("model", nargs="?", default=HERE / "model.onnx")
parser.add_argument("--games", type=int, default=10)
parser.add_argument("--baseline", type=Path, default=HERE / "baseline.onnx")
args = parser.parse_args()

entry = check(args.model)
if entry is None:
    sys.exit(1)
if args.games:
    if not args.baseline.exists():
        print("\nExporting the baseline for the match …")
        subprocess.run([sys.executable, str(HERE / "export.py"), "--int4", "--output", str(args.baseline)], check=True)
    print()
    play(entry, Entry(args.baseline), args.games)
