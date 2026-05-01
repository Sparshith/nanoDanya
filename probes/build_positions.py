from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chess
import chess.engine
import numpy as np
import torch

from chess_token_utils import normalized_legal_sans, strip_san


def token_items(itos):
    if isinstance(itos, dict):
        return ((int(idx), token) for idx, token in itos.items())
    return enumerate(itos)


def legal_token_map(itos) -> dict[str, list[int]]:
    by_san: dict[str, list[int]] = {}
    for idx, token in token_items(itos):
        if token.startswith("<"):
            continue
        by_san.setdefault(strip_san(token), []).append(idx)
    return by_san


def piece_vector(board: chess.Board) -> np.ndarray:
    pieces = np.zeros(64, dtype=np.int64)
    for sq in range(64):
        piece = board.piece_at(sq)
        if piece is not None:
            pieces[sq] = piece.piece_type + (6 if piece.color == chess.BLACK else 0)
    return pieces


def legal_move_ids(board: chess.Board, san_to_ids: dict[str, list[int]]) -> list[int]:
    ids: list[int] = []
    for san in normalized_legal_sans(board):
        ids.extend(san_to_ids.get(san, []))
    return sorted(set(ids))


def stockfish_cp(engine: chess.engine.SimpleEngine, board: chess.Board, depth: int) -> float:
    if board.is_game_over():
        result = board.result()
        if result == "1-0":
            return 10000.0
        if result == "0-1":
            return -10000.0
        return 0.0
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    return float(info["score"].white().score(mate_score=10000))


def load_games(data_path: Path, bos_id: int, eos_id: int, max_games: int) -> list[list[int]]:
    raw = np.memmap(data_path, dtype=np.uint16, mode="r")
    starts = np.flatnonzero(raw == bos_id)
    games: list[list[int]] = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(raw)
        tokens = [int(t) for t in raw[start:end]]
        if tokens and tokens[-1] == eos_id:
            tokens = tokens[:-1]
        if len(tokens) >= 2:
            games.append(tokens)
        if max_games > 0 and len(games) >= max_games:
            break
    return games


def build_positions(args) -> dict:
    with open(args.meta, "rb") as f:
        meta = pickle.load(f)
    stoi = meta["stoi"]
    itos = meta["itos"]
    bos_id = int(stoi["<bos>"])
    eos_id = int(stoi["<eos>"])
    san_to_ids = legal_token_map(itos)
    games = load_games(Path(args.data), bos_id, eos_id, args.max_games)

    engine = None
    if args.stockfish:
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)

    prefixes: list[list[int]] = []
    fens: list[str] = []
    game_ids: list[int] = []
    plies: list[int] = []
    pieces: list[np.ndarray] = []
    side_to_move: list[float] = []
    in_check: list[float] = []
    legal_ids: list[list[int]] = []
    eval_cp: list[float] = []

    try:
        for game_id, tokens in enumerate(games):
            board = chess.Board()
            prefix = [tokens[0]]
            for ply in range(len(tokens)):
                prefixes.append(prefix.copy())
                fens.append(board.fen())
                game_ids.append(game_id)
                plies.append(ply)
                pieces.append(piece_vector(board))
                side_to_move.append(float(board.turn == chess.WHITE))
                in_check.append(float(board.is_check()))
                legal_ids.append(legal_move_ids(board, san_to_ids))
                if engine is not None:
                    eval_cp.append(stockfish_cp(engine, board, args.stockfish_depth))

                if ply + 1 >= len(tokens):
                    break
                token = itos[int(tokens[ply + 1])]
                try:
                    board.push_san(token)
                except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
                    break
                prefix.append(int(tokens[ply + 1]))

                if args.max_positions > 0 and len(prefixes) >= args.max_positions:
                    break
            if args.max_positions > 0 and len(prefixes) >= args.max_positions:
                break
    finally:
        if engine is not None:
            engine.quit()

    out = {
        "prefixes": prefixes,
        "fen": fens,
        "game_id": torch.tensor(game_ids, dtype=torch.long),
        "ply": torch.tensor(plies, dtype=torch.long),
        "pieces": torch.from_numpy(np.stack(pieces)).long(),
        "side_to_move": torch.tensor(side_to_move, dtype=torch.float32).unsqueeze(1),
        "in_check": torch.tensor(in_check, dtype=torch.float32).unsqueeze(1),
        "legal_move_ids": legal_ids,
        "eval_cp": torch.tensor(eval_cp, dtype=torch.float32) if eval_cp else None,
        "vocab_size": int(meta["vocab_size"]),
        "source": {
            "data": str(args.data),
            "meta": str(args.meta),
            "max_games": args.max_games,
            "stockfish": args.stockfish,
            "stockfish_depth": args.stockfish_depth if args.stockfish else None,
        },
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build labeled positions for linear probes.")
    parser.add_argument("--data", default="data/processed/val.bin")
    parser.add_argument("--meta", default="data/processed/meta.pkl")
    parser.add_argument("--out", default="data/probes/positions_val_small.pt")
    parser.add_argument("--max-games", type=int, default=100)
    parser.add_argument("--max-positions", type=int, default=0)
    parser.add_argument("--stockfish", default="")
    parser.add_argument("--stockfish-depth", type=int, default=8)
    args = parser.parse_args()

    positions = build_positions(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(positions, out_path)
    print(f"wrote {out_path} ({len(positions['prefixes'])} positions)")


if __name__ == "__main__":
    main()
