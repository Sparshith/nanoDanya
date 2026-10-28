from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chess
import chess.pgn
import torch
import torch.nn.functional as F

from chess_token_utils import resolve_token_id
from probes.build_positions import piece_vector
from probes.cache_hiddens import cache_hiddens


def game_position(pgn: str, stoi: dict[str, int], ply: int) -> dict:
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None or game.errors:
        raise ValueError("PGN must contain a valid game")
    if game.headers.get("SetUp") == "1":
        raise ValueError("Probing requires move history from the initial position")
    moves = list(game.mainline_moves())
    if not 0 <= ply <= len(moves):
        raise ValueError(f"Ply must be between 0 and {len(moves)}")
    board = game.board()
    prefix = [stoi["<bos>"]]
    history = []
    for move in moves[:ply]:
        san = board.san(move)
        token = resolve_token_id(stoi, san)
        if token is None:
            raise ValueError(f"Move {san} at ply {len(history) + 1} is absent from tokenizer")
        history.append(san)
        prefix.append(token)
        board.push(move)
    return {
        "prefixes": [prefix], "fen": [board.fen()], "history": history,
        "ply": torch.tensor([ply]), "pieces": torch.from_numpy(piece_vector(board))[None],
        "vocab_size": len(stoi), "stoi": stoi,
        "next_move": board.san(moves[ply]) if ply < len(moves) else None,
    }


def piece_symbol(label: int) -> str:
    return ".PNBRQKpnbrqk"[label]


def board_text(labels: list[int]) -> str:
    return "\n".join(
        f"{rank + 1} " + " ".join(piece_symbol(labels[chess.square(file, rank)]) for file in range(8))
        for rank in range(7, -1, -1)
    ) + "\n  a b c d e f g h"


def decode_position(positions: dict, cached: dict, heads_dir: Path) -> dict:
    truth = positions["pieces"][0]
    report = {
        "model": cached["model"], "ply": int(positions["ply"][0]),
        "fen": positions["fen"][0], "next_move": positions["next_move"],
        "history": positions["history"], "actual_board": board_text(truth.tolist()),
        "layers": [],
    }
    for layer, h in cached["hiddens"].items():
        saved = torch.load(heads_dir / f"pieces_layer_{layer}.pt", map_location="cpu", weights_only=False)
        if saved["model"] != cached["model"] or saved["layer"] != layer:
            raise ValueError("Readout and hidden-state model/layer do not match")
        state = saved["state_dict"]
        probabilities = F.linear(h, state["weight"], state["bias"]).view(64, 13).softmax(-1)
        pred = probabilities.argmax(-1)
        occupied = truth != 0
        squares = []
        for sq in chess.SQUARES:
            values, classes = probabilities[sq].topk(3)
            squares.append({
                "square": chess.square_name(sq), "actual": piece_symbol(int(truth[sq])),
                "predicted": piece_symbol(int(pred[sq])),
                "actual_probability": float(probabilities[sq, truth[sq]]),
                "top3": [{"piece": piece_symbol(int(c)), "probability": float(p)}
                         for c, p in zip(classes, values)],
            })
        report["layers"].append({
            "layer": layer, "validation": saved["metrics"],
            "correct_squares": int((pred == truth).sum()),
            "correct_occupied": int((pred[occupied] == truth[occupied]).sum()),
            "occupied_squares": int(occupied.sum()),
            "predicted_board": board_text(pred.tolist()), "squares": squares,
        })
    report["selected_layer"] = max(report["layers"], key=lambda row: row["validation"]["occupied_acc"])["layer"]
    return report


def inspect_game(model: str, pgn: str, ply: int, heads_dir: Path, out_dir: Path) -> dict:
    checkpoint = torch.load(model, map_location="cpu", weights_only=False)
    stoi = checkpoint["meta"]["tokenizer"]["stoi"]
    step = checkpoint.get("step")
    del checkpoint
    positions = game_position(pgn, stoi, ply)
    out_dir.mkdir(parents=True, exist_ok=True)
    positions_path = out_dir / "game_position.pt"
    torch.save(positions, positions_path)
    layers = sorted(int(p.stem.removeprefix("pieces_layer_")) for p in heads_dir.glob("pieces_layer_*.pt"))
    if not layers:
        raise ValueError(f"No saved board readouts in {heads_dir}")
    cached = cache_hiddens(SimpleNamespace(
        model=model, positions=str(positions_path), layers=",".join(map(str, layers)), batch_size=1,
    ))
    report = decode_position(positions, cached, heads_dir)
    report["checkpoint_step"] = step
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def inspect_timeline(model: str, pgn: str, plies: list[int], heads_dir: Path, out_dir: Path) -> list[dict]:
    checkpoint = torch.load(model, map_location="cpu", weights_only=False)
    stoi = checkpoint["meta"]["tokenizer"]["stoi"]
    step = checkpoint.get("step")
    del checkpoint
    positions = [game_position(pgn, stoi, ply) for ply in plies]
    out_dir.mkdir(parents=True, exist_ok=True)
    positions_path = out_dir / "timeline_positions.pt"
    torch.save({
        "prefixes": [p["prefixes"][0] for p in positions],
        "stoi": stoi, "vocab_size": len(stoi),
    }, positions_path)
    layers = sorted(int(p.stem.removeprefix("pieces_layer_")) for p in heads_dir.glob("pieces_layer_*.pt"))
    if not layers:
        raise ValueError(f"No saved board readouts in {heads_dir}")
    cached = cache_hiddens(SimpleNamespace(
        model=model, positions=str(positions_path), layers=",".join(map(str, layers)), batch_size=16,
    ))
    reports = []
    for row, position in enumerate(positions):
        single = {"model": model, "hiddens": {layer: h[row:row + 1] for layer, h in cached["hiddens"].items()}}
        report = decode_position(position, single, heads_dir)
        report["checkpoint_step"] = step
        reports.append(report)
    (out_dir / "timeline.json").write_text(json.dumps(reports, indent=2) + "\n")
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode the board after N plies, before the next move.")
    parser.add_argument("--model", required=True, help="Checkpoint file used to train the readouts")
    parser.add_argument("--pgn", required=True)
    parser.add_argument("--ply", type=int, required=True)
    parser.add_argument("--heads", required=True)
    parser.add_argument("--out", default="data/probes/game")
    args = parser.parse_args()
    report = inspect_game(args.model, Path(args.pgn).read_text(), args.ply, Path(args.heads), Path(args.out))
    selected = next(row for row in report["layers"] if row["layer"] == report["selected_layer"])
    print(f"Before {report['next_move']}, layer {selected['layer']} selected by held-out occupied accuracy")
    print("Actual:\n" + report["actual_board"])
    print("Decoded:\n" + selected["predicted_board"])
    print(f"Report: {args.out}/report.json")


if __name__ == "__main__":
    main()
