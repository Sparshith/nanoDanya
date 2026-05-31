from __future__ import annotations

import argparse
import heapq
import json
import pickle
import random
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chess
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class HardStats:
    scanned_games: int = 0
    valid_games: int = 0
    scanned_positions: int = 0
    hard_positions: int = 0
    invalid_positions: int = 0
    in_check: int = 0
    low_legal_count: int = 0
    pinned_piece: int = 0
    unsafe_king_capture: int = 0
    late_position: int = 0
    slider_pressure: int = 0
    target_gives_check: int = 0
    accepted_examples: int = 0


RAY_DIRECTIONS = (
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)

FEATURE_BITS = {
    "in_check": 1 << 0,
    "low_legal_count": 1 << 1,
    "pinned_piece": 1 << 2,
    "unsafe_king_capture": 1 << 3,
    "late_position": 1 << 4,
    "slider_pressure": 1 << 5,
    "target_gives_check": 1 << 6,
}


def token_for_id(itos: list[str] | dict[int | str, str], idx: int) -> str:
    if isinstance(itos, dict):
        return itos.get(idx, itos.get(str(idx), ""))
    return itos[idx]


def resolve_path(path: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw
    return (PROJECT_ROOT / raw).resolve()


def load_meta(data_dir: Path) -> dict[str, Any]:
    return pickle.loads((data_dir / "meta.pkl").read_bytes())


def iter_token_games(
    data_path: Path,
    *,
    bos_id: int,
    eos_id: int,
    max_games: int,
):
    ids = np.fromfile(data_path, dtype=np.uint16)
    current: list[int] = []
    yielded = 0

    for raw_id in ids:
        token_id = int(raw_id)
        if token_id == bos_id:
            current = [token_id]
            continue
        if not current:
            continue
        current.append(token_id)
        if token_id == eos_id:
            yield current
            yielded += 1
            current = []
            if max_games > 0 and yielded >= max_games:
                return


def legal_count_bucket(legal_count: int) -> str:
    if legal_count <= 1:
        return "le_1"
    if legal_count <= 3:
        return "le_3"
    if legal_count <= 6:
        return "le_6"
    if legal_count <= 10:
        return "le_10"
    return "gt_10"


def phase_for_ply(ply: int) -> str:
    if ply <= 20:
        return "opening"
    if ply <= 80:
        return "middlegame"
    return "endgame"


def pinned_squares(board: chess.Board, color: chess.Color) -> list[int]:
    return [
        square
        for square, piece in board.piece_map().items()
        if piece.color == color and piece.piece_type != chess.KING and board.is_pinned(color, square)
    ]


def unsafe_king_captures(board: chess.Board, legal_moves: set[chess.Move]) -> int:
    king_square = board.king(board.turn)
    if king_square is None:
        return 0

    count = 0
    for move in board.generate_pseudo_legal_moves(from_mask=chess.BB_SQUARES[king_square]):
        captured = board.piece_at(move.to_square)
        if captured is None or captured.color == board.turn:
            continue
        if move not in legal_moves:
            count += 1
    return count


def compatible_slider(piece_type: int, df: int, dr: int) -> bool:
    diagonal = abs(df) == abs(dr)
    straight = df == 0 or dr == 0
    if piece_type == chess.QUEEN:
        return diagonal or straight
    if piece_type == chess.ROOK:
        return straight
    if piece_type == chess.BISHOP:
        return diagonal
    return False


def slider_pressure_to_king(board: chess.Board, color: chess.Color) -> int:
    king_square = board.king(color)
    if king_square is None:
        return 0

    king_file = chess.square_file(king_square)
    king_rank = chess.square_rank(king_square)
    pressure = 0

    for df, dr in RAY_DIRECTIONS:
        file = king_file + df
        rank = king_rank + dr
        seen_own_blocker = False
        while 0 <= file < 8 and 0 <= rank < 8:
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece is None:
                file += df
                rank += dr
                continue

            if piece.color == color:
                if seen_own_blocker:
                    break
                seen_own_blocker = True
                file += df
                rank += dr
                continue

            if compatible_slider(piece.piece_type, df, dr):
                pressure += 1
            break

    return pressure


def score_position(
    board: chess.Board,
    target_move: chess.Move,
    *,
    ply: int,
    late_ply: int,
    low_legal_threshold: int,
) -> tuple[int, dict[str, Any]]:
    legal_moves_list = list(board.legal_moves)
    legal_moves = set(legal_moves_list)
    legal_count = len(legal_moves_list)
    pinned = pinned_squares(board, board.turn)
    unsafe_captures = unsafe_king_captures(board, legal_moves)
    slider_pressure = slider_pressure_to_king(board, board.turn)
    in_check = board.is_check()
    late_position = ply > late_ply
    target_gives_check = board.gives_check(target_move)

    score = 0
    if in_check:
        score += 6
    if legal_count <= 1:
        score += 6
    elif legal_count <= 3:
        score += 5
    elif legal_count <= 6:
        score += 4
    elif legal_count <= low_legal_threshold:
        score += 2
    score += min(len(pinned) * 3, 9)
    score += min(unsafe_captures * 3, 6)
    if late_position:
        score += 2 if ply > 80 else 1
    score += min(slider_pressure * 2, 4)
    if target_gives_check:
        score += 1

    features = {
        "in_check": in_check,
        "legal_count": legal_count,
        "legal_bucket": legal_count_bucket(legal_count),
        "pinned_count": len(pinned),
        "unsafe_king_captures": unsafe_captures,
        "late_position": late_position,
        "slider_pressure": slider_pressure,
        "target_gives_check": target_gives_check,
        "phase": phase_for_ply(ply),
    }
    return score, features


def feature_mask(features: dict[str, Any], *, low_legal_threshold: int) -> int:
    mask = 0
    if features["in_check"]:
        mask |= FEATURE_BITS["in_check"]
    if features["legal_count"] <= low_legal_threshold:
        mask |= FEATURE_BITS["low_legal_count"]
    if features["pinned_count"] > 0:
        mask |= FEATURE_BITS["pinned_piece"]
    if features["unsafe_king_captures"] > 0:
        mask |= FEATURE_BITS["unsafe_king_capture"]
    if features["late_position"]:
        mask |= FEATURE_BITS["late_position"]
    if features["slider_pressure"] > 0:
        mask |= FEATURE_BITS["slider_pressure"]
    if features["target_gives_check"]:
        mask |= FEATURE_BITS["target_gives_check"]
    return mask


class WeightedReservoir:
    def __init__(self, *, capacity: int, context_length: int, pad_id: int, seed: int):
        self.capacity = capacity
        self.context_length = context_length
        self.pad_id = pad_id
        self.rng = random.Random(seed)
        self.prefixes = np.full((capacity, context_length), pad_id, dtype=np.uint16)
        self.lengths = np.zeros(capacity, dtype=np.uint16)
        self.targets = np.zeros(capacity, dtype=np.uint16)
        self.scores = np.zeros(capacity, dtype=np.uint16)
        self.feature_masks = np.zeros(capacity, dtype=np.uint16)
        self.legal_counts = np.zeros(capacity, dtype=np.uint16)
        self.plies = np.zeros(capacity, dtype=np.uint16)
        self.heap: list[tuple[float, int]] = []
        self.size = 0

    def add(self, *, prefix: list[int], target: int, score: int, feature_mask: int, legal_count: int, ply: int) -> bool:
        if self.capacity <= 0:
            return False

        weight = max(score, 1)
        key = self.rng.random() ** (1.0 / weight)
        if self.size < self.capacity:
            slot = self.size
            self.size += 1
            heapq.heappush(self.heap, (key, slot))
        elif key > self.heap[0][0]:
            _, slot = heapq.heapreplace(self.heap, (key, self.heap[0][1]))
        else:
            return False

        length = min(len(prefix), self.context_length)
        row = self.prefixes[slot]
        row.fill(self.pad_id)
        row[:length] = np.asarray(prefix[-length:], dtype=np.uint16)
        self.lengths[slot] = length
        self.targets[slot] = target
        self.scores[slot] = score
        self.feature_masks[slot] = feature_mask
        self.legal_counts[slot] = min(legal_count, np.iinfo(np.uint16).max)
        self.plies[slot] = min(ply, np.iinfo(np.uint16).max)
        return True

    def write(self, out_dir: Path, split: str) -> dict[str, int]:
        prefix_path = out_dir / f"{split}_hard_prefixes.bin"
        length_path = out_dir / f"{split}_hard_lengths.bin"
        target_path = out_dir / f"{split}_hard_targets.bin"
        score_path = out_dir / f"{split}_hard_scores.bin"
        mask_path = out_dir / f"{split}_hard_feature_masks.bin"
        legal_count_path = out_dir / f"{split}_hard_legal_counts.bin"
        ply_path = out_dir / f"{split}_hard_plies.bin"

        prefixes = self.prefixes[: self.size]
        lengths = self.lengths[: self.size]
        targets = self.targets[: self.size]
        scores = self.scores[: self.size]
        feature_masks = self.feature_masks[: self.size]
        legal_counts = self.legal_counts[: self.size]
        plies = self.plies[: self.size]

        prefixes.tofile(prefix_path)
        lengths.tofile(length_path)
        targets.tofile(target_path)
        scores.tofile(score_path)
        feature_masks.tofile(mask_path)
        legal_counts.tofile(legal_count_path)
        plies.tofile(ply_path)

        return {
            "examples": self.size,
            "prefix_tokens": int(prefixes.size),
            "prefix_bytes": int(prefixes.nbytes),
            "length_bytes": int(lengths.nbytes),
            "target_bytes": int(targets.nbytes),
            "score_bytes": int(scores.nbytes),
            "feature_mask_bytes": int(feature_masks.nbytes),
            "legal_count_bytes": int(legal_counts.nbytes),
            "ply_bytes": int(plies.nbytes),
        }


def collect_split(
    *,
    data_dir: Path,
    split: str,
    meta: dict[str, Any],
    max_games: int,
    max_examples: int,
    min_score: int,
    low_legal_threshold: int,
    late_ply: int,
    seed: int,
) -> tuple[WeightedReservoir, HardStats, dict[str, Counter]]:
    stoi = meta["stoi"]
    itos = meta["itos"]
    context_length = int(meta["context_length"])
    bos_id = int(stoi["<bos>"])
    eos_id = int(stoi["<eos>"])

    reservoir = WeightedReservoir(
        capacity=max_examples,
        context_length=context_length,
        pad_id=eos_id,
        seed=seed,
    )
    stats = HardStats()
    counters = {
        "score": Counter(),
        "phase": Counter(),
        "legal_bucket": Counter(),
    }

    for game in iter_token_games(data_dir / f"{split}.bin", bos_id=bos_id, eos_id=eos_id, max_games=max_games):
        stats.scanned_games += 1
        board = chess.Board()
        prefix = deque([bos_id], maxlen=context_length)
        valid_game = True

        for ply, target_id in enumerate(game[1:], start=1):
            if target_id == eos_id:
                break

            token = token_for_id(itos, int(target_id))
            try:
                target_move = board.parse_san(token)
            except ValueError:
                stats.invalid_positions += 1
                valid_game = False
                break

            stats.scanned_positions += 1
            score, features = score_position(
                board,
                target_move,
                ply=ply,
                late_ply=late_ply,
                low_legal_threshold=low_legal_threshold,
            )
            hard = (
                score >= min_score
                or features["in_check"]
                or features["pinned_count"] > 0
                or features["unsafe_king_captures"] > 0
                or features["legal_count"] <= low_legal_threshold
            )

            if hard:
                stats.hard_positions += 1
                stats.in_check += int(features["in_check"])
                stats.low_legal_count += int(features["legal_count"] <= low_legal_threshold)
                stats.pinned_piece += int(features["pinned_count"] > 0)
                stats.unsafe_king_capture += int(features["unsafe_king_captures"] > 0)
                stats.late_position += int(features["late_position"])
                stats.slider_pressure += int(features["slider_pressure"] > 0)
                stats.target_gives_check += int(features["target_gives_check"])
                counters["score"][score] += 1
                counters["phase"][features["phase"]] += 1
                counters["legal_bucket"][features["legal_bucket"]] += 1
                mask = feature_mask(features, low_legal_threshold=low_legal_threshold)
                accepted = reservoir.add(
                    prefix=list(prefix),
                    target=int(target_id),
                    score=score,
                    feature_mask=mask,
                    legal_count=int(features["legal_count"]),
                    ply=ply,
                )
                stats.accepted_examples += int(accepted)

            board.push(target_move)
            prefix.append(int(target_id))

        stats.valid_games += int(valid_game)

    return reservoir, stats, counters


def counter_to_json(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build targeted king-safety SFT examples from the actual_5m tokenized corpus."
    )
    parser.add_argument("--data-dir", default="data/actual_5m", help="Source tokenized actual-game dataset.")
    parser.add_argument("--out-dir", default="data/king_safety_sft", help="Output directory.")
    parser.add_argument("--max-games-train", type=int, default=0, help="Optional cap on train games; 0 scans all.")
    parser.add_argument("--max-games-val", type=int, default=0, help="Optional cap on val games; 0 scans all.")
    parser.add_argument("--max-hard-train", type=int, default=300_000)
    parser.add_argument("--max-hard-val", type=int, default=20_000)
    parser.add_argument("--min-score", type=int, default=4)
    parser.add_argument("--low-legal-threshold", type=int, default=10)
    parser.add_argument("--late-ply", type=int, default=60)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    data_dir = resolve_path(args.data_dir)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = load_meta(data_dir)
    context_length = int(meta["context_length"])
    if context_length > np.iinfo(np.uint16).max:
        raise ValueError(f"context_length too large for uint16 lengths: {context_length}")

    print(f"source: {data_dir}")
    print(f"output: {out_dir}")
    print(
        f"config: context_length={context_length} min_score={args.min_score} "
        f"low_legal_threshold={args.low_legal_threshold} late_ply={args.late_ply}"
    )

    split_summaries = {}
    for split, max_games, max_examples, seed in (
        ("train", args.max_games_train, args.max_hard_train, args.seed),
        ("val", args.max_games_val, args.max_hard_val, args.seed + 1),
    ):
        print(f"collecting {split}: max_games={max_games} max_examples={max_examples}")
        reservoir, stats, counters = collect_split(
            data_dir=data_dir,
            split=split,
            meta=meta,
            max_games=max_games,
            max_examples=max_examples,
            min_score=args.min_score,
            low_legal_threshold=args.low_legal_threshold,
            late_ply=args.late_ply,
            seed=seed,
        )
        write_info = reservoir.write(out_dir, split)
        split_summaries[split] = {
            "stats": asdict(stats),
            "files": write_info,
            "counters": {name: counter_to_json(counter) for name, counter in counters.items()},
        }
        print(
            f"{split}: scanned_positions={stats.scanned_positions:,} "
            f"hard_positions={stats.hard_positions:,} examples={reservoir.size:,}"
        )

    with (out_dir / "meta.pkl").open("wb") as f:
        pickle.dump(meta, f)

    summary = {
        "type": "king_safety_sft_data",
        "source_data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "format": {
            "prefixes": "<split>_hard_prefixes.bin uint16, shape [examples, context_length], right-padded with <eos>",
            "lengths": "<split>_hard_lengths.bin uint16, prefix lengths before right padding",
            "targets": "<split>_hard_targets.bin uint16, actual next SAN token id",
            "scores": "<split>_hard_scores.bin uint16, heuristic hardness score",
            "feature_masks": "<split>_hard_feature_masks.bin uint16, bitmask of king-safety features",
            "legal_counts": "<split>_hard_legal_counts.bin uint16, legal move count at the target position",
            "plies": "<split>_hard_plies.bin uint16, target ply in the source game",
            "feature_bits": FEATURE_BITS,
        },
        "config": {
            "max_games_train": args.max_games_train,
            "max_games_val": args.max_games_val,
            "max_hard_train": args.max_hard_train,
            "max_hard_val": args.max_hard_val,
            "min_score": args.min_score,
            "low_legal_threshold": args.low_legal_threshold,
            "late_ply": args.late_ply,
            "seed": args.seed,
            "context_length": context_length,
        },
        "splits": split_summaries,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
