from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import chess
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
NANOCHAT_ROOT = PROJECT_ROOT / "nanochat"
if str(NANOCHAT_ROOT) not in sys.path:
    sys.path.insert(0, str(NANOCHAT_ROOT))

from chess_inference import legal_token_ids, token_for_id
from chess_token_utils import (
    normalized_legal_sans,
    strip_san,
    token_is_legal_prediction,
    under_disambiguated_legal_matches,
)
from model_loading import load_model


FEATURE_BITS = {
    "in_check": 1 << 0,
    "low_legal_count": 1 << 1,
    "pinned_piece": 1 << 2,
    "unsafe_king_capture": 1 << 3,
    "late_position": 1 << 4,
    "slider_pressure": 1 << 5,
    "target_gives_check": 1 << 6,
}


@dataclass
class Position:
    board: chess.Board
    prefix_ids: list[int]
    length: int
    target_id: int
    feature_mask: int
    legal_count: int
    ply: int
    index: int


@dataclass
class Stats:
    positions: int = 0
    exact_illegal: int = 0
    repaired_illegal: int = 0
    under_disambiguated: int = 0
    legal_mass_sum: float = 0.0
    target_top1: int = 0

    def add(
        self,
        *,
        exact_legal: bool,
        repaired_legal: bool,
        under_disambiguated: bool,
        legal_mass: float,
        target_is_top1: bool,
    ) -> None:
        self.positions += 1
        self.exact_illegal += 0 if exact_legal else 1
        self.repaired_illegal += 0 if repaired_legal else 1
        self.under_disambiguated += 1 if under_disambiguated else 0
        self.legal_mass_sum += legal_mass
        self.target_top1 += 1 if target_is_top1 else 0

    def as_dict(self) -> dict[str, Any]:
        if self.positions == 0:
            return {
                "positions": 0,
                "exact_illegal_rate": None,
                "notation_repaired_illegal_rate": None,
                "real_illegal_rate": None,
                "avg_legal_mass": None,
                "target_top1_rate": None,
            }
        real_illegal = self.repaired_illegal - self.under_disambiguated
        return {
            "positions": self.positions,
            "exact_illegal": self.exact_illegal,
            "notation_repaired_illegal": self.repaired_illegal,
            "under_disambiguated": self.under_disambiguated,
            "real_illegal": real_illegal,
            "exact_illegal_rate": self.exact_illegal / self.positions,
            "notation_repaired_illegal_rate": self.repaired_illegal / self.positions,
            "real_illegal_rate": real_illegal / self.positions,
            "avg_legal_mass": self.legal_mass_sum / self.positions,
            "target_top1_rate": self.target_top1 / self.positions,
        }


def resolve_path(path: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw
    return (PROJECT_ROOT / raw).resolve()


def resolve_device(device_arg: str) -> str:
    if device_arg:
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_meta(data_dir: Path) -> dict[str, Any]:
    return pickle.loads((data_dir / "meta.pkl").read_bytes())


def feature_names(mask: int) -> list[str]:
    return [name for name, bit in FEATURE_BITS.items() if mask & bit]


def board_from_prefix(prefix_ids: list[int], itos) -> chess.Board | None:
    board = chess.Board()
    for token_id in prefix_ids[1:]:
        token = token_for_id(itos, token_id)
        if token == "<eos>":
            continue
        try:
            board.push_san(token)
        except ValueError:
            return None
    return board


def load_positions(
    *,
    data_dir: Path,
    split: str,
    max_positions: int,
) -> tuple[list[Position], dict[str, Any], dict[str, int]]:
    meta = load_meta(data_dir)
    context_length = int(meta["context_length"])
    prefixes = np.fromfile(data_dir / f"{split}_hard_prefixes.bin", dtype=np.uint16)
    lengths = np.fromfile(data_dir / f"{split}_hard_lengths.bin", dtype=np.uint16)
    targets = np.fromfile(data_dir / f"{split}_hard_targets.bin", dtype=np.uint16)
    masks = np.fromfile(data_dir / f"{split}_hard_feature_masks.bin", dtype=np.uint16)
    legal_counts = np.fromfile(data_dir / f"{split}_hard_legal_counts.bin", dtype=np.uint16)
    plies = np.fromfile(data_dir / f"{split}_hard_plies.bin", dtype=np.uint16)

    examples = len(targets)
    if examples == 0:
        raise ValueError(f"{split}: no hard examples")
    if prefixes.size != examples * context_length:
        raise ValueError(f"{split}: prefix shape mismatch")
    prefixes = prefixes.reshape(examples, context_length)

    limit = examples if max_positions <= 0 else min(max_positions, examples)
    positions: list[Position] = []
    skipped = 0
    for idx in range(limit):
        length = int(lengths[idx])
        prefix_ids = [int(x) for x in prefixes[idx, :length]]
        board = board_from_prefix(prefix_ids, meta["itos"])
        if board is None:
            skipped += 1
            continue
        positions.append(
            Position(
                board=board,
                prefix_ids=prefix_ids,
                length=length,
                target_id=int(targets[idx]),
                feature_mask=int(masks[idx]),
                legal_count=int(legal_counts[idx]),
                ply=int(plies[idx]),
                index=idx,
            )
        )
    return positions, meta, {"requested": limit, "skipped_invalid_prefixes": skipped}


def last_logits_for_prefixes(
    model,
    prefixes: list[list[int]],
    *,
    pad_id: int,
    device: str,
    batch_size: int,
) -> Iterable[tuple[int, torch.Tensor]]:
    for start in range(0, len(prefixes), batch_size):
        batch = prefixes[start : start + batch_size]
        max_len = max(len(prefix) for prefix in batch)
        x = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        last_idx = []
        for row, prefix in enumerate(batch):
            x[row, : len(prefix)] = torch.tensor(prefix, dtype=torch.long, device=device)
            last_idx.append(len(prefix) - 1)
        logits = model(x)
        for row, idx in enumerate(last_idx):
            yield start + row, logits[row, idx, :].detach()


@torch.inference_mode()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    positions, data_meta, load_info = load_positions(
        data_dir=resolve_path(args.hard_data_dir),
        split=args.split,
        max_positions=args.max_positions,
    )
    model, _, stoi, itos = load_model(args.model, device)
    if stoi["<bos>"] != data_meta["stoi"]["<bos>"] or stoi["<eos>"] != data_meta["stoi"]["<eos>"]:
        raise ValueError("model tokenizer does not match hard dataset tokenizer")

    overall = Stats()
    by_feature: dict[str, Stats] = defaultdict(Stats)
    by_legal_bucket: dict[str, Stats] = defaultdict(Stats)
    top_illegal = Counter()
    failures = []

    prefixes = [pos.prefix_ids for pos in positions]
    for idx, logits in last_logits_for_prefixes(
        model,
        prefixes,
        pad_id=stoi["<eos>"],
        device=device,
        batch_size=args.batch_size,
    ):
        pos = positions[idx]
        raw_top1_idx = int(torch.argmax(logits).item())
        raw_top1 = token_for_id(itos, raw_top1_idx)
        target_token = token_for_id(itos, pos.target_id)
        exact_legal = raw_top1 in {pos.board.san(move) for move in pos.board.legal_moves}
        normalized_legal = normalized_legal_sans(pos.board)
        repaired_legal = token_is_legal_prediction(raw_top1, normalized_legal)
        under_disambiguated_matches = under_disambiguated_legal_matches(raw_top1, normalized_legal)
        probs = torch.softmax(logits, dim=-1)
        ids = legal_token_ids(stoi, pos.board, allow_eos=False)
        legal_mass = float(probs[ids].sum().item()) if ids else 0.0
        target_is_top1 = raw_top1_idx == pos.target_id

        overall.add(
            exact_legal=exact_legal,
            repaired_legal=repaired_legal,
            under_disambiguated=bool(under_disambiguated_matches),
            legal_mass=legal_mass,
            target_is_top1=target_is_top1,
        )
        names = feature_names(pos.feature_mask)
        for name in names:
            by_feature[name].add(
                exact_legal=exact_legal,
                repaired_legal=repaired_legal,
                under_disambiguated=bool(under_disambiguated_matches),
                legal_mass=legal_mass,
                target_is_top1=target_is_top1,
            )
        bucket = "le_3" if pos.legal_count <= 3 else "le_6" if pos.legal_count <= 6 else "le_10" if pos.legal_count <= 10 else "gt_10"
        by_legal_bucket[bucket].add(
            exact_legal=exact_legal,
            repaired_legal=repaired_legal,
            under_disambiguated=bool(under_disambiguated_matches),
            legal_mass=legal_mass,
            target_is_top1=target_is_top1,
        )

        if not repaired_legal:
            top_illegal[strip_san(raw_top1)] += 1
            if len(failures) < args.write_failures:
                failures.append(
                    {
                        "index": pos.index,
                        "ply": pos.ply,
                        "features": names,
                        "fen": pos.board.fen(),
                        "raw_top1": raw_top1,
                        "raw_top1_under_disambiguated": bool(under_disambiguated_matches),
                        "raw_top1_under_disambiguated_matches": under_disambiguated_matches,
                        "target": target_token,
                        "raw_top1_prob": round(float(probs[raw_top1_idx].item()), 6),
                        "legal_mass": round(legal_mass, 6),
                        "legal_count": pos.legal_count,
                    }
                )

    return {
        "type": "king_safety_legality_summary",
        "model": args.model,
        "hard_data_dir": str(resolve_path(args.hard_data_dir)),
        "split": args.split,
        "device": device,
        "load_info": load_info | {"positions": len(positions)},
        "metrics": overall.as_dict(),
        "by_feature": {name: stats.as_dict() for name, stats in sorted(by_feature.items())},
        "by_legal_bucket": {name: stats.as_dict() for name, stats in sorted(by_legal_bucket.items())},
        "top_illegal_raw_top1": top_illegal.most_common(args.top_illegal),
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate raw legality on king-safety hard examples.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--hard-data-dir", default="data/king_safety_sft")
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--max-positions", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="", choices=["", "cuda", "cpu", "mps"])
    parser.add_argument("--top-illegal", type=int, default=20)
    parser.add_argument("--write-failures", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    summary = evaluate(args)
    output = resolve_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
