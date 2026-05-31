from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_BITS = {
    "in_check": 1 << 0,
    "low_legal_count": 1 << 1,
    "pinned_piece": 1 << 2,
    "unsafe_king_capture": 1 << 3,
    "late_position": 1 << 4,
    "slider_pressure": 1 << 5,
    "target_gives_check": 1 << 6,
}
KING_SAFETY_MASK = (
    FEATURE_BITS["in_check"]
    | FEATURE_BITS["low_legal_count"]
    | FEATURE_BITS["pinned_piece"]
    | FEATURE_BITS["unsafe_king_capture"]
    | FEATURE_BITS["slider_pressure"]
    | FEATURE_BITS["target_gives_check"]
)


def resolve_path(path: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw
    return (PROJECT_ROOT / raw).resolve()


def load_meta(data_dir: Path) -> dict[str, Any]:
    return pickle.loads((data_dir / "meta.pkl").read_bytes())


def tokenizer_fingerprint(meta: dict[str, Any]) -> tuple[int, int, str, str]:
    stoi = meta["stoi"]
    return (
        int(meta["vocab_size"]),
        int(meta["context_length"]),
        str(stoi["<bos>"]),
        str(stoi["<eos>"]),
    )


def load_required(path: Path, dtype) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return np.fromfile(path, dtype=dtype)


def validate_split(
    hard_data_dir: Path,
    split: str,
    *,
    context_length: int,
    min_king_safety_rate: float,
) -> dict[str, Any]:
    prefixes = load_required(hard_data_dir / f"{split}_hard_prefixes.bin", np.uint16)
    lengths = load_required(hard_data_dir / f"{split}_hard_lengths.bin", np.uint16)
    targets = load_required(hard_data_dir / f"{split}_hard_targets.bin", np.uint16)
    scores = load_required(hard_data_dir / f"{split}_hard_scores.bin", np.uint16)
    feature_masks = load_required(hard_data_dir / f"{split}_hard_feature_masks.bin", np.uint16)
    legal_counts = load_required(hard_data_dir / f"{split}_hard_legal_counts.bin", np.uint16)
    plies = load_required(hard_data_dir / f"{split}_hard_plies.bin", np.uint16)

    examples = len(targets)
    if examples == 0:
        raise ValueError(f"{split}: no hard examples")
    if prefixes.size != examples * context_length:
        raise ValueError(f"{split}: prefix shape mismatch")
    for name, values in {
        "lengths": lengths,
        "scores": scores,
        "feature_masks": feature_masks,
        "legal_counts": legal_counts,
        "plies": plies,
    }.items():
        if len(values) != examples:
            raise ValueError(f"{split}: {name} count mismatch: {len(values)} vs {examples}")

    if int(lengths.min()) < 1 or int(lengths.max()) > context_length:
        raise ValueError(f"{split}: invalid prefix lengths")
    if int(scores.min()) < 1:
        raise ValueError(f"{split}: invalid zero score in hard examples")
    if int(feature_masks.min()) == 0:
        raise ValueError(f"{split}: at least one hard example has no feature mask")

    king_safety = (feature_masks & KING_SAFETY_MASK) != 0
    king_safety_rate = float(king_safety.mean())
    if king_safety_rate < min_king_safety_rate:
        raise ValueError(
            f"{split}: king-safety feature rate {king_safety_rate:.4f} "
            f"is below required {min_king_safety_rate:.4f}"
        )

    feature_counts = {
        name: int(((feature_masks & bit) != 0).sum())
        for name, bit in FEATURE_BITS.items()
    }
    return {
        "examples": examples,
        "king_safety_feature_rate": king_safety_rate,
        "score_min": int(scores.min()),
        "score_mean": float(scores.mean()),
        "score_max": int(scores.max()),
        "legal_count_mean": float(legal_counts.mean()),
        "ply_mean": float(plies.mean()),
        "feature_counts": feature_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate king-safety SFT hard examples and the configured 70/30 training mixture."
    )
    parser.add_argument("--normal-data-dir", default="data/actual_5m")
    parser.add_argument("--hard-data-dir", default="data/king_safety_sft")
    parser.add_argument("--hard-fraction", type=float, default=0.30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--min-king-safety-rate", type=float, default=0.99)
    parser.add_argument("--output", default="", help="Optional JSON report path.")
    args = parser.parse_args()

    if not (0.0 < args.hard_fraction < 1.0):
        raise ValueError("--hard-fraction must be between 0 and 1")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be >= 2")

    normal_data_dir = resolve_path(args.normal_data_dir)
    hard_data_dir = resolve_path(args.hard_data_dir)
    normal_meta = load_meta(normal_data_dir)
    hard_meta = load_meta(hard_data_dir)
    if tokenizer_fingerprint(normal_meta) != tokenizer_fingerprint(hard_meta):
        raise ValueError("normal and hard datasets do not share the same tokenizer/context contract")

    context_length = int(normal_meta["context_length"])
    splits = {
        split: validate_split(
            hard_data_dir,
            split,
            context_length=context_length,
            min_king_safety_rate=args.min_king_safety_rate,
        )
        for split in ("train", "val")
    }

    hard_batch = round(args.batch_size * args.hard_fraction)
    normal_batch = args.batch_size - hard_batch
    if hard_batch <= 0 or normal_batch <= 0:
        raise ValueError("configured batch leaves no room for normal or hard examples")

    report = {
        "type": "king_safety_sft_validation",
        "normal_data_dir": str(normal_data_dir),
        "hard_data_dir": str(hard_data_dir),
        "context_length": context_length,
        "hard_fraction": args.hard_fraction,
        "objective_mix": {
            "normal": 1.0 - args.hard_fraction,
            "hard": args.hard_fraction,
            "note": "The trainer weights the normal and hard losses by these exact fractions.",
        },
        "batch_mix": {
            "batch_size": args.batch_size,
            "normal_batch": normal_batch,
            "hard_batch": hard_batch,
            "hard_batch_fraction": hard_batch / args.batch_size,
            "note": "Batch composition is rounded; the loss mixture remains exact.",
        },
        "splits": splits,
    }

    print(json.dumps(report, indent=2))
    if args.output:
        output_path = resolve_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
