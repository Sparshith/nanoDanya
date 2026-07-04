from __future__ import annotations

import argparse
import json
import pickle
import re
from array import array
from collections import Counter
from pathlib import Path


RESULT_TOKENS = {"1-0", "0-1", "1/2-1/2", "*"}
MOVE_NUMBER_RE = re.compile(r"\d+\.(?:\.\.)?$")


def iter_input_files(input_paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for raw_path in input_paths:
        matches = sorted(Path().glob(raw_path))
        if matches:
            files.extend(path.resolve() for path in matches if path.is_file())
            continue

        path = Path(raw_path).expanduser().resolve()
        if path.is_dir():
            files.extend(sorted(p.resolve() for p in path.iterdir() if p.is_file()))
        elif path.is_file():
            files.append(path)
        else:
            raise FileNotFoundError(f"No file or glob match for input path: {raw_path}")

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    if not deduped:
        raise FileNotFoundError("No input files found")
    return deduped


def iter_game_moves(paths: list[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                moves: list[str] = []
                for token in line.split():
                    if token in RESULT_TOKENS or MOVE_NUMBER_RE.fullmatch(token):
                        continue
                    moves.append(token)
                if moves:
                    yield path, line_no, moves


def next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def build_vocab(
    input_paths: list[Path],
    min_freq: int,
    limit_games: int | None,
    progress_every: int,
    context_length_override: int | None,
):
    move_counts: Counter[str] = Counter()
    scanned_games = 0
    max_seq_len = 0

    for _, _, moves in iter_game_moves(input_paths):
        scanned_games += 1
        move_counts.update(moves)
        max_seq_len = max(max_seq_len, len(moves) + 2)  # <bos>, <eos>
        if progress_every > 0 and scanned_games % progress_every == 0:
            print(f"pass1: scanned {scanned_games:,} games, vocab candidates={len(move_counts):,}")
        if limit_games is not None and scanned_games >= limit_games:
            break

    vocab_moves = sorted(move for move, count in move_counts.items() if count >= min_freq)
    pruned_moves = len(move_counts) - len(vocab_moves)
    stoi = {"<bos>": 0, "<eos>": 1}
    for idx, move in enumerate(vocab_moves, start=2):
        stoi[move] = idx
    itos = [""] * len(stoi)
    for token, idx in stoi.items():
        itos[idx] = token

    if len(stoi) > 65535:
        raise ValueError(f"Vocabulary too large for uint16 tokens: {len(stoi):,}")

    inferred_context_length = next_power_of_two(max_seq_len)
    context_length = context_length_override if context_length_override is not None else inferred_context_length
    print(
        f"pass1: complete. scanned={scanned_games:,} games, "
        f"vocab={len(stoi):,} tokens, pruned={pruned_moves:,}, "
        f"max_seq_len={max_seq_len}, inferred_context_length={inferred_context_length}, "
        f"context_length={context_length}"
    )
    return {
        "scanned_games": scanned_games,
        "max_seq_len": max_seq_len,
        "inferred_context_length": inferred_context_length,
        "move_counts": move_counts,
        "stoi": stoi,
        "itos": itos,
        "context_length": context_length,
        "pruned_moves": pruned_moves,
    }


def count_kept_games(
    input_paths: list[Path],
    stoi: dict[str, int],
    limit_games: int | None,
    progress_every: int,
):
    scanned_games = 0
    kept_games = 0
    dropped_oov_games = 0
    kept_tokens = 0

    for _, _, moves in iter_game_moves(input_paths):
        scanned_games += 1
        if all(move in stoi for move in moves):
            kept_games += 1
            kept_tokens += len(moves) + 2
        else:
            dropped_oov_games += 1
        if progress_every > 0 and scanned_games % progress_every == 0:
            print(
                f"pass2: scanned {scanned_games:,} games, kept={kept_games:,}, "
                f"dropped_oov={dropped_oov_games:,}"
            )
        if limit_games is not None and scanned_games >= limit_games:
            break

    print(
        f"pass2: complete. kept={kept_games:,} games, dropped_oov={dropped_oov_games:,}, "
        f"kept_tokens={kept_tokens:,}"
    )
    return {
        "scanned_games": scanned_games,
        "kept_games": kept_games,
        "dropped_oov_games": dropped_oov_games,
        "kept_tokens": kept_tokens,
    }


def write_split(
    out_path: Path,
    game_iter,
):
    tokens_written = 0
    games_written = 0
    with out_path.open("wb") as f:
        for token_ids in game_iter:
            arr = array("H", token_ids)
            arr.tofile(f)
            games_written += 1
            tokens_written += len(token_ids)
    return games_written, tokens_written


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a plain SAN next-token dataset from one-game-per-line actual-game text."
    )
    parser.add_argument(
        "--input-path",
        nargs="+",
        required=True,
        help="One or more text files, directories, or glob patterns with one SAN game per line.",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory for train.bin, val.bin, meta.pkl.")
    parser.add_argument("--min-freq", type=int, default=1, help="Drop move tokens seen fewer than this many times.")
    parser.add_argument("--train-ratio", type=float, default=0.9, help="Train split ratio on kept games.")
    parser.add_argument("--limit-games", type=int, default=None, help="Optional cap for pilot runs.")
    parser.add_argument("--progress-every", type=int, default=500000, help="Progress print interval in games.")
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help="Override context_length in meta.pkl. Defaults to next power of two covering the longest game.",
    )
    args = parser.parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be between 0 and 1")
    if args.min_freq < 1:
        raise ValueError("--min-freq must be >= 1")
    if args.context_length is not None and args.context_length < 2:
        raise ValueError("--context-length must be >= 2")

    input_paths = iter_input_files(args.input_path)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"inputs: {len(input_paths)} file(s)")
    for path in input_paths[:5]:
        print(f"  {path}")
    if len(input_paths) > 5:
        print(f"  ... ({len(input_paths) - 5} more)")
    print(f"output: {out_dir}")
    print(
        f"config: min_freq={args.min_freq} train_ratio={args.train_ratio} "
        f"limit_games={args.limit_games} progress_every={args.progress_every} "
        f"context_length={args.context_length}"
    )

    vocab_info = build_vocab(
        input_paths=input_paths,
        min_freq=args.min_freq,
        limit_games=args.limit_games,
        progress_every=args.progress_every,
        context_length_override=args.context_length,
    )
    stoi = vocab_info["stoi"]
    itos = vocab_info["itos"]

    keep_info = count_kept_games(
        input_paths=input_paths,
        stoi=stoi,
        limit_games=args.limit_games,
        progress_every=args.progress_every,
    )
    kept_games = keep_info["kept_games"]
    train_games_target = int(kept_games * args.train_ratio)

    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"

    current_kept = 0

    def iter_train_games():
        nonlocal current_kept
        for _, _, moves in iter_game_moves(input_paths):
            if args.limit_games is not None and current_kept >= kept_games:
                break
            if not all(move in stoi for move in moves):
                continue
            if current_kept >= train_games_target:
                break
            current_kept += 1
            yield [stoi["<bos>"], *[stoi[move] for move in moves], stoi["<eos>"]]

    train_games_written, train_tokens_written = write_split(train_path, iter_train_games())
    print(f"pass3: wrote train split: games={train_games_written:,} tokens={train_tokens_written:,}")

    def iter_val_games():
        seen_kept = 0
        for _, _, moves in iter_game_moves(input_paths):
            if args.limit_games is not None and seen_kept >= kept_games:
                break
            if not all(move in stoi for move in moves):
                continue
            seen_kept += 1
            if seen_kept <= train_games_target:
                continue
            yield [stoi["<bos>"], *[stoi[move] for move in moves], stoi["<eos>"]]

    val_games_written, val_tokens_written = write_split(val_path, iter_val_games())
    print(f"pass3: wrote val split: games={val_games_written:,} tokens={val_tokens_written:,}")

    meta = {
        "vocab_size": len(itos),
        "itos": itos,
        "stoi": stoi,
        "context_length": vocab_info["context_length"],
    }
    with (out_dir / "meta.pkl").open("wb") as f:
        pickle.dump(meta, f)

    summary = {
        "input_paths": [str(path) for path in input_paths],
        "out_dir": str(out_dir),
        "min_freq": args.min_freq,
        "train_ratio": args.train_ratio,
        "limit_games": args.limit_games,
        "progress_every": args.progress_every,
        "context_length_override": args.context_length,
        "scanned_games": vocab_info["scanned_games"],
        "kept_games": kept_games,
        "dropped_oov_games": keep_info["dropped_oov_games"],
        "train_games": train_games_written,
        "val_games": val_games_written,
        "train_tokens": train_tokens_written,
        "val_tokens": val_tokens_written,
        "vocab_size": len(itos),
        "max_seq_len": vocab_info["max_seq_len"],
        "inferred_context_length": vocab_info["inferred_context_length"],
        "context_length": vocab_info["context_length"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote metadata to {out_dir / 'meta.pkl'}")
    print(f"wrote summary to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
