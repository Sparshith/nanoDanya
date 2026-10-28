"""Compare teacher, float student, and exported student on untouched validation games."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F

from benchmark.run import (
    last_logits_for_prefixes,
    phase_for_ply,
    raw_metrics_from_logits,
    resolve_device,
    write_jsonl,
)
from chess_inference import legal_token_ids, token_for_id
from chess_token_utils import resolve_token_id
from model_loading import load_model


@dataclass
class EvalPosition:
    board: chess.Board
    prefix: list[str]
    target: str
    game_index: int
    ply: int


def load_games(data_dir: Path, split: str, start_fraction: float, max_games: int):
    meta = pickle.loads((data_dir / "meta.pkl").read_bytes())
    tokens = np.memmap(data_dir / f"{split}.bin", dtype=np.uint16, mode="r")
    bos_id = meta["stoi"]["<bos>"]
    eos_id = meta["stoi"]["<eos>"]
    start = int(len(tokens) * start_fraction)
    games = []
    current = []
    for raw_id in tokens[start:]:
        token_id = int(raw_id)
        if token_id == bos_id:
            current = [token_id]
        elif current:
            current.append(token_id)
            if token_id == eos_id:
                games.append(current)
                current = []
                if len(games) == max_games:
                    break
    return games, meta


def select_positions(games, itos, target_plies, max_positions, seed):
    candidates = []
    for game_index, game in enumerate(games):
        board = chess.Board()
        prefix = ["<bos>"]
        for ply, token_id in enumerate(game[1:], start=1):
            target = token_for_id(itos, token_id)
            if target == "<eos>":
                break
            try:
                move = board.parse_san(target)
            except ValueError:
                break
            if ply in target_plies:
                candidates.append(EvalPosition(board.copy(stack=False), prefix.copy(), target, game_index, ply))
            board.push(move)
            prefix.append(target)
    if len(candidates) > max_positions:
        candidates = random.Random(seed).sample(candidates, max_positions)
    candidates.sort(key=lambda pos: (pos.game_index, pos.ply))
    return candidates


def selected_hash(positions):
    payload = "\n".join(
        f"{pos.game_index}:{pos.ply}:{' '.join(pos.prefix)}:{pos.target}" for pos in positions
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@torch.inference_mode()
def evaluate(model, positions, stoi, itos, batch_size, device, teacher=None):
    prefixes = [[resolve_token_id(stoi, token) for token in pos.prefix] for pos in positions]
    if any(None in prefix for prefix in prefixes):
        raise ValueError("A heldout SAN token is absent from the model tokenizer")
    targets = [resolve_token_id(stoi, pos.target) for pos in positions]
    if any(target is None for target in targets):
        raise ValueError("A heldout target is absent from the model tokenizer")

    counts = {phase: {"positions": 0, "ce": 0.0, "raw_illegal": 0, "legal_mass": 0.0,
                      "raw_agree": 0, "legal_agree": 0, "kl": 0.0}
              for phase in ("opening", "middlegame", "endgame")}
    decisions = [None] * len(positions)
    for idx, logits in last_logits_for_prefixes(
        model, prefixes, pad_id=stoi["<eos>"], device=device, batch_size=batch_size
    ):
        pos = positions[idx]
        phase = phase_for_ply(pos.ply)
        stats = counts[phase]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        legal_ids = legal_token_ids(stoi, pos.board, allow_eos=False)
        if not legal_ids:
            raise ValueError(f"No legal SAN token at game {pos.game_index}, ply {pos.ply}")
        legal_top1 = legal_ids[int(torch.argmax(logits[legal_ids]).item())]
        raw_top1 = int(torch.argmax(logits).item())
        raw = raw_metrics_from_logits(logits, pos.board, stoi, itos, allow_eos=False)
        stats["positions"] += 1
        stats["ce"] -= float(log_probs[targets[idx]].item())
        stats["raw_illegal"] += int(not raw["raw_top1_legal"])
        stats["legal_mass"] += raw["legal_mass"]
        if teacher is not None:
            teacher_raw, teacher_legal, teacher_log_probs = teacher[idx]
            stats["raw_agree"] += int(raw_top1 == teacher_raw)
            stats["legal_agree"] += int(legal_top1 == teacher_legal)
            stats["kl"] += float(
                (teacher_log_probs.exp() * (teacher_log_probs - log_probs.cpu())).sum().item()
            )
        decisions[idx] = (raw_top1, legal_top1, log_probs.cpu())

    def rates(stats):
        n = stats["positions"]
        if n == 0:
            return None
        result = {
            "positions": n,
            "cross_entropy": stats["ce"] / n,
            "raw_top1_illegal_rate": stats["raw_illegal"] / n,
            "avg_legal_mass": stats["legal_mass"] / n,
        }
        if teacher is not None:
            result |= {
                "teacher_raw_top1_agreement": stats["raw_agree"] / n,
                "teacher_legal_top1_agreement": stats["legal_agree"] / n,
                "teacher_kl": stats["kl"] / n,
            }
        return result

    overall = {key: sum(stats[key] for stats in counts.values()) for key in counts["opening"]}
    return {"overall": rates(overall), "by_phase": {phase: rates(stats) for phase, stats in counts.items()}}, decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", default=os.getenv("TEACHER_CKPT", "plain/games-15m/l12"))
    parser.add_argument("--student-checkpoint", default=os.getenv("STUDENT_CHECKPOINT"))
    parser.add_argument("--quantized-artifact", default=os.getenv("STUDENT_ARTIFACT", ""))
    parser.add_argument("--data-dir", default=os.getenv("DISTILL_EVAL_DATA_DIR", "/data/datasets/games/15m"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--start-fraction", type=float,
                        default=float(os.getenv("DISTILL_EVAL_START_FRACTION", "0.5")))
    parser.add_argument("--max-games", type=int, default=int(os.getenv("DISTILL_EVAL_MAX_GAMES", "1000")))
    parser.add_argument("--max-positions", type=int,
                        default=int(os.getenv("DISTILL_EVAL_MAX_POSITIONS", "1024")))
    parser.add_argument("--plies", type=int, nargs="+", default=[7, 8, 19, 20, 39, 40, 79, 80])
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("DISTILL_EVAL_BATCH_SIZE", "64")))
    parser.add_argument("--device", default="", choices=["", "cpu", "cuda", "mps"])
    parser.add_argument("--output", default=os.getenv("DISTILL_EVAL_OUTPUT"))
    args = parser.parse_args()
    if not args.student_checkpoint or not args.output:
        parser.error("student-checkpoint and output are required via CLI or environment")
    if not 0 <= args.start_fraction < 1 or args.max_games <= 0 or args.max_positions <= 0:
        parser.error("start-fraction must be in [0,1), max-games and max-positions must be positive")

    device = resolve_device(args.device)
    games, data_meta = load_games(Path(args.data_dir), args.split, args.start_fraction, args.max_games)
    positions = select_positions(games, data_meta["itos"], set(args.plies), args.max_positions, args.seed)
    if not positions:
        raise ValueError("No heldout positions matched the requested plies")

    teacher_model, teacher_config, stoi, itos = load_model(args.teacher, device)
    student_model, student_config, student_stoi, student_itos = load_model(args.student_checkpoint, device)
    if stoi != student_stoi:
        raise ValueError("Teacher and student tokenizer mappings differ")
    if max(pos.ply for pos in positions) >= min(teacher_config.sequence_len, student_config.sequence_len):
        raise ValueError("Selected positions exceed teacher or student context length")

    teacher_metrics, teacher_decisions = evaluate(
        teacher_model, positions, stoi, itos, args.batch_size, device
    )
    student_qat = student_model.qat
    student_model.qat = False
    student_metrics, _ = evaluate(
        student_model, positions, student_stoi, student_itos, args.batch_size, device, teacher_decisions
    )
    models = {"teacher": teacher_metrics, "student_float": student_metrics}
    if student_qat:
        student_model.qat = True
        qat_metrics, _ = evaluate(
            student_model, positions, student_stoi, student_itos, args.batch_size, device,
            teacher_decisions,
        )
        models["student_fake_quantized"] = qat_metrics
    if args.quantized_artifact:
        from inference.onnx_policy import load_torch_from_onnx as load_quantized_model

        quant_model, quant_stoi, quant_itos = load_quantized_model(args.quantized_artifact, device=device)
        if stoi != quant_stoi:
            raise ValueError("Teacher and quantized student tokenizer mappings differ")
        quant_metrics, _ = evaluate(
            quant_model.eval(), positions, quant_stoi, quant_itos, args.batch_size, device,
            teacher_decisions,
        )
        models["student_quantized"] = quant_metrics

    summary = {
        "type": "distill_summary",
        "schema_version": 1,
        "data_dir": args.data_dir,
        "split": args.split,
        "start_fraction": args.start_fraction,
        "selected_sha256": selected_hash(positions),
        "games_scanned": len(games),
        "positions": len(positions),
        "models": models,
    }
    write_jsonl(args.output, [{"type": "run", "command": "distill_eval", "config": vars(args)}, summary])
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
