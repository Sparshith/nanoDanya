from __future__ import annotations

import argparse
import json
import pickle
import random
import statistics
import sys
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import chess
import chess.engine
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from chess_token_utils import (
    normalized_legal_sans,
    resolve_token_id,
    strip_san,
    token_is_legal_prediction,
)
from chess_inference import choose_move_from_logits, legal_token_ids, token_for_id
from inference.kv_cache import KVCache
from model_loading import load_model
from model_registry import model_ref_help


SCHEMA_VERSION = 1
MATE_SCORE = 100_000


@dataclass
class Position:
    board: chess.Board
    prefix_ids: list[int]
    ply: int
    game_index: int


@dataclass
class LegalityStats:
    positions: int = 0
    illegal_top1: int = 0
    legal_mass_sum: float = 0.0

    def add(self, *, raw_top1_legal: bool, legal_mass: float) -> None:
        self.positions += 1
        self.illegal_top1 += 0 if raw_top1_legal else 1
        self.legal_mass_sum += legal_mass

    def as_metrics(self) -> dict:
        if self.positions == 0:
            return {
                "positions": 0,
                "raw_top1_illegal_rate": None,
                "avg_legal_mass": None,
            }
        return {
            "positions": self.positions,
            "raw_top1_illegal_rate": self.illegal_top1 / self.positions,
            "avg_legal_mass": self.legal_mass_sum / self.positions,
        }


@dataclass
class TerminationStats:
    positions: int = 0
    eos_top1: int = 0
    eos_prob_sum: float = 0.0
    eos_ranks: list[int] | None = None

    def add(self, *, eos_is_top1: bool, eos_prob: float, eos_rank: int) -> None:
        self.positions += 1
        self.eos_top1 += 1 if eos_is_top1 else 0
        self.eos_prob_sum += eos_prob
        if self.eos_ranks is None:
            self.eos_ranks = []
        self.eos_ranks.append(eos_rank)

    def as_metrics(self) -> dict:
        if self.positions == 0:
            return {
                "positions": 0,
                "eos_top1_rate": None,
                "avg_eos_prob": None,
                "median_eos_rank": None,
            }
        return {
            "positions": self.positions,
            "eos_top1_rate": self.eos_top1 / self.positions,
            "avg_eos_prob": self.eos_prob_sum / self.positions,
            "median_eos_rank": statistics.median(self.eos_ranks or []),
        }


@dataclass
class GameState:
    game_id: str
    game_num: int
    board: chess.Board
    model_color: chess.Color | None
    prefixes: dict[str, list[int]]
    moves: list[str]
    finished: bool = False
    termination: str = ""
    result: str = "*"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def resolve_device(device_arg: str) -> str:
    if device_arg:
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def write_jsonl(path: str | Path, records: Iterable[dict], *, append: bool = False) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with out.open(mode) as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def run_record(command: str, args: argparse.Namespace, extra: dict | None = None) -> dict:
    config = vars(args).copy()
    for internal_key in ("func", "command"):
        if internal_key in config:
            del config[internal_key]
    return {
        "type": "run",
        "schema_version": SCHEMA_VERSION,
        "run_id": str(uuid.uuid4()),
        "created_at": now_iso(),
        "command": command,
        "config": config | (extra or {}),
    }


def load_token_games(data_dir: Path, split: str, max_games: int) -> tuple[list[list[str]], dict]:
    meta = pickle.loads((data_dir / "meta.pkl").read_bytes())
    ids = np.fromfile(data_dir / f"{split}.bin", dtype=np.uint16)
    bos_id = meta["stoi"]["<bos>"]
    eos_id = meta["stoi"]["<eos>"]
    games: list[list[str]] = []
    current: list[str] = []

    for raw_id in ids:
        token_id = int(raw_id)
        if token_id == bos_id:
            current = ["<bos>"]
            continue
        if not current:
            continue
        current.append(token_for_id(meta["itos"], token_id))
        if token_id == eos_id:
            games.append(current)
            current = []
            if max_games > 0 and len(games) >= max_games:
                break

    return games, meta


def load_id_games(data_dir: Path, split: str, max_games: int) -> tuple[list[list[int]], dict]:
    meta = pickle.loads((data_dir / "meta.pkl").read_bytes())
    ids = np.fromfile(data_dir / f"{split}.bin", dtype=np.uint16)
    bos_id = meta["stoi"]["<bos>"]
    eos_id = meta["stoi"]["<eos>"]
    games: list[list[int]] = []
    current: list[int] = []

    for raw_id in ids:
        token_id = int(raw_id)
        if token_id == bos_id:
            current = [token_id]
            continue
        if not current:
            continue
        current.append(token_id)
        if token_id == eos_id:
            games.append(current)
            current = []
            if max_games > 0 and len(games) >= max_games:
                break

    return games, meta


def collect_positions(
    *,
    data_dir: Path,
    split: str,
    max_games: int,
    max_positions: int,
    max_ply: int,
    model_stoi: dict[str, int],
) -> list[Position]:
    games, _ = load_token_games(data_dir, split, max_games)
    positions: list[Position] = []
    bos_id = model_stoi["<bos>"]

    for game_index, game in enumerate(games):
        board = chess.Board()
        prefix_ids = [bos_id]
        ply = 0

        for token in game[1:]:
            if token == "<eos>":
                break

            target_id = resolve_token_id(model_stoi, token)
            if target_id is None:
                break

            ply += 1
            if max_ply <= 0 or ply <= max_ply:
                positions.append(
                    Position(
                        board=board.copy(stack=False),
                        prefix_ids=prefix_ids.copy(),
                        ply=ply,
                        game_index=game_index,
                    )
                )
                if max_positions > 0 and len(positions) >= max_positions:
                    return positions

            try:
                board.push_san(token)
            except ValueError:
                break
            prefix_ids.append(target_id)

    return positions


def last_logits_for_prefixes(
    model,
    prefixes: list[list[int]],
    *,
    pad_id: int,
    device: str,
    batch_size: int,
) -> Iterable[tuple[int, torch.Tensor]]:
    for start in range(0, len(prefixes), batch_size):
        batch = prefixes[start:start + batch_size]
        max_len = max(len(prefix) for prefix in batch)
        x = torch.full((len(batch), max_len), pad_id, dtype=torch.long, device=device)
        last_idx = []
        for row, prefix in enumerate(batch):
            x[row, :len(prefix)] = torch.tensor(prefix, dtype=torch.long, device=device)
            last_idx.append(len(prefix) - 1)
        logits = model(x)
        for row, idx in enumerate(last_idx):
            yield start + row, logits[row, idx, :].detach()


def raw_metrics_from_logits(
    logits: torch.Tensor,
    board: chess.Board,
    stoi: dict[str, int],
    itos: dict,
    *,
    allow_eos: bool,
) -> dict:
    raw_top1_idx = int(torch.argmax(logits).item())
    raw_top1 = token_for_id(itos, raw_top1_idx)
    legal_san = normalized_legal_sans(board)
    raw_top1_legal = token_is_legal_prediction(raw_top1, legal_san)

    probs = torch.softmax(logits, dim=-1)
    legal_ids = legal_token_ids(stoi, board, allow_eos=allow_eos)
    legal_mass = float(probs[legal_ids].sum().item()) if legal_ids else 0.0

    return {
        "raw_top1": raw_top1,
        "raw_top1_legal": raw_top1_legal,
        "raw_top1_prob": round(float(probs[raw_top1_idx].item()), 6),
        "legal_mass": legal_mass,
    }


def phase_for_ply(ply: int) -> str:
    if ply <= 20:
        return "opening"
    if ply <= 80:
        return "middlegame"
    return "endgame"


@torch.inference_mode()
def command_legality(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, _, stoi, itos = load_model(args.model, device)
    positions = collect_positions(
        data_dir=Path(args.data_dir),
        split=args.split,
        max_games=args.max_games,
        max_positions=args.max_positions,
        max_ply=args.max_ply,
        model_stoi=stoi,
    )

    overall = LegalityStats()
    by_phase: dict[str, LegalityStats] = defaultdict(LegalityStats)
    top_illegal = Counter()

    prefixes = [pos.prefix_ids for pos in positions]
    for idx, logits in last_logits_for_prefixes(
        model,
        prefixes,
        pad_id=stoi["<eos>"],
        device=device,
        batch_size=args.batch_size,
    ):
        pos = positions[idx]
        metrics = raw_metrics_from_logits(
            logits,
            pos.board,
            stoi,
            itos,
            allow_eos=args.allow_eos,
        )
        overall.add(
            raw_top1_legal=metrics["raw_top1_legal"],
            legal_mass=metrics["legal_mass"],
        )
        by_phase[phase_for_ply(pos.ply)].add(
            raw_top1_legal=metrics["raw_top1_legal"],
            legal_mass=metrics["legal_mass"],
        )
        if not metrics["raw_top1_legal"]:
            top_illegal[metrics["raw_top1"]] += 1

    summary = {
        "type": "legality_summary",
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "device": device,
        "metrics": overall.as_metrics(),
        "by_phase": {phase: stats.as_metrics() for phase, stats in sorted(by_phase.items())},
        "top_illegal_raw_top1": top_illegal.most_common(args.top_illegal),
    }

    records = [run_record("legality", args, {"device": device}), summary]
    write_jsonl(args.output, records)
    print_summary_record(summary)
    print(f"wrote {args.output}")


def termination_metrics_from_logits(logits: torch.Tensor, eos_id: int) -> dict:
    probs = torch.softmax(logits, dim=-1)
    eos_prob = float(probs[eos_id].item())
    raw_top1_idx = int(torch.argmax(logits).item())
    eos_rank = int((logits > logits[eos_id]).sum().item()) + 1
    return {
        "eos_is_top1": raw_top1_idx == eos_id,
        "eos_prob": eos_prob,
        "eos_rank": eos_rank,
    }


@torch.inference_mode()
def command_termination(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, _, stoi, _ = load_model(args.model, device)
    positions = collect_positions(
        data_dir=Path(args.data_dir),
        split=args.split,
        max_games=args.max_games,
        max_positions=args.max_positions,
        max_ply=args.max_ply,
        model_stoi=stoi,
    )

    overall = TerminationStats()
    by_phase: dict[str, TerminationStats] = defaultdict(TerminationStats)
    eos_id = stoi["<eos>"]

    prefixes = [pos.prefix_ids for pos in positions]
    for idx, logits in last_logits_for_prefixes(
        model,
        prefixes,
        pad_id=eos_id,
        device=device,
        batch_size=args.batch_size,
    ):
        pos = positions[idx]
        metrics = termination_metrics_from_logits(logits, eos_id)
        overall.add(**metrics)
        by_phase[phase_for_ply(pos.ply)].add(**metrics)

    summary = {
        "type": "termination_summary",
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "device": device,
        "metrics": overall.as_metrics(),
        "by_phase": {phase: stats.as_metrics() for phase, stats in sorted(by_phase.items())},
    }

    records = [run_record("termination", args, {"device": device}), summary]
    write_jsonl(args.output, records)
    print_summary_record(summary)
    print(f"wrote {args.output}")


def cp_for_side(engine: chess.engine.SimpleEngine, board: chess.Board, side: chess.Color, limit) -> int:
    info = engine.analyse(board, limit)
    return int(info["score"].pov(side).score(mate_score=MATE_SCORE))


@torch.inference_mode()
def command_move_quality(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, _, stoi, itos = load_model(args.model, device)
    positions = collect_positions(
        data_dir=Path(args.data_dir),
        split=args.split,
        max_games=args.max_games,
        max_positions=args.max_positions,
        max_ply=args.max_ply,
        model_stoi=stoi,
    )
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    limit = stockfish_limit(args)

    records = [run_record("move-quality", args, {"device": device, "stockfish_limit": str(limit)})]
    losses: list[int] = []
    failures = 0

    try:
        prefixes = [pos.prefix_ids for pos in positions]
        logits_by_index = dict(
            last_logits_for_prefixes(
                model,
                prefixes,
                pad_id=stoi["<eos>"],
                device=device,
                batch_size=args.batch_size,
            )
        )
        for idx, pos in enumerate(positions):
            side = pos.board.turn
            before_cp = cp_for_side(engine, pos.board, side, limit)
            next_id, token = choose_move_from_logits(
                logits_by_index[idx],
                pos.board,
                stoi,
                itos,
                temperature=0.0,
                allow_eos=False,
                legal_mask=True,
            )
            if next_id is None or token is None:
                failures += 1
                continue

            board_after = pos.board.copy(stack=False)
            try:
                board_after.push_san(token)
            except ValueError:
                failures += 1
                continue

            after_cp = cp_for_side(engine, board_after, side, limit)
            cp_loss = max(0, before_cp - after_cp)
            losses.append(cp_loss)
            if args.write_positions:
                records.append({
                    "type": "move_quality_position",
                    "schema_version": SCHEMA_VERSION,
                    "model": args.model,
                    "ply": pos.ply,
                    "fen": pos.board.fen(),
                    "played": strip_san(token),
                    "before_cp": before_cp,
                    "after_cp": after_cp,
                    "cp_loss": cp_loss,
                })
    finally:
        engine.quit()

    summary = {
        "type": "move_quality_summary",
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "device": device,
        "positions": len(positions),
        "scored_positions": len(losses),
        "failures": failures,
        "avg_cp_loss": statistics.mean(losses) if losses else None,
        "median_cp_loss": statistics.median(losses) if losses else None,
        "blunder_rate_cp_300": sum(loss >= 300 for loss in losses) / len(losses) if losses else None,
    }
    records.append(summary)
    write_jsonl(args.output, records)
    print_summary_record(summary)
    print(f"wrote {args.output}")


def stockfish_limit(args: argparse.Namespace):
    if getattr(args, "stockfish_nodes", 0) > 0:
        return chess.engine.Limit(nodes=args.stockfish_nodes)
    if getattr(args, "stockfish_depth", 0) > 0:
        return chess.engine.Limit(depth=args.stockfish_depth)
    return chess.engine.Limit(time=args.stockfish_time)


def apply_prompt(board: chess.Board, prompt_moves: list[str], prefixes: dict[str, list[int]], tokenizers: dict[str, dict]) -> None:
    for san in prompt_moves:
        board.push_san(san)
        for name, stoi in tokenizers.items():
            token_id = resolve_token_id(stoi, san)
            if token_id is None:
                raise KeyError(f"Prompt token {san!r} is not in {name} tokenizer")
            prefixes[name].append(token_id)


@torch.inference_mode()
def choose_for_states(
    *,
    states: list[GameState],
    model_name: str,
    model,
    stoi: dict[str, int],
    itos: dict,
    device: str,
    batch_size: int,
    temperature: float,
    allow_eos: bool,
    legal_mask: bool,
) -> list[tuple[GameState, int | None, str | None]]:
    prefixes = [state.prefixes[model_name] for state in states]
    output: list[tuple[GameState, int | None, str | None]] = []
    for idx, logits in last_logits_for_prefixes(
        model,
        prefixes,
        pad_id=stoi["<eos>"],
        device=device,
        batch_size=batch_size,
    ):
        state = states[idx]
        next_id, token = choose_move_from_logits(
            logits,
            state.board,
            stoi,
            itos,
            temperature=temperature,
            allow_eos=allow_eos,
            legal_mask=legal_mask,
        )
        output.append((state, next_id, token))
    return output


def finish_game(state: GameState, termination: str | None = None) -> None:
    state.finished = True
    if state.board.is_game_over():
        state.result = state.board.result()
        state.termination = state.board.outcome().termination.name
    else:
        state.result = state.board.result()
        state.termination = termination or "unfinished"


def game_outcome_for_model(result: str, model_color: chess.Color | None, termination: str = "") -> str:
    if termination.startswith("model_") and not termination.endswith("_oov"):
        return "loss"
    if termination.startswith("opponent_") and not termination.endswith("_oov"):
        return "win"
    if model_color is None:
        return "n/a"
    if result == "1-0":
        return "win" if model_color == chess.WHITE else "loss"
    if result == "0-1":
        return "loss" if model_color == chess.WHITE else "win"
    return "draw"


def game_record(state: GameState, args: argparse.Namespace) -> dict:
    return {
        "type": "game",
        "schema_version": SCHEMA_VERSION,
        "game_id": state.game_id,
        "game_num": state.game_num,
        "mode": args.game_mode,
        "model": args.model,
        "opponent_model": args.opponent_model,
        "model_color": "white" if state.model_color == chess.WHITE else ("black" if state.model_color == chess.BLACK else None),
        "moves": state.moves,
        "result": state.result,
        "outcome": game_outcome_for_model(state.result, state.model_color, state.termination),
        "termination": state.termination,
        "num_plies": len(state.moves),
    }


@torch.inference_mode()
def command_games(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, model_config, stoi, itos = load_model(args.model, device)
    opponent = None
    opponent_config = None
    opponent_stoi = opponent_itos = None
    if args.game_mode == "h2h":
        opponent, opponent_config, opponent_stoi, opponent_itos = load_model(args.opponent_model, device)

    prompt_moves = [tok for tok in args.prompt.split() if tok]
    tokenizers = {"model": stoi}
    if args.game_mode == "h2h":
        tokenizers["opponent"] = opponent_stoi
    states: list[GameState] = []
    for local_game_num in range(args.games):
        game_num = args.game_offset + local_game_num
        board = chess.Board()
        model_color = chess.WHITE if game_num % 2 == 0 else chess.BLACK
        prefixes = {"model": [stoi["<bos>"]]}
        if args.game_mode == "h2h":
            prefixes["opponent"] = [opponent_stoi["<bos>"]]
        apply_prompt(board, prompt_moves, prefixes, tokenizers)
        states.append(
            GameState(
                game_id=str(uuid.uuid4()),
                game_num=game_num,
                board=board,
                model_color=model_color,
                prefixes=prefixes,
                moves=list(prompt_moves),
            )
        )

    if args.kv_cache:
        if args.game_mode != "h2h":
            raise ValueError("--kv-cache is currently implemented for games --game-mode h2h only")
        command_h2h_games_kv(
            args=args,
            device=device,
            model=model,
            model_config=model_config,
            stoi=stoi,
            itos=itos,
            opponent=opponent,
            opponent_config=opponent_config,
            opponent_stoi=opponent_stoi,
            opponent_itos=opponent_itos,
            states=states,
            tokenizers=tokenizers,
        )
        return

    engine = None
    limit = None
    if args.game_mode == "stockfish":
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
        if args.stockfish_elo > 0:
            engine.configure({"UCI_LimitStrength": True, "UCI_Elo": args.stockfish_elo})
        limit = stockfish_limit(args)

    try:
        for _ in range(args.max_plies):
            active = [state for state in states if not state.finished]
            if not active:
                break

            for state in active:
                if state.board.is_game_over():
                    finish_game(state)
                elif len(state.moves) >= args.max_plies:
                    finish_game(state, "max_plies")

            if args.game_mode == "stockfish":
                opponent_turns = [s for s in states if not s.finished and s.board.turn != s.model_color]
                model_turns = [s for s in states if not s.finished and s.board.turn == s.model_color]

                for state in opponent_turns:
                    result = engine.play(state.board, limit)
                    san = strip_san(state.board.san(result.move))
                    state.board.push(result.move)
                    state.moves.append(san)
                    token_id = resolve_token_id(stoi, san)
                    if token_id is None:
                        finish_game(state, "opponent_oov")
                    else:
                        state.prefixes["model"].append(token_id)

                choices = choose_for_states(
                    states=model_turns,
                    model_name="model",
                    model=model,
                    stoi=stoi,
                    itos=itos,
                    device=device,
                    batch_size=args.batch_size,
                    temperature=args.temperature,
                    allow_eos=args.allow_eos,
                    legal_mask=args.legal_mask,
                )
                for state, next_id, token in choices:
                    apply_model_choice(state, "model", next_id, token, tokenizers, args.allow_eos)

            else:
                model_turns = [s for s in states if not s.finished and s.board.turn == s.model_color]
                opponent_turns = [s for s in states if not s.finished and s.board.turn != s.model_color]
                for model_name, turn_states, mdl, s_to_i, i_to_s in (
                    ("model", model_turns, model, stoi, itos),
                    ("opponent", opponent_turns, opponent, opponent_stoi, opponent_itos),
                ):
                    choices = choose_for_states(
                        states=turn_states,
                        model_name=model_name,
                        model=mdl,
                        stoi=s_to_i,
                        itos=i_to_s,
                        device=device,
                        batch_size=args.batch_size,
                        temperature=args.temperature,
                        allow_eos=args.allow_eos,
                        legal_mask=args.legal_mask,
                    )
                    for state, next_id, token in choices:
                        apply_model_choice(state, model_name, next_id, token, tokenizers, args.allow_eos)

        for state in states:
            if not state.finished:
                finish_game(state, "max_plies")
    finally:
        if engine is not None:
            engine.quit()

    records = [
        run_record(
            "games",
            args,
            {
                "device": device,
                "stockfish_limit": str(limit) if limit is not None else None,
            },
        )
    ]
    records.extend(game_record(state, args) for state in states)
    records.append(summarize_game_records(records[1:]))
    write_jsonl(args.output, records)
    print_summary_record(records[-1])
    print(f"wrote {args.output}")


def make_kv_cache(config, batch_size: int):
    return KVCache(
        batch_size=batch_size,
        num_heads=config.n_kv_head,
        seq_len=config.sequence_len,
        head_dim=config.n_embd // config.n_head,
        num_layers=config.n_layer,
    )


@torch.inference_mode()
def command_h2h_games_kv(
    *,
    args: argparse.Namespace,
    device: str,
    model,
    model_config,
    stoi: dict[str, int],
    itos: dict,
    opponent,
    opponent_config,
    opponent_stoi: dict[str, int],
    opponent_itos: dict,
    states: list[GameState],
    tokenizers: dict[str, dict],
) -> None:
    batch_size = len(states)
    model_cache = make_kv_cache(model_config, batch_size)
    opponent_cache = make_kv_cache(opponent_config, batch_size)
    model_eos = stoi["<eos>"]
    opponent_eos = opponent_stoi["<eos>"]

    model_x = torch.tensor([state.prefixes["model"] for state in states], dtype=torch.long, device=device)
    opponent_x = torch.tensor([state.prefixes["opponent"] for state in states], dtype=torch.long, device=device)
    model_logits = model(model_x, kv_cache=model_cache)
    opponent_logits = opponent(opponent_x, kv_cache=opponent_cache)

    for _ in range(args.max_plies):
        active = [state for state in states if not state.finished]
        if not active:
            break

        for state in active:
            if state.board.is_game_over():
                finish_game(state)
            elif len(state.moves) >= args.max_plies:
                finish_game(state, "max_plies")

        if all(state.finished for state in states):
            break

        next_model_ids = [model_eos] * batch_size
        next_opponent_ids = [opponent_eos] * batch_size
        moved = False

        for row, state in enumerate(states):
            if state.finished:
                continue

            if state.board.turn == state.model_color:
                model_name = "model"
                logits = model_logits[row, -1, :]
                s_to_i = stoi
                i_to_s = itos
            else:
                model_name = "opponent"
                logits = opponent_logits[row, -1, :]
                s_to_i = opponent_stoi
                i_to_s = opponent_itos

            before_model_len = len(state.prefixes["model"])
            before_opponent_len = len(state.prefixes["opponent"])
            next_id, token = choose_move_from_logits(
                logits,
                state.board,
                s_to_i,
                i_to_s,
                temperature=args.temperature,
                allow_eos=args.allow_eos,
                legal_mask=args.legal_mask,
            )
            apply_model_choice(state, model_name, next_id, token, tokenizers, args.allow_eos)
            if len(state.prefixes["model"]) > before_model_len:
                next_model_ids[row] = state.prefixes["model"][-1]
            if len(state.prefixes["opponent"]) > before_opponent_len:
                next_opponent_ids[row] = state.prefixes["opponent"][-1]
            moved = True

        if not moved or all(state.finished for state in states):
            break

        model_x = torch.tensor(next_model_ids, dtype=torch.long, device=device).unsqueeze(1)
        opponent_x = torch.tensor(next_opponent_ids, dtype=torch.long, device=device).unsqueeze(1)
        model_logits = model(model_x, kv_cache=model_cache)
        opponent_logits = opponent(opponent_x, kv_cache=opponent_cache)

    for state in states:
        if not state.finished:
            finish_game(state, "max_plies")

    records = [
        run_record(
            "games",
            args,
            {
                "device": device,
                "stockfish_limit": None,
                "kv_cache": True,
            },
        )
    ]
    records.extend(game_record(state, args) for state in states)
    records.append(summarize_game_records(records[1:]))
    write_jsonl(args.output, records)
    print_summary_record(records[-1])
    print(f"wrote {args.output}")


def apply_model_choice(
    state: GameState,
    model_name: str,
    next_id: int | None,
    token: str | None,
    tokenizers: dict[str, dict[str, int]],
    allow_eos: bool,
) -> None:
    if next_id is None or token is None:
        finish_game(state, f"{model_name}_no_move")
        return
    state.prefixes[model_name].append(next_id)
    if token == "<eos>":
        if allow_eos:
            finish_game(state, f"{model_name}_eos")
        else:
            finish_game(state, f"{model_name}_unexpected_eos")
        return
    try:
        state.board.push_san(token)
    except ValueError:
        finish_game(state, f"{model_name}_illegal_played")
        return
    played = strip_san(token)
    state.moves.append(played)
    for other_name, prefix in state.prefixes.items():
        if other_name == model_name:
            continue
        other_id = resolve_token_id(tokenizers[other_name], played)
        if other_id is None:
            finish_game(state, f"{other_name}_oov")
            return
        prefix.append(other_id)


def summarize_game_records(records: list[dict]) -> dict:
    outcomes = Counter(record["outcome"] for record in records)
    terminations = Counter(record["termination"] for record in records)
    games = len(records)
    score = (outcomes["win"] + 0.5 * outcomes["draw"]) / games if games else None
    return {
        "type": "games_summary",
        "schema_version": SCHEMA_VERSION,
        "games": games,
        "score": score,
        "outcomes": dict(outcomes),
        "terminations": dict(terminations),
        "avg_num_plies": statistics.mean(record["num_plies"] for record in records) if records else None,
    }


def print_summary_record(record: dict) -> None:
    print(json.dumps(record, indent=2, sort_keys=True))


def bucketed_loss_by_ply(
    *,
    model_ref: str,
    games: list[list[int]],
    eos_id: int,
    max_ply: int,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    model, _, _, _ = load_model(model_ref, device)
    sums = np.zeros(max_ply + 1, dtype=np.float64)
    counts = np.zeros(max_ply + 1, dtype=np.int64)

    for start in range(0, len(games), batch_size):
        batch = games[start:start + batch_size]
        if not batch:
            continue
        max_len = max(len(game) for game in batch)
        x = torch.full((len(batch), max_len - 1), eos_id, dtype=torch.long, device=device)
        y = torch.full((len(batch), max_len - 1), eos_id, dtype=torch.long, device=device)
        mask = torch.zeros((len(batch), max_len - 1), dtype=torch.bool, device=device)
        ply = torch.zeros((len(batch), max_len - 1), dtype=torch.long, device=device)

        for row, game in enumerate(batch):
            seq = torch.tensor(game, dtype=torch.long, device=device)
            valid = len(game) - 1
            x[row, :valid] = seq[:-1]
            y[row, :valid] = seq[1:]
            mask[row, :valid] = True

            game_ply = 0
            for idx in range(valid):
                target = game[idx + 1]
                if target == eos_id:
                    continue
                game_ply += 1
                if game_ply <= max_ply:
                    ply[row, idx] = game_ply

        logits = model(x)
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="none",
        ).reshape_as(y)

        valid_mask = mask & (y != eos_id) & (ply > 0) & (ply <= max_ply)
        if not valid_mask.any():
            continue
        valid_losses = per_token[valid_mask].detach().cpu().numpy()
        valid_ply = ply[valid_mask].detach().cpu().numpy()
        sums += np.bincount(valid_ply, weights=valid_losses, minlength=max_ply + 1)
        counts += np.bincount(valid_ply, minlength=max_ply + 1)

        done = min(start + len(batch), len(games))
        print(f"{model_ref}: {done}/{len(games)} games")

    return sums, counts


@torch.inference_mode()
def command_loss_by_ply(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    games, meta = load_id_games(Path(args.data_dir), args.split, args.max_games)
    eos_id = meta["stoi"]["<eos>"]
    model_summaries = []

    for model_ref in args.models:
        sums, counts = bucketed_loss_by_ply(
            model_ref=model_ref,
            games=games,
            eos_id=eos_id,
            max_ply=args.max_ply,
            batch_size=args.batch_size,
            device=device,
        )
        by_ply = []
        values = []
        for ply_idx in range(1, args.max_ply + 1):
            count = int(counts[ply_idx])
            loss = float(sums[ply_idx] / count) if count else None
            by_ply.append({"ply": ply_idx, "loss": loss, "count": count})
            if loss is not None:
                values.append((ply_idx, loss))
        peak_ply, peak_loss = max(values, key=lambda item: item[1]) if values else (None, None)
        model_summaries.append(
            {
                "model": model_ref,
                "positions": int(counts.sum()),
                "avg_loss": float(sums.sum() / counts.sum()) if counts.sum() else None,
                "peak_ply": peak_ply,
                "peak_loss": peak_loss,
                "by_ply": by_ply,
            }
        )
        if peak_ply is not None:
            print(f"{model_ref}: peak loss {peak_loss:.3f} at ply {peak_ply}")

    summary = {
        "type": "loss_by_ply_summary",
        "schema_version": SCHEMA_VERSION,
        "device": device,
        "data_dir": args.data_dir,
        "split": args.split,
        "games": len(games),
        "max_ply": args.max_ply,
        "models": model_summaries,
    }
    records = [run_record("loss-by-ply", args, {"device": device}), summary]
    write_jsonl(args.output, records)
    print_summary_record(summary)
    print(f"wrote {args.output}")


def command_summarize(args: argparse.Namespace) -> None:
    summaries = []
    games = []
    for path in args.paths:
        with open(path) as f:
            for line in f:
                record = json.loads(line)
                if record.get("type", "").endswith("_summary"):
                    summaries.append((path, record))
                elif record.get("type") == "game":
                    games.append(record)

    for path, record in summaries:
        print(f"\n{path}: {record['type']}")
        print_summary_record(record)
    if games:
        print("\ncombined games:")
        print_summary_record(summarize_game_records(games))


def add_common_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help=model_ref_help())
    parser.add_argument("--device", default="", choices=["", "cuda", "cpu", "mps"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", required=True)


def add_position_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-games", type=int, default=0)
    parser.add_argument("--max-positions", type=int, default=4096)
    parser.add_argument("--max-ply", type=int, default=140)


def add_stockfish_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stockfish", default="/opt/homebrew/bin/fairy-stockfish")
    parser.add_argument("--stockfish-time", type=float, default=0.02)
    parser.add_argument("--stockfish-depth", type=int, default=0)
    parser.add_argument("--stockfish-nodes", type=int, default=0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified nanoDanya benchmark runner")
    sub = parser.add_subparsers(dest="command", required=True)

    legality = sub.add_parser("legality", help="Raw top-1 legality and legal probability mass")
    add_common_model_args(legality)
    add_position_args(legality)
    legality.add_argument("--allow-eos", action="store_true")
    legality.add_argument("--top-illegal", type=int, default=20)
    legality.set_defaults(func=command_legality)

    termination = sub.add_parser("termination", help="Raw EOS propensity on non-terminal positions")
    add_common_model_args(termination)
    add_position_args(termination)
    termination.set_defaults(func=command_termination)

    move = sub.add_parser("move-quality", help="Fixed-position move quality scored by Stockfish")
    add_common_model_args(move)
    add_position_args(move)
    add_stockfish_args(move)
    move.add_argument("--write-positions", action="store_true")
    move.set_defaults(func=command_move_quality)

    loss = sub.add_parser("loss-by-ply", help="Held-out next-token loss bucketed by ply")
    loss.add_argument("--models", nargs="+", required=True, help=model_ref_help())
    loss.add_argument("--device", default="", choices=["", "cuda", "cpu", "mps"])
    loss.add_argument("--data-dir", default="data/processed")
    loss.add_argument("--split", default="val")
    loss.add_argument("--max-games", type=int, default=4000)
    loss.add_argument("--batch-size", type=int, default=32)
    loss.add_argument("--max-ply", type=int, default=140)
    loss.add_argument("--output", required=True)
    loss.set_defaults(func=command_loss_by_ply)

    games = sub.add_parser("games", help="Full scaffolded games")
    add_common_model_args(games)
    add_stockfish_args(games)
    games.add_argument("--game-mode", choices=["stockfish", "h2h"], default="stockfish")
    games.add_argument("--opponent-model", default="")
    games.add_argument("--stockfish-elo", type=int, default=500)
    games.add_argument("--games", type=int, default=50)
    games.add_argument("--game-offset", type=int, default=0)
    games.add_argument("--max-plies", type=int, default=200)
    games.add_argument("--temperature", type=float, default=0.8)
    games.add_argument("--prompt", default="")
    games.add_argument("--allow-eos", action="store_true")
    games.add_argument("--legal-mask", action=argparse.BooleanOptionalAction, default=True)
    games.add_argument("--kv-cache", action="store_true", help="Use batched KV-cache decoding for h2h games")
    games.set_defaults(func=command_games)

    summarize = sub.add_parser("summarize", help="Summarize benchmark JSONL artifacts")
    summarize.add_argument("paths", nargs="+")
    summarize.set_defaults(func=command_summarize)

    return parser


def main() -> None:
    random.seed(0)
    torch.manual_seed(0)
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "game_mode", None) == "h2h" and not args.opponent_model:
        parser.error("games --game-mode h2h requires --opponent-model")
    t0 = time.time()
    args.func(args)
    print(f"elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
