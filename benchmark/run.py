from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import re
import statistics
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import chess
import chess.engine
import numpy as np
import requests
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
    under_disambiguated_legal_matches,
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
class PromptPosition:
    board: chess.Board
    moves: list[str]
    ply: int
    game_index: int


@dataclass
class LegalityStats:
    positions: int = 0
    illegal_top1: int = 0
    under_disambiguated_top1: int = 0
    legal_mass_sum: float = 0.0

    def add(self, *, raw_top1_legal: bool, under_disambiguated: bool, legal_mass: float) -> None:
        self.positions += 1
        self.illegal_top1 += 0 if raw_top1_legal else 1
        self.under_disambiguated_top1 += 1 if under_disambiguated else 0
        self.legal_mass_sum += legal_mass

    def as_metrics(self) -> dict:
        if self.positions == 0:
            return {
                "positions": 0,
                "raw_top1_illegal": 0,
                "raw_top1_under_disambiguated": 0,
                "raw_top1_real_illegal": 0,
                "raw_top1_illegal_rate": None,
                "raw_top1_real_illegal_rate": None,
                "avg_legal_mass": None,
            }
        real_illegal = self.illegal_top1 - self.under_disambiguated_top1
        return {
            "positions": self.positions,
            "raw_top1_illegal": self.illegal_top1,
            "raw_top1_under_disambiguated": self.under_disambiguated_top1,
            "raw_top1_real_illegal": real_illegal,
            "raw_top1_illegal_rate": self.illegal_top1 / self.positions,
            "raw_top1_real_illegal_rate": real_illegal / self.positions,
            "avg_legal_mass": self.legal_mass_sum / self.positions,
        }


@dataclass
class ApiLegalityStats:
    positions: int = 0
    api_errors: int = 0
    parseable: int = 0
    legal: int = 0

    def add(self, *, api_error: bool, parseable: bool, legal: bool) -> None:
        self.positions += 1
        self.api_errors += 1 if api_error else 0
        self.parseable += 1 if parseable else 0
        self.legal += 1 if legal else 0

    def as_metrics(self) -> dict:
        if self.positions == 0:
            return {
                "positions": 0,
                "api_error_rate": None,
                "parseable_rate": None,
                "legal_rate": None,
            }
        return {
            "positions": self.positions,
            "api_error_rate": self.api_errors / self.positions,
            "parseable_rate": self.parseable / self.positions,
            "legal_rate": self.legal / self.positions,
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


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


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


def collect_prompt_positions(
    *,
    data_dir: Path,
    split: str,
    max_games: int,
    max_positions: int,
    max_ply: int,
) -> list[PromptPosition]:
    games, _ = load_token_games(data_dir, split, max_games)
    positions: list[PromptPosition] = []

    for game_index, game in enumerate(games):
        board = chess.Board()
        moves: list[str] = []
        ply = 0

        for token in game[1:]:
            if token == "<eos>":
                break

            ply += 1
            if max_ply <= 0 or ply <= max_ply:
                positions.append(
                    PromptPosition(
                        board=board.copy(stack=False),
                        moves=moves.copy(),
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
            moves.append(token)

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
    under_disambiguated_matches = under_disambiguated_legal_matches(raw_top1, legal_san)

    probs = torch.softmax(logits, dim=-1)
    legal_ids = legal_token_ids(stoi, board, allow_eos=allow_eos)
    legal_mass = float(probs[legal_ids].sum().item()) if legal_ids else 0.0

    return {
        "raw_top1": raw_top1,
        "raw_top1_legal": raw_top1_legal,
        "raw_top1_under_disambiguated": bool(under_disambiguated_matches),
        "raw_top1_under_disambiguated_matches": under_disambiguated_matches,
        "raw_top1_prob": round(float(probs[raw_top1_idx].item()), 6),
        "legal_mass": legal_mass,
    }


def phase_for_ply(ply: int) -> str:
    if ply <= 20:
        return "opening"
    if ply <= 80:
        return "middlegame"
    return "endgame"


def format_move_history(moves: list[str]) -> str:
    if not moves:
        return "(start position)"
    chunks = []
    for idx in range(0, len(moves), 2):
        move_no = idx // 2 + 1
        white = moves[idx]
        if idx + 1 < len(moves):
            chunks.append(f"{move_no}. {white} {moves[idx + 1]}")
        else:
            chunks.append(f"{move_no}. {white}")
    return " ".join(chunks)


def api_legality_prompt(position: PromptPosition) -> list[dict[str, str]]:
    side = "White" if position.board.turn == chess.WHITE else "Black"
    return [
        {
            "role": "system",
            "content": "You are a chess player. Return exactly one legal SAN chess move and nothing else.",
        },
        {
            "role": "user",
            "content": (
                f"Game so far:\n{format_move_history(position.moves)}\n\n"
                f"Side to move: {side}.\n"
                "Return exactly one legal SAN move."
            ),
        },
    ]


SAN_CANDIDATE_RE = re.compile(
    r"(?:O-O-O|O-O|0-0-0|0-0|[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?)[+#]?[!?]*"
)


def clean_san_candidate(candidate: str) -> str:
    candidate = candidate.strip().strip("`'\"")
    candidate = re.sub(r"^(?:move|answer)\s*:\s*", "", candidate, flags=re.IGNORECASE).strip()
    candidate = candidate.rstrip(".,;")
    if candidate in {"0-0", "0-0+", "0-0#", "0-0-0", "0-0-0+", "0-0-0#"}:
        candidate = candidate.replace("0", "O")
    candidate = re.sub(r"[!?]+$", "", candidate)
    return candidate


def san_candidates(raw_response: str) -> list[str]:
    cleaned = raw_response.strip()
    cleaned = cleaned.replace("```", "").strip()
    candidates = [match.group(0) for match in SAN_CANDIDATE_RE.finditer(cleaned)]

    seen = set()
    unique = []
    for candidate in candidates:
        candidate = clean_san_candidate(candidate)
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def parse_api_move(board: chess.Board, raw_response: str) -> tuple[str | None, bool]:
    candidates = san_candidates(raw_response)
    if not candidates:
        return None, False

    for candidate in candidates:
        try:
            move = board.parse_san(candidate)
        except ValueError:
            continue
        return board.san(move), True

    return candidates[0], False


def call_openrouter(
    *,
    model: str,
    messages: list[dict[str, str]],
    api_key: str,
    base_url: str,
    app_name: str,
    site_url: str,
    timeout: float,
    retries: int,
    retry_sleep: float,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": app_name,
    }
    if site_url:
        headers["HTTP-Referer"] = site_url

    uses_gemini_thinking = model.startswith("google/gemini-3.")
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 96 if uses_gemini_thinking else 32,
        "include_reasoning": False,
    }
    if uses_gemini_thinking:
        payload["reasoning"] = {"effort": "minimal", "exclude": True}
    url = f"{base_url.rstrip('/')}/chat/completions"

    last_error = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504} and attempt < retries:
                time.sleep(retry_sleep * (attempt + 1))
                continue
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"].get("content")
            if content is None:
                return ""
            if isinstance(content, str):
                return content
            return json.dumps(content)
        except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_sleep * (attempt + 1))
                continue
    raise RuntimeError(str(last_error))


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
            under_disambiguated=metrics["raw_top1_under_disambiguated"],
            legal_mass=metrics["legal_mass"],
        )
        by_phase[phase_for_ply(pos.ply)].add(
            raw_top1_legal=metrics["raw_top1_legal"],
            under_disambiguated=metrics["raw_top1_under_disambiguated"],
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


def evaluate_api_legality_position(
    *,
    model: str,
    pos: PromptPosition,
    api_key: str,
    args: argparse.Namespace,
) -> dict:
    phase = phase_for_ply(pos.ply)
    messages = api_legality_prompt(pos)
    raw_response = ""
    parsed_move = None
    legal = False
    api_error = False
    error = ""

    try:
        raw_response = call_openrouter(
            model=model,
            messages=messages,
            api_key=api_key,
            base_url=args.base_url,
            app_name=args.app_name,
            site_url=args.site_url,
            timeout=args.timeout,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        )
        parsed_move, legal = parse_api_move(pos.board, raw_response)
    except RuntimeError as exc:
        api_error = True
        error = str(exc)[:300]

    parseable = parsed_move is not None
    record = {
        "type": "api_legality_position",
        "schema_version": SCHEMA_VERSION,
        "provider": "openrouter",
        "model": model,
        "game_index": pos.game_index,
        "ply": pos.ply,
        "phase": phase,
        "fen": pos.board.fen(),
        "side": "white" if pos.board.turn == chess.WHITE else "black",
        "moves": pos.moves,
        "raw_response": raw_response,
        "parsed_move": parsed_move,
        "parseable": parseable,
        "legal": legal,
        "api_error": api_error,
    }
    if error:
        record["error"] = error
    if args.write_prompts:
        record["messages"] = messages
    return record


def api_position_key(record: dict) -> tuple[int, int, str]:
    return (int(record["game_index"]), int(record["ply"]), str(record["fen"]))


def command_api_legality(args: argparse.Namespace) -> None:
    load_dotenv()
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit("Set OPENROUTER_API_KEY or pass --api-key")

    positions = collect_prompt_positions(
        data_dir=Path(args.data_dir),
        split=args.split,
        max_games=args.max_games,
        max_positions=args.max_positions,
        max_ply=args.max_ply,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    safe_args = argparse.Namespace(**(vars(args) | {"api_key": "<redacted>" if args.api_key else ""}))

    existing_by_model: dict[str, dict[tuple[int, int, str], dict]] = defaultdict(dict)
    if args.resume and out.exists():
        with out.open() as existing_file:
            for line in existing_file:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("type") == "api_legality_position":
                    if record.get("api_error"):
                        continue
                    existing_by_model[record["model"]][api_position_key(record)] = record

    mode = "a" if args.resume and out.exists() else "w"
    with out.open(mode) as f:
        if mode == "w":
            f.write(json.dumps(run_record("api-legality", safe_args, {"provider": "openrouter"}), sort_keys=True) + "\n")

        for model in args.models:
            overall = ApiLegalityStats()
            by_phase: dict[str, ApiLegalityStats] = defaultdict(ApiLegalityStats)
            top_illegal = Counter()
            top_unparseable = Counter()
            existing = existing_by_model.get(model, {})
            remaining_positions = [
                pos for pos in positions
                if (pos.game_index, pos.ply, pos.board.fen()) not in existing
            ]
            completed = 0

            for record in existing.values():
                phase = record["phase"]
                api_error = bool(record.get("api_error"))
                parseable = bool(record["parseable"])
                legal = bool(record["legal"])
                parsed_move = record["parsed_move"]
                raw_response = record["raw_response"]

                overall.add(api_error=api_error, parseable=parseable, legal=legal)
                by_phase[phase].add(api_error=api_error, parseable=parseable, legal=legal)
                if api_error:
                    top_unparseable["<api_error>"] += 1
                elif not parseable:
                    top_unparseable[raw_response[:120]] += 1
                elif not legal:
                    top_illegal[parsed_move] += 1

            if existing:
                print(f"{model}: resuming from {len(existing)}/{len(positions)} existing positions")

            with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                futures = [
                    executor.submit(
                        evaluate_api_legality_position,
                        model=model,
                        pos=pos,
                        api_key=api_key,
                        args=args,
                    )
                    for pos in remaining_positions
                ]

                for future in as_completed(futures):
                    record = future.result()
                    phase = record["phase"]
                    api_error = bool(record["api_error"])
                    parseable = bool(record["parseable"])
                    legal = bool(record["legal"])
                    parsed_move = record["parsed_move"]
                    raw_response = record["raw_response"]

                    overall.add(api_error=api_error, parseable=parseable, legal=legal)
                    by_phase[phase].add(api_error=api_error, parseable=parseable, legal=legal)

                    if api_error:
                        top_unparseable["<api_error>"] += 1
                    elif not parseable:
                        top_unparseable[raw_response[:120]] += 1
                    elif not legal:
                        top_illegal[parsed_move] += 1

                    if args.write_positions:
                        f.write(json.dumps(record, sort_keys=True) + "\n")

                    completed += 1
                    total_done = len(existing) + completed
                    if args.progress_every > 0 and total_done % args.progress_every == 0:
                        print(f"{model}: {total_done}/{len(positions)} positions")
                    if args.sleep > 0:
                        time.sleep(args.sleep)

            summary = {
                "type": "api_legality_summary",
                "schema_version": SCHEMA_VERSION,
                "provider": "openrouter",
                "model": model,
                "metrics": overall.as_metrics(),
                "by_phase": {phase: stats.as_metrics() for phase, stats in sorted(by_phase.items())},
                "top_illegal_parsed_moves": top_illegal.most_common(args.top_errors),
                "top_unparseable_responses": top_unparseable.most_common(args.top_errors),
            }
            f.write(json.dumps(summary, sort_keys=True) + "\n")
            f.flush()
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
    model, model_config, stoi, itos = load_model(args.model, device)
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
    best_move_matches = 0
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
        tokens_by_index = {}
        for idx, pos in enumerate(positions):
            _nid, token = choose_move_from_logits(
                logits_by_index[idx], pos.board, stoi, itos,
                temperature=0.0, allow_eos=False, legal_mask=True,
            )
            if token is not None:
                tokens_by_index[idx] = token

        for idx, pos in enumerate(positions):
            side = pos.board.turn
            token = tokens_by_index.get(idx)
            if token is None:
                failures += 1
                continue

            board_after = pos.board.copy(stack=False)
            try:
                board_after.push_san(token)
            except ValueError:
                failures += 1
                continue

            before_cp = cp_for_side(engine, pos.board, side, limit)
            best_move = engine.play(pos.board, limit).move
            best_san = strip_san(pos.board.san(best_move))
            is_best = strip_san(token) == best_san
            after_cp = cp_for_side(engine, board_after, side, limit)
            cp_loss = max(0, before_cp - after_cp)
            losses.append(cp_loss)
            best_move_matches += int(is_best)
            if args.write_positions:
                records.append({
                    "type": "move_quality_position",
                    "schema_version": SCHEMA_VERSION,
                    "model": args.model,
                    "ply": pos.ply,
                    "fen": pos.board.fen(),
                    "played": strip_san(token),
                    "stockfish_best": best_san,
                    "best_move_match": is_best,
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
        "best_move_match_rate": best_move_matches / len(losses) if losses else None,
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


def configure_stockfish_strength(engine: chess.engine.SimpleEngine, requested_elo: int) -> None:
    if requested_elo <= 0:
        return
    elo_option = engine.options.get("UCI_Elo")
    if elo_option is None:
        raise ValueError("Stockfish engine does not expose UCI_Elo")
    if elo_option.min is not None and requested_elo < elo_option.min:
        raise ValueError(
            f"--stockfish-elo {requested_elo} is below this engine's UCI_Elo minimum "
            f"of {elo_option.min}; use a Fairy-Stockfish binary with the extended Elo range"
        )
    if elo_option.max is not None and requested_elo > elo_option.max:
        raise ValueError(
            f"--stockfish-elo {requested_elo} is above this engine's UCI_Elo maximum of {elo_option.max}"
        )
    engine.configure({"UCI_LimitStrength": True, "UCI_Elo": requested_elo})


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
    opponent = args.opponent_model
    if args.game_mode == "stockfish":
        opponent = f"sf{args.stockfish_elo}"
    return {
        "type": "game",
        "schema_version": SCHEMA_VERSION,
        "game_id": state.game_id,
        "game_num": state.game_num,
        "mode": args.game_mode,
        "model": args.model,
        "opponent_model": args.opponent_model,
        "opponent": opponent,
        "stockfish_elo": args.stockfish_elo if args.game_mode == "stockfish" else None,
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

    def choose_model_moves(turn_states):
        return choose_for_states(
            states=turn_states,
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
        configure_stockfish_strength(engine, args.stockfish_elo)
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

                choices = choose_model_moves(model_turns)
                for state, next_id, token in choices:
                    apply_model_choice(state, "model", next_id, token, tokenizers, args.allow_eos)

            else:
                model_turns = [s for s in states if not s.finished and s.board.turn == s.model_color]
                opponent_turns = [s for s in states if not s.finished and s.board.turn != s.model_color]
                for model_name, turn_states, mdl, s_to_i, i_to_s in (
                    ("model", model_turns, model, stoi, itos),
                    ("opponent", opponent_turns, opponent, opponent_stoi, opponent_itos),
                ):
                    if model_name == "model":
                        choices = choose_model_moves(turn_states)
                    else:
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


PUZZLE_ID_RE = re.compile(rb'"id":"([^"]+)"')


@dataclass
class PuzzleState:
    rating: int
    bin: int
    puzzle_id: str
    game_id: str
    prefix: list[int]
    board: chess.Board
    sol: list[str]
    ptr: int = 1
    first_correct: bool | None = None
    solved: bool = True
    alive: bool = True


def sample_puzzles_by_rating(
    *,
    ndjson: Path,
    metadata: Path,
    rating_min: int,
    rating_max: int,
    bin_width: int,
    per_bin: int,
    scan_cap: int,
) -> tuple[list[dict], dict[int, int], int]:
    print(f"loading metadata {metadata} ...", flush=True)
    meta = json.load(metadata.open())
    print(f"metadata games: {len(meta):,}", flush=True)

    counts = {b: 0 for b in range(rating_min, rating_max, bin_width)}
    chosen: list[dict] = []
    scanned = 0

    with ndjson.open("rb") as f:
        for line in f:
            scanned += 1
            if scan_cap and scanned > scan_cap:
                break
            match = PUZZLE_ID_RE.search(line)
            if not match:
                continue
            gid = match.group(1).decode()
            puzzles = meta.get(gid)
            if not puzzles:
                continue
            rating = puzzles[0]["rating"]
            if rating < rating_min or rating >= rating_max:
                continue
            b = rating_min + ((rating - rating_min) // bin_width) * bin_width
            if counts[b] >= per_bin:
                continue
            game = json.loads(line)
            chosen.append({"game_id": gid, "moves": game["moves"].split(), "puzzle": puzzles[0], "bin": b})
            counts[b] += 1
            if all(c >= per_bin for c in counts.values()):
                break

    print(f"scanned {scanned:,} games, chose {len(chosen)} puzzles", flush=True)
    return chosen, counts, scanned


def build_puzzle_state(item: dict, stoi: dict[str, int], max_len: int) -> PuzzleState | None:
    """Replay the source game to the puzzle position; return a solvable state or None if unusable."""
    moves = item["moves"]
    p = item["puzzle"]
    move_num = p["move_num"]
    sol = p["moves"].split()
    if move_num - 1 > len(moves):
        return None

    board = chess.Board()
    prefix = [stoi["<bos>"]]
    for san in moves[: move_num - 1]:
        tid = resolve_token_id(stoi, san)
        if tid is None:
            return None
        try:
            board.push_san(san)
        except ValueError:
            return None
        prefix.append(tid)

    if board.fen() != p["fen"]:
        return None

    try:
        lead = board.parse_uci(sol[0])
    except ValueError:
        return None
    lead_tid = resolve_token_id(stoi, board.san(lead))
    if lead_tid is None:
        return None
    prefix.append(lead_tid)
    board.push(lead)
    if len(prefix) >= max_len:
        return None

    return PuzzleState(
        rating=p["rating"],
        bin=item["bin"],
        puzzle_id=p["puzzle_id"],
        game_id=item["game_id"],
        prefix=prefix,
        board=board,
        sol=sol,
    )


@torch.inference_mode()
def solve_puzzles(
    states: list[PuzzleState],
    model,
    stoi: dict[str, int],
    itos,
    *,
    device: str,
    batch_size: int,
    max_len: int,
) -> None:
    """Batched, multi-step solve. Each round every live puzzle is at a solver turn; we predict its
    move, and if correct apply it plus the forced opponent reply, then re-batch the survivors."""
    eos_id = stoi["<eos>"]
    for _ in range(64):
        active = [s for s in states if s.alive]
        if not active:
            break
        prefixes = [s.prefix for s in active]
        for idx, logits in last_logits_for_prefixes(
            model, prefixes, pad_id=eos_id, device=device, batch_size=batch_size
        ):
            s = active[idx]
            next_id, token = choose_move_from_logits(
                logits, s.board, stoi, itos, temperature=0.0, allow_eos=False, legal_mask=True
            )
            expected = s.board.parse_uci(s.sol[s.ptr])
            model_move = s.board.parse_san(token) if token else None
            correct = model_move == expected
            if not correct and model_move is not None:
                probe = s.board.copy(stack=False)
                probe.push(model_move)
                if probe.is_checkmate():
                    correct = True  # any mate ends the puzzle (Lichess accepts it)

            if s.ptr == 1:
                s.first_correct = correct
            if not correct:
                s.solved = False
                s.alive = False
                continue

            s.board.push(model_move)
            s.prefix.append(next_id)
            s.ptr += 1
            if model_move != expected:  # accepted alternative mate; line ends here
                s.alive = False
                continue
            if s.ptr >= len(s.sol):
                s.alive = False  # matched the whole solver line
                continue

            reply = s.board.parse_uci(s.sol[s.ptr])
            reply_tid = resolve_token_id(stoi, s.board.san(reply))
            s.board.push(reply)
            s.ptr += 1
            if reply_tid is None or len(s.prefix) + 1 >= max_len:
                s.alive = False
                if s.ptr < len(s.sol):
                    s.solved = False  # more solver moves remain but we cannot probe further
                continue
            s.prefix.append(reply_tid)


def maybe_plot_puzzle_curve(by_bin: list[dict], path: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    scored = [b for b in by_bin if b["n"] > 0]
    centers = [(b["rating_lo"] + b["rating_hi"]) / 2 for b in scored]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(centers, [b["full_solve_acc"] for b in scored], "o-", label="full solve")
    ax.plot(centers, [b["first_move_acc"] for b in scored], "s--", color="gray", label="first move")
    ax.set_xlabel("puzzle rating")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Puzzle accuracy vs rating")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


@torch.inference_mode()
def command_puzzles(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, config, stoi, itos = load_model(args.model, device)
    max_len = config.sequence_len

    items, _bin_counts, scanned = sample_puzzles_by_rating(
        ndjson=Path(args.ndjson),
        metadata=Path(args.metadata),
        rating_min=args.rating_min,
        rating_max=args.rating_max,
        bin_width=args.bin_width,
        per_bin=args.per_bin,
        scan_cap=args.scan_cap,
    )

    states: list[PuzzleState] = []
    skipped = 0
    for item in items:
        state = build_puzzle_state(item, stoi, max_len)
        if state is None:
            skipped += 1
            continue
        states.append(state)
    print(f"built {len(states)} solvable puzzles, skipped {skipped}", flush=True)

    solve_puzzles(states, model, stoi, itos, device=device, batch_size=args.batch_size, max_len=max_len)

    agg: dict[int, dict] = {}
    for s in states:
        d = agg.setdefault(s.bin, {"n": 0, "first": 0, "full": 0})
        d["n"] += 1
        d["first"] += int(bool(s.first_correct))
        d["full"] += int(bool(s.solved))

    by_bin = []
    for b in sorted(agg):
        d = agg[b]
        by_bin.append({
            "rating_lo": b,
            "rating_hi": b + args.bin_width,
            "n": d["n"],
            "first_move_acc": d["first"] / d["n"],
            "full_solve_acc": d["full"] / d["n"],
        })

    n = len(states)
    summary = {
        "type": "puzzles_summary",
        "schema_version": SCHEMA_VERSION,
        "model": args.model,
        "device": device,
        "puzzles_scored": n,
        "puzzles_skipped": skipped,
        "games_scanned": scanned,
        "first_move_acc": sum(s.first_correct for s in states) / n if n else None,
        "full_solve_acc": sum(s.solved for s in states) / n if n else None,
        "bin_width": args.bin_width,
        "by_bin": by_bin,
    }

    records = [run_record("puzzles", args, {"device": device}), summary]
    if args.write_positions:
        records.extend({
            "type": "puzzle_position",
            "schema_version": SCHEMA_VERSION,
            "puzzle_id": s.puzzle_id,
            "game_id": s.game_id,
            "rating": s.rating,
            "depth": len(s.sol) // 2,
            "first_move_correct": s.first_correct,
            "full_solved": s.solved,
        } for s in states)
    write_jsonl(args.output, records)

    plot_path = args.plot or str(Path(args.output).with_suffix(".png"))
    if maybe_plot_puzzle_curve(by_bin, plot_path):
        print(f"wrote plot {plot_path}")
    else:
        print("matplotlib unavailable; skipped plot")

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

    api_legality = sub.add_parser("api-legality", help="Prompted API-model legality on fixed positions")
    api_legality.add_argument("--models", nargs="+", required=True)
    add_position_args(api_legality)
    api_legality.add_argument("--output", required=True)
    api_legality.add_argument("--api-key", default="")
    api_legality.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    api_legality.add_argument("--app-name", default="nanoDanya benchmark")
    api_legality.add_argument("--site-url", default="")
    api_legality.add_argument("--timeout", type=float, default=30.0)
    api_legality.add_argument("--retries", type=int, default=2)
    api_legality.add_argument("--retry-sleep", type=float, default=2.0)
    api_legality.add_argument("--concurrency", type=int, default=8)
    api_legality.add_argument("--sleep", type=float, default=0.0)
    api_legality.add_argument("--progress-every", type=int, default=100)
    api_legality.add_argument("--resume", action="store_true")
    api_legality.add_argument("--top-errors", type=int, default=20)
    api_legality.add_argument("--write-positions", action=argparse.BooleanOptionalAction, default=True)
    api_legality.add_argument("--write-prompts", action="store_true")
    api_legality.set_defaults(func=command_api_legality)

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
    games.add_argument("--seed", type=int, default=0)
    games.add_argument("--allow-eos", action="store_true")
    games.add_argument("--legal-mask", action=argparse.BooleanOptionalAction, default=True)
    games.add_argument("--kv-cache", action="store_true", help="Use batched KV-cache decoding for h2h games")
    games.set_defaults(func=command_games)

    puzzles = sub.add_parser("puzzles", help="Puzzle-solve accuracy binned by puzzle rating")
    add_common_model_args(puzzles)
    puzzles.add_argument("--ndjson", default="data/puzzle_games_ndjson.txt")
    puzzles.add_argument("--metadata", default="data/puzzle_metadata.txt")
    puzzles.add_argument("--rating-min", type=int, default=600)
    puzzles.add_argument("--rating-max", type=int, default=2800)
    puzzles.add_argument("--bin-width", type=int, default=100)
    puzzles.add_argument("--per-bin", type=int, default=400)
    puzzles.add_argument("--scan-cap", type=int, default=0, help="max ndjson lines to scan (0 = no cap)")
    puzzles.add_argument("--write-positions", action="store_true")
    puzzles.add_argument("--plot", default="", help="PNG path (defaults to output with .png)")
    puzzles.set_defaults(func=command_puzzles)

    summarize = sub.add_parser("summarize", help="Summarize benchmark JSONL artifacts")
    summarize.add_argument("paths", nargs="+")
    summarize.set_defaults(func=command_summarize)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    seed = getattr(args, "seed", 0)
    random.seed(seed)
    torch.manual_seed(seed)
    if getattr(args, "game_mode", None) == "h2h" and not args.opponent_model:
        parser.error("games --game-mode h2h requires --opponent-model")
    t0 = time.time()
    args.func(args)
    print(f"elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
