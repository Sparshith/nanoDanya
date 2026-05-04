from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from time import perf_counter
from typing import Mapping, Sequence

import chess
import numpy as np
import torch

from chess_token_utils import normalized_legal_sans, strip_san


@dataclass(frozen=True)
class LegalTargets:
    flat_positions: torch.Tensor
    flat_token_ids: torch.Tensor
    valid_mask: torch.Tensor


@dataclass(frozen=True)
class LegalTargetStats:
    total_s: float
    replay_s: float
    legal_ids_s: float
    target_advance_s: float
    tensorize_s: float
    rows: int
    positions: int
    flat_legal_ids: int


def _token_items(itos: Sequence[str] | Mapping[int, str]):
    if isinstance(itos, Mapping):
        return itos.items()
    return enumerate(itos)


def build_normalized_token_id_map(itos: Sequence[str] | Mapping[int, str]) -> dict[str, tuple[int, ...]]:
    san_to_ids: dict[str, list[int]] = defaultdict(list)
    for idx, token in _token_items(itos):
        if token.startswith("<"):
            continue
        san_to_ids[strip_san(token)].append(idx)
    return {san: tuple(ids) for san, ids in san_to_ids.items()}


def _advance_board(board: chess.Board, token: str) -> None:
    if token == "<bos>":
        board.reset()
        return
    if token == "<eos>":
        return
    try:
        board.push_san(token)
    except ValueError as exc:
        raise ValueError(f"Could not replay token {token!r} from board {board.fen()}") from exc


def _legal_token_ids(
    board: chess.Board,
    san_to_ids: dict[str, tuple[int, ...]],
    *,
    eos_id: int,
    allow_eos: bool,
) -> list[int]:
    legal_ids = [eos_id] if allow_eos else []
    for san in normalized_legal_sans(board):
        legal_ids.extend(san_to_ids.get(san, ()))
    return legal_ids


def build_legal_targets(
    *,
    token_stream: np.ndarray,
    start_indices: torch.Tensor | np.ndarray,
    target_batch: torch.Tensor | np.ndarray,
    itos: Sequence[str] | Mapping[int, str],
    bos_positions: np.ndarray,
    san_to_ids: dict[str, tuple[int, ...]],
    bos_id: int,
    eos_id: int,
    device: torch.device | str,
    allow_eos: bool = True,
    return_stats: bool = False,
) -> LegalTargets | tuple[LegalTargets, LegalTargetStats]:
    token_stream = np.asarray(token_stream)
    bos_positions = np.asarray(bos_positions)
    total_t0 = perf_counter() if return_stats else 0.0

    if isinstance(start_indices, torch.Tensor):
        start_idx_np = start_indices.cpu().numpy()
    else:
        start_idx_np = np.asarray(start_indices)

    if isinstance(target_batch, torch.Tensor):
        target_np = target_batch.cpu().numpy()
    else:
        target_np = np.asarray(target_batch)

    batch_size, block_size = target_np.shape
    valid_mask = np.zeros((batch_size, block_size), dtype=np.bool_)
    flat_positions: list[int] = []
    flat_token_ids: list[int] = []
    replay_s = 0.0
    legal_ids_s = 0.0
    target_advance_s = 0.0

    for row, start in enumerate(start_idx_np.tolist()):
        board = chess.Board()
        bos_idx = np.searchsorted(bos_positions, start, side="right") - 1
        replay_start = int(bos_positions[bos_idx]) if bos_idx >= 0 else 0

        replay_t0 = perf_counter() if return_stats else 0.0
        for token_id in token_stream[replay_start : start + 1]:
            _advance_board(board, itos[int(token_id)])
        if return_stats:
            replay_s += perf_counter() - replay_t0

        for col in range(block_size):
            target_id = int(target_np[row, col])
            if target_id != bos_id:
                legal_t0 = perf_counter() if return_stats else 0.0
                legal_ids = _legal_token_ids(board, san_to_ids, eos_id=eos_id, allow_eos=allow_eos)
                if return_stats:
                    legal_ids_s += perf_counter() - legal_t0
                if not legal_ids:
                    raise ValueError(f"No legal target ids available for board {board.fen()}")
                flat_index = row * block_size + col
                flat_positions.extend([flat_index] * len(legal_ids))
                flat_token_ids.extend(legal_ids)
                valid_mask[row, col] = True

            advance_t0 = perf_counter() if return_stats else 0.0
            _advance_board(board, itos[target_id])
            if return_stats:
                target_advance_s += perf_counter() - advance_t0

    tensorize_t0 = perf_counter() if return_stats else 0.0
    targets = LegalTargets(
        flat_positions=torch.tensor(flat_positions, dtype=torch.long, device=device),
        flat_token_ids=torch.tensor(flat_token_ids, dtype=torch.long, device=device),
        valid_mask=torch.from_numpy(valid_mask).to(device=device),
    )
    if not return_stats:
        return targets

    tensorize_s = perf_counter() - tensorize_t0
    stats = LegalTargetStats(
        total_s=perf_counter() - total_t0,
        replay_s=replay_s,
        legal_ids_s=legal_ids_s,
        target_advance_s=target_advance_s,
        tensorize_s=tensorize_s,
        rows=batch_size,
        positions=int(valid_mask.sum()),
        flat_legal_ids=len(flat_token_ids),
    )
    return targets, stats


def legality_loss_from_logits(
    logits: torch.Tensor,
    weights: torch.Tensor,
    legal_targets: LegalTargets,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    flat_logits = logits.view(-1, logits.size(-1))
    valid_weights = weights.reshape(-1) * legal_targets.valid_mask.reshape(-1).to(weights.dtype)
    denom = valid_weights.sum()
    if denom.item() == 0:
        return flat_logits.new_zeros(())

    flat_probs = torch.softmax(flat_logits, dim=-1)
    legal_mass = flat_probs.new_zeros(flat_probs.size(0))
    if legal_targets.flat_positions.numel() > 0:
        legal_mass.index_add_(
            0,
            legal_targets.flat_positions,
            flat_probs[legal_targets.flat_positions, legal_targets.flat_token_ids],
        )

    per_pos = -torch.log(legal_mass.clamp_min(eps))
    return (per_pos * valid_weights).sum() / denom
