from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from chess_token_utils import normalized_legal_sans, token_is_playable

if TYPE_CHECKING:
    import chess
    import torch


def token_for_id(itos: Mapping[int | str, str] | Sequence[str], idx: int) -> str:
    if isinstance(itos, Sequence) and not isinstance(itos, str):
        return itos[idx]
    if idx in itos:
        return itos[idx]
    return itos[str(idx)]


def legal_token_ids(
    stoi: Mapping[str, int],
    board: chess.Board,
    *,
    allow_eos: bool,
) -> list[int]:
    legal_san = normalized_legal_sans(board)
    return [
        idx
        for token, idx in stoi.items()
        if token_is_playable(token, legal_san, allow_eos=allow_eos)
    ]


def choose_token_from_logits(
    logits: torch.Tensor,
    token_ids: Sequence[int],
    itos: Mapping[int | str, str] | Sequence[str],
    *,
    temperature: float,
) -> tuple[int | None, str | None]:
    import torch

    if not token_ids:
        return None, None

    mask = torch.full_like(logits, float("-inf"))
    mask[list(token_ids)] = logits[list(token_ids)]
    if temperature <= 0:
        next_id = int(torch.argmax(mask).item())
    else:
        probs = torch.softmax(mask / temperature, dim=-1)
        next_id = int(torch.multinomial(probs, num_samples=1).item())
    return next_id, token_for_id(itos, next_id)


def choose_move_from_logits(
    logits: torch.Tensor,
    board: chess.Board,
    stoi: Mapping[str, int],
    itos: Mapping[int | str, str] | Sequence[str],
    *,
    temperature: float,
    allow_eos: bool,
    legal_mask: bool,
) -> tuple[int | None, str | None]:
    token_ids = (
        legal_token_ids(stoi, board, allow_eos=allow_eos)
        if legal_mask
        else list(range(len(itos)))
    )
    return choose_token_from_logits(
        logits,
        token_ids,
        itos,
        temperature=temperature,
    )
