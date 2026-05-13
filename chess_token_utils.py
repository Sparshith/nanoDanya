from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import chess


def strip_san(move: str) -> str:
    return move.rstrip("+#")


def normalized_legal_sans(board: chess.Board) -> set[str]:
    return {strip_san(board.san(mv)) for mv in board.legal_moves}


def token_is_legal_prediction(token: str, legal_sans: set[str]) -> bool:
    if token == "<eos>":
        return True
    if token == "<bos>":
        return False
    return strip_san(token) in legal_sans


def token_is_playable(token: str, legal_sans: set[str], allow_eos: bool = True) -> bool:
    if token == "<bos>":
        return False
    if token == "<eos>":
        return allow_eos
    return strip_san(token) in legal_sans


def resolve_token_id(stoi: dict[str, int], san: str) -> int | None:
    for candidate in (san, strip_san(san)):
        token_id = stoi.get(candidate)
        if token_id is not None:
            return token_id

    normalized = strip_san(san)
    for token, idx in stoi.items():
        if token.startswith("<"):
            continue
        if strip_san(token) == normalized:
            return idx

    return None
