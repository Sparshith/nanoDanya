from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import chess

PIECE_MOVE_RE = re.compile(r"^([KQRBN])([a-h1-8]{0,2})(x?)([a-h][1-8])$")


def strip_san(move: str) -> str:
    return move.rstrip("+#")


def strip_san_disambiguation(move: str) -> str:
    normalized = strip_san(move)
    match = PIECE_MOVE_RE.match(normalized)
    if match is None:
        return normalized

    piece, _, capture, destination = match.groups()
    return f"{piece}{capture}{destination}"


def normalized_legal_sans(board: chess.Board) -> set[str]:
    return {strip_san(board.san(mv)) for mv in board.legal_moves}


def under_disambiguated_legal_matches(token: str, legal_sans: set[str]) -> tuple[str, ...]:
    if token in {"<bos>", "<eos>"}:
        return ()

    normalized = strip_san(token)
    if normalized in legal_sans:
        return ()

    matches = [
        san
        for san in legal_sans
        if san != normalized and strip_san_disambiguation(san) == normalized
    ]
    return tuple(sorted(matches))


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
