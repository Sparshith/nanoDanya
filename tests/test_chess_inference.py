from pathlib import Path
import sys

import chess
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chess_inference import choose_move_from_logits, legal_token_ids, token_for_id


def test_token_for_id_handles_list_and_string_key_dict():
    assert token_for_id(["<bos>", "e4"], 1) == "e4"
    assert token_for_id({"0": "<bos>", "1": "e4"}, 1) == "e4"


def test_choose_move_from_logits_masks_to_legal_moves_and_argmaxes_at_zero_temperature():
    stoi = {"<bos>": 0, "<eos>": 1, "e4": 2, "e5": 3, "Nf3": 4}
    itos = {idx: token for token, idx in stoi.items()}
    board = chess.Board()
    logits = torch.tensor([0.0, 100.0, 2.0, 20.0, 1.0])

    next_id, token = choose_move_from_logits(
        logits,
        board,
        stoi,
        itos,
        temperature=0.0,
        allow_eos=False,
        legal_mask=True,
    )

    assert (next_id, token) == (2, "e4")


def test_legal_token_ids_can_include_eos_when_allowed():
    stoi = {"<bos>": 0, "<eos>": 1, "e4": 2}

    assert 1 not in legal_token_ids(stoi, chess.Board(), allow_eos=False)
    assert 1 in legal_token_ids(stoi, chess.Board(), allow_eos=True)
