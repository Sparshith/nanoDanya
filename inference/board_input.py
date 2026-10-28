"""Replay observed SAN tokens to supply a board to board-conditioned policies."""
from weakref import WeakKeyDictionary

import chess
import numpy as np
import torch
from torch import nn

from chess_inference import token_for_id
from data_prep.prepare_involved_pieces import INITIAL_SQUARES, label_move


class BoardState:
    def __init__(self):
        self.board = None
        self.identities = {}

    def append(self, token):
        if token == "<bos>":
            self.board = chess.Board()
            self.identities = dict(zip(INITIAL_SQUARES, range(32)))
        elif token == "<eos>":
            self.board = None
            self.identities = {}
        else:
            if self.board is None:
                raise ValueError("Board input requires history beginning with <bos>")
            move = self.board.parse_san(token)
            if not move:
                raise ValueError("Null moves are not supported")
            label_move(self.board, self.identities, move)
        features = np.zeros((32, 2), dtype=np.uint8)
        features[:, 0] = 64
        for square, identity in self.identities.items():
            features[identity] = square, self.board.piece_type_at(square)
        return features


class BoardInputPlayer(nn.Module):
    def __init__(self, model, itos):
        super().__init__()
        self.model = model
        self.config = model.policy.config
        self.itos = itos
        self.blank_board = False
        self._board_caches = WeakKeyDictionary()

    def forward(self, tokens, targets=None, kv_cache=None, loss_reduction="mean"):
        rows = tokens.detach().cpu().tolist()
        position = 0 if kv_cache is None else kv_cache.get_pos()
        if position == 0:
            states = [BoardState() for _ in rows]
        else:
            expected_position, states = self._board_caches[kv_cache]
            if expected_position != position or len(states) != len(rows):
                raise ValueError("Board history and KV cache are out of sync")
        boards = np.stack([[state.append(token_for_id(self.itos, token)) for token in row]
                           for state, row in zip(states, rows)])
        boards = torch.from_numpy(boards.astype(np.int64)).to(tokens.device)
        logits = self.model(tokens, boards, targets=targets, kv_cache=kv_cache,
                            loss_reduction=loss_reduction, blank=self.blank_board)
        if kv_cache is not None:
            self._board_caches[kv_cache] = (position + tokens.size(1), states)
        return logits
