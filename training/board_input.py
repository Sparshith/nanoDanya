"""Current board information added to SAN embeddings before the transformer."""
import torch
from torch import nn
from torch.nn import functional as F

from training.common import SOFTCAP, forward_hidden


class BoardEncoder(nn.Module):
    def __init__(self, n_embd, piece_dim=32):
        super().__init__()
        self.locations = nn.Embedding(32 * 65, piece_dim)
        self.kinds = nn.Embedding(7, piece_dim)
        self.projection = nn.Linear(32 * piece_dim, n_embd, bias=False)
        nn.init.zeros_(self.projection.weight)
        self.register_buffer("offsets", torch.arange(32) * 65, persistent=False)

    def forward(self, boards):
        pieces = self.locations(boards[..., 0] + self.offsets) + self.kinds(boards[..., 1])
        return self.projection(pieces.flatten(-2))


class BoardInputModel(nn.Module):
    def __init__(self, policy, enabled=True, piece_dim=32):
        super().__init__()
        self.policy = policy
        self.encoder = BoardEncoder(policy.config.n_embd, piece_dim)
        self.enabled = enabled
        self.piece_dim = piece_dim

    def forward(self, tokens, boards, targets=None, kv_cache=None, loss_reduction="mean", blank=False):
        if not self.enabled or blank:
            boards = torch.stack((torch.full_like(boards[..., 0], 64),
                                  torch.zeros_like(boards[..., 1])), -1)
        additions = self.encoder(boards)
        hidden = forward_hidden(self.policy, tokens, additions, kv_cache)
        logits = self.policy.lm_head(hidden)
        logits = SOFTCAP * torch.tanh(logits / SOFTCAP)
        if targets is not None:
            return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1),
                                   ignore_index=-1, reduction=loss_reduction)
        return logits

    def checkpoint_metadata(self):
        return {"version": 1, "enabled": self.enabled, "piece_dim": self.piece_dim,
                "encoder": self.encoder.state_dict()}
