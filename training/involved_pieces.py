import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from training.common import forward_hidden, move_logits


def make_batch(tokens, labels, starts, block_size, bos_id):
    inputs = torch.from_numpy(np.stack([tokens[i:i + block_size] for i in starts]).astype(np.int64))
    targets = torch.from_numpy(np.stack([tokens[i + 1:i + block_size + 1] for i in starts]).astype(np.int64))
    involved = torch.from_numpy(np.stack([labels[i + 1:i + block_size + 1] for i in starts]).astype(np.int64))
    # A cropped game prefix does not determine the current piece locations.
    visible_start = (inputs == bos_id).cumsum(dim=1) > 0
    involved[~visible_start] = -1
    return inputs, targets, involved


class InvolvedPieceModel(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.piece_head = nn.Linear(policy.config.n_embd, 32 * 64)

    def forward(self, tokens):
        hidden = forward_hidden(self.policy, tokens)
        return move_logits(self.policy, hidden), self.piece_head(hidden).unflatten(-1, (32, 64))


def loss_terms(logits, locations, targets, involved):
    move_loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
    identities, squares, _ = involved.unbind(-1)
    selected = locations.gather(-2, identities.clamp_min(0).unsqueeze(-1).expand(*identities.shape, 64))
    valid = squares >= 0
    losses = F.cross_entropy(selected.flatten(0, 2), squares.clamp_min(0).flatten(), reduction="none")
    piece_loss = (losses.view_as(squares) * valid).sum() / valid.sum().clamp_min(1)
    correct = (selected.argmax(-1) == squares) & valid
    return move_loss, piece_loss, correct


def occupancy_batch(masks, starts, block_size, inputs, bos_id):
    packed = torch.from_numpy(np.stack([masks[i + 1:i + block_size + 1] for i in starts]).astype(np.int64))
    alive = (packed.unsqueeze(-1) >> torch.arange(32)) & 1
    return alive.bool() & ((inputs == bos_id).cumsum(dim=1) > 0).unsqueeze(-1)


def collision_loss(locations, alive):
    probabilities = locations.float().softmax(-1) * alive.unsqueeze(-1)
    mass = probabilities.sum(-2)
    pairs = ((mass.square() - probabilities.square().sum(-2)).sum(-1) / 2).clamp_min(0)
    count = alive.sum(-1)
    valid = count > 1
    return ((pairs / count.clamp_min(1)) * valid).sum() / valid.sum().clamp_min(1)


def living_piece_loss(locations, truth, alive):
    losses = F.cross_entropy(locations.flatten(0, 2), truth.clamp_min(0).flatten(), reduction="none")
    return (losses.view_as(truth) * alive).sum() / alive.sum().clamp_min(1)


def occupancy_counts(locations, alive, truth):
    predictions = locations.argmax(-1)
    counts = torch.zeros_like(locations[..., 0, :], dtype=torch.long)
    counts.scatter_add_(-1, predictions, alive.long())
    valid = alive.any(-1)
    correct = ((predictions == truth) & alive).sum()
    return {"living_correct": int(correct), "living_count": int(alive.sum()),
            "collision_positions": int(((counts > 1).any(-1) & valid).sum()),
            "collision_squares": int((counts > 1).sum()),
            "scored_positions": int(valid.sum()),
            "exact_boards": int((((predictions == truth) | ~alive).all(-1) & valid).sum())}
