import torch
import torch.nn.functional as F

from chess_token_utils import strip_san
from training.common import forward_hidden, move_logits


def reward_vocabulary(stoi):
    strip_ids = {}
    for token, token_id in stoi.items():
        if not token.startswith("<"):
            strip_ids.setdefault(strip_san(token), []).append(token_id)
    return strip_ids


def encode_rewards(moves, strip_ids):
    ids, rewards = [], []
    for san, reward in moves.items():
        for token_id in strip_ids.get(strip_san(san), ()):
            ids.append(token_id)
            rewards.append(reward)
    return ids, rewards


def last_logprobs(model, tokens, last, temperature):
    hidden = forward_hidden(model, tokens)
    last_hidden = hidden[torch.arange(tokens.size(0), device=tokens.device), last]
    return F.log_softmax(move_logits(model, last_hidden) / temperature, dim=-1)


def exact_loss_terms(logp, ref_logp, rewards, legal_mask=None):
    probabilities = logp.exp()
    objective_probabilities = probabilities if legal_mask is None else logp.masked_fill(~legal_mask, -torch.inf).softmax(dim=-1)
    expected_reward = (objective_probabilities * rewards).sum(dim=-1, keepdim=True)
    advantages = (rewards - expected_reward).detach()
    if legal_mask is not None:
        advantages = advantages.masked_fill(~legal_mask, 0)
    policy_loss = -expected_reward.mean()
    kl = (probabilities * (logp - ref_logp)).sum(dim=-1).mean()
    return policy_loss, kl, advantages
