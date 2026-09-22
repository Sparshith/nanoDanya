import pytest
import torch

from training.grpo import encode_rewards, exact_loss_terms, reward_vocabulary


def test_exact_reward_gradient_matches_probability_weighted_advantages():
    logits = torch.tensor([[0.9, 0.1]], dtype=torch.float64).log().requires_grad_()
    logp = logits.log_softmax(dim=-1)
    rewards = torch.tensor([[0.5, 0.9754]], dtype=torch.float64)
    policy_loss, kl, advantages = exact_loss_terms(logp, logp.detach(), rewards)
    gradient = torch.autograd.grad(policy_loss + 0.05 * kl, logits, retain_graph=True)[0]
    surrogate = -(logp.exp().detach() * advantages * logp).sum()
    surrogate_gradient = torch.autograd.grad(surrogate, logits)[0]

    assert policy_loss.item() == pytest.approx(-0.54754)
    assert not advantages.requires_grad
    torch.testing.assert_close(advantages, torch.tensor([[-0.04754, 0.42786]], dtype=torch.float64))
    torch.testing.assert_close(gradient, surrogate_gradient)
    torch.testing.assert_close(gradient, torch.tensor([[0.042786, -0.042786]], dtype=torch.float64))


def test_annotation_variants_share_the_move_reward():
    stoi = {"<bos>": 0, "e4": 1, "e4+": 2, "e4#": 3, "Nf3": 4}
    ids, rewards = encode_rewards({"e4": 0.75}, reward_vocabulary(stoi))
    assert ids == [1, 2, 3]
    assert rewards == [0.75, 0.75, 0.75]
