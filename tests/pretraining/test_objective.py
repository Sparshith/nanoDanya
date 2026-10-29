import copy
import math

import pytest
import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig
from training.pretrain import backward_batch, make_loss, token_losses


@pytest.mark.parametrize('weighted', [False, True])
def test_microbatch_gradients_match_full_batch_objective(recipe, weighted):
    torch.manual_seed(4)
    model = GPT(GPTConfig(**recipe['model_config']))
    reference = copy.deepcopy(model)
    x = torch.tensor([[0, 2, 3], [0, 4, 1], [0, 5, 6], [0, 7, 1], [1, 1, 1]])
    y = torch.tensor([[2, 3, 1], [4, 1, -1], [5, 6, 1], [7, -1, -1], [-1, -1, -1]])
    ids = torch.tensor([0, 0, 1, 1, -1])
    recipe['loss_weights'] = {'archive': .3, 'engine': .7} if weighted else None
    loss, counts = backward_batch(model, (x, y, ids), recipe, loss_fn=make_loss(model, recipe))
    logits = reference(x)
    if weighted:
        expected = .3 * F.cross_entropy(logits[:2].reshape(-1, 128), y[:2].reshape(-1), ignore_index=-1)
        expected += .7 * F.cross_entropy(logits[2:4].reshape(-1, 128), y[2:4].reshape(-1), ignore_index=-1)
    else:
        expected = F.cross_entropy(logits.reshape(-1, 128), y.reshape(-1), ignore_index=-1)
    expected.backward()
    assert loss == pytest.approx(expected.item())
    assert counts == {'archive': 5, 'engine': 4}
    for parameter, full_batch in zip(model.parameters(), reference.parameters()):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, full_batch.grad, atol=1e-7, rtol=1e-5)
    assert sum(p.grad.abs().sum() for p in model.parameters()) > 0


def test_bfloat16_model_retains_float32_loss_precision(recipe):
    model = GPT(GPTConfig(**recipe['model_config']))
    model.init_weights()
    model.to(dtype=torch.bfloat16)
    x = torch.tensor([[0, 2, 3]])
    y = torch.tensor([[2, 3, -1]])
    with torch.no_grad():
        loss = token_losses(model, x, y)
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, torch.tensor([[math.log(128), math.log(128), 0.]]), atol=1e-6, rtol=0)
