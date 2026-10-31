import copy

import torch

from nanochat.gpt import GPT, GPTConfig
from training import distributed
from training.pretrain import backward_batch, make_loss, training_loss


def main():
    distributed.initialize('cpu')
    try:
        torch.manual_seed(4)
        model = GPT(GPTConfig(sequence_len=4, vocab_size=128, n_layer=1,
                              n_head=1, n_kv_head=1, n_embd=16))
        reference = copy.deepcopy(model)
        settings = dict(device='cpu', compile=False, datasets={'games': ''},
                        loss_weights=None, microbatch_size=1)
        loss_fn = make_loss(model, settings)
        x = torch.tensor([[0, 2, 3], [0, 4, 1], [0, 5, 6], [1, 1, 1]])
        for y in [torch.tensor([[2, 3, 1], [4, 1, -1], [5, -1, -1], [-1, -1, -1]]),
                  torch.tensor([[2, 3, 1], [4, 1, -1], [-1, -1, -1], [-1, -1, -1]])]:
            model.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
            loss, _ = backward_batch(model, (x, y, torch.zeros(4, dtype=torch.long)), settings, loss_fn)
            expected = training_loss(reference, x, y, torch.full((4,), 1 / int((y != -1).sum())))
            expected.backward()
            torch.testing.assert_close(torch.tensor(loss), expected.detach(), atol=1e-6, rtol=1e-6)
            for actual, target in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(actual.grad, target.grad, atol=1e-7, rtol=1e-5)
    finally:
        torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
