import numpy as np
import pytest
import torch

from inference.board_input import BoardInputPlayer, BoardState
from model_loading import load_model
from nanochat.engine import KVCache
from nanochat.gpt import GPT, GPTConfig
from training.board_input import BoardInputModel
from training.common import build_checkpoint


def model_and_inputs():
    vocabulary = ["<bos>", "<eos>", "e4", "d5", "exd5", "Qxd5", "Nc3"]
    tokens = torch.tensor([[0, 2, 3, 4, 5, 6]])
    state = BoardState()
    boards = torch.tensor(np.stack([state.append(vocabulary[t]) for t in tokens[0].tolist()])[None].astype(np.int64))
    config = GPTConfig(sequence_len=16, vocab_size=len(vocabulary), n_layer=2,
                       n_head=2, n_kv_head=2, n_embd=16)
    return BoardInputModel(GPT(config), piece_dim=4), vocabulary, tokens, boards


def test_zero_initialization_preserves_policy_and_board_gradient_exists():
    model, _, tokens, boards = model_and_inputs()
    torch.testing.assert_close(model(tokens, boards), model.policy(tokens), rtol=0, atol=0)
    loss = model(tokens[:, :-1], boards[:, :-1], targets=tokens[:, 1:])
    loss.backward()
    assert model.encoder.projection.weight.grad.abs().sum() > 0
    assert model.policy.transformer.h[0].attn.c_q.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_zero_initialization_preserves_policy_loss(reduction):
    model, _, tokens, boards = model_and_inputs()
    targets = tokens.clone()
    targets[:, 0] = -1
    actual = model(tokens, boards, targets=targets, loss_reduction=reduction)
    expected = model.policy(tokens, targets=targets, loss_reduction=reduction)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_board_input_is_causal_and_control_does_not_see_boards():
    model, _, tokens, boards = model_and_inputs()
    torch.nn.init.normal_(model.encoder.projection.weight, std=0.1)
    other = boards.clone()
    other[:, 3:, :, 0] = 64
    other[:, 3:, :, 1] = 0
    actual, changed = model(tokens, boards), model(tokens, other)
    torch.testing.assert_close(actual[:, :3], changed[:, :3], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(actual[:, 3:], changed[:, 3:])
    model.enabled = False
    torch.testing.assert_close(model(tokens, boards), model(tokens, other), atol=0, rtol=0)


def cache(config):
    return KVCache(1, config.n_kv_head, config.sequence_len,
                   config.n_embd // config.n_head, config.n_layer)


def test_inference_replay_matches_training_and_cached_chunks(tmp_path):
    model, vocabulary, tokens, boards = model_and_inputs()
    torch.nn.init.normal_(model.encoder.projection.weight, std=0.1)
    meta = {"itos": vocabulary, "stoi": {s: i for i, s in enumerate(vocabulary)}}
    path = tmp_path / "model.pt"
    torch.save(build_checkpoint(model.policy, model.policy.config, meta, 1,
                                board_input=model.checkpoint_metadata()), path)
    player, config, _, _ = load_model(str(path), "cpu")
    with torch.inference_mode():
        expected = model(tokens, boards)
        torch.testing.assert_close(player(tokens), expected)
        kv = cache(config)
        chunks = [player(tokens[:, :2], kv_cache=kv), player(tokens[:, 2:3], kv_cache=kv),
                  player(tokens[:, 3:], kv_cache=kv)]
        torch.testing.assert_close(torch.cat(chunks, 1), expected, rtol=1e-5, atol=1e-6)
        second = cache(config)
        torch.testing.assert_close(player(tokens, kv_cache=second), expected)
        kv.reset()
        torch.testing.assert_close(player(tokens, kv_cache=kv), expected)
        padded = torch.cat((tokens, torch.tensor([[1, 1]])), 1)
        torch.testing.assert_close(player(padded)[:, :tokens.size(1)], expected)


def test_inference_rejects_missing_history():
    model, vocabulary, tokens, _ = model_and_inputs()
    player = BoardInputPlayer(model, vocabulary)
    with pytest.raises(ValueError, match="history beginning"):
        player(tokens[:, 1:])


def test_batched_cached_boards_stay_independent_across_eos_and_new_games():
    model, vocabulary, _, _ = model_and_inputs()
    torch.nn.init.normal_(model.encoder.projection.weight, std=0.1)
    player = BoardInputPlayer(model, vocabulary)
    tokens = torch.tensor([[0, 2, 3, 4, 1, 1, 0, 2], [0, 2, 3, 4, 5, 6, 1, 1]])
    config = model.policy.config
    kv = KVCache(2, config.n_kv_head, config.sequence_len,
                 config.n_embd // config.n_head, config.n_layer)
    with torch.inference_mode():
        expected = player(tokens)
        first = player(tokens[:, :3], kv_cache=kv)
        # A separate uncached request must not replace the running games' boards.
        player(torch.tensor([[0, 2]]))
        rest = [player(tokens[:, i:i + 1], kv_cache=kv) for i in range(3, tokens.size(1))]
        torch.testing.assert_close(torch.cat([first, *rest], 1), expected, rtol=1e-5, atol=1e-6)
