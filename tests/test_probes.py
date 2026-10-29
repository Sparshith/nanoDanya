from types import SimpleNamespace

import chess
import pytest
import torch

from nanochat.gpt import GPT, GPTConfig
from probes.cache_hiddens import cache_hiddens
from probes.inspect_game import game_position
from probes.run import split_indices


def test_game_position_excludes_next_move_and_resolves_check_suffix():
    stoi = {"<bos>": 0, "<eos>": 1, "f3": 2, "e5": 3, "g4": 4, "Qh4": 5}
    pgn = '1. f3 e5 2. g4 Qh4# 0-1'
    position = game_position(pgn, stoi, 3)
    assert position["prefixes"] == [[0, 2, 3, 4]]
    assert position["next_move"] == "Qh4#"
    assert not chess.Board(position["fen"][0]).is_game_over()
    final = game_position(pgn, stoi, 4)
    assert final["prefixes"] == [[0, 2, 3, 4, 5]]
    assert chess.Board(final["fen"][0]).is_checkmate()
    with pytest.raises(ValueError, match="absent from tokenizer"):
        game_position(pgn, {"<bos>": 0}, 1)


def test_hidden_cache_uses_last_real_token_and_serving_context(tmp_path, monkeypatch):
    config = GPTConfig(sequence_len=4, vocab_size=8, n_layer=2, n_head=2, n_kv_head=2, n_embd=16)
    model = GPT(config).eval()
    stoi = {"<bos>": 0, "<eos>": 1}
    monkeypatch.setattr("probes.cache_hiddens.device_name", lambda: "cpu")
    monkeypatch.setattr("probes.cache_hiddens.load_model", lambda *_: (model, config, stoi, {}))
    path = tmp_path / "positions.pt"
    prefixes = [[0, 2], [0, 2, 3, 4, 5, 6]]
    torch.save({"prefixes": prefixes, "vocab_size": 8, "stoi": stoi}, path)
    result = cache_hiddens(SimpleNamespace(model="test", positions=path, layers="all", batch_size=2))
    for layer in range(2):
        captured = []
        hook = model.transformer.h[layer].register_forward_hook(
            lambda _m, _i, out: captured.append(out[:, -1].detach())
        )
        with torch.inference_mode():
            for prefix in prefixes:
                model(torch.tensor([prefix[-config.sequence_len:]]))
        hook.remove()
        torch.testing.assert_close(result["hiddens"][layer], torch.cat(captured), atol=1e-5, rtol=1e-5)


def test_probe_split_holds_out_whole_games():
    labels = {"prefixes": [[0]] * 12, "game_id": torch.tensor([0, 1, 2, 3] * 3)}
    train, val = split_indices(labels, 0.8, 0)
    assert set(labels["game_id"][train].tolist()).isdisjoint(labels["game_id"][val].tolist())
    assert len(train) + len(val) == 12
    with pytest.raises(ValueError, match="At least two games"):
        split_indices({"prefixes": [[0]] * 4, "game_id": torch.zeros(4)}, 0.8, 0)
