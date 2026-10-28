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


def test_piece_metrics_penalize_missing_wrong_and_invented_pieces():
    from probes.compare_involved import piece_metrics

    truth = torch.tensor([[1, 7, 6, 0, 0]])
    prediction = torch.tensor([[1, 0, 12, 7, 0]])
    metrics = piece_metrics(prediction, truth)
    assert metrics["recall"] == metrics["precision"] == 1 / 3
    assert metrics["missing"] == metrics["wrong_piece"] == metrics["invented"] == 1
    assert metrics["exact_positions"] == 0
    padded_truth = torch.cat([truth, torch.zeros((1, 59), dtype=torch.long)], dim=1)
    padded_prediction = torch.cat([prediction, torch.zeros((1, 59), dtype=torch.long)], dim=1)
    assert piece_metrics(padded_prediction, padded_truth) == metrics


def test_identity_trace_scores_pre_move_living_pieces_without_future_input(tmp_path, monkeypatch):
    from probes.involved_trace import inspect_head
    from training.common import build_checkpoint, forward_hidden
    from training.involved_pieces import InvolvedPieceModel

    vocabulary = ["<bos>", "<eos>", "e4", "d5", "exd5", "Qxd5"]
    stoi = {san: i for i, san in enumerate(vocabulary)}
    config = GPTConfig(sequence_len=8, vocab_size=6, n_layer=1, n_head=2, n_kv_head=2, n_embd=16)
    model = InvolvedPieceModel(GPT(config))
    saved = build_checkpoint(model.policy, config, {"stoi": stoi}, 0)
    saved["piece_head"] = model.piece_head.state_dict()
    checkpoint = tmp_path / "model.pt"
    torch.save(saved, checkpoint)
    inputs = []

    def capture(policy, tokens):
        inputs.append(tokens.clone())
        return forward_hidden(policy, tokens)

    monkeypatch.setattr("probes.involved_trace.forward_hidden", capture)
    report = inspect_head(checkpoint, "1. e4 d5 2. exd5 Qxd5 *", "cpu")
    assert inputs[0].tolist() == [[0, 2, 1, 1], [0, 2, 3, 4]]
    assert report["known_living_count"] == 63
    assert report["involved_count"] == 3
    last = report["timeline"][-1]
    assert last["ply"] == 3 and last["next_move"] == "Qxd5"
    assert {(p["identity"], p["actual"]) for p in last["pieces"] if p["involved"]} == {
        ("d8", "d8"), ("e2", "d5")}
    assert "d7" not in {p["identity"] for p in last["pieces"] if p["alive"]}
    assert len(last["pieces"]) == 32
    captured = next(p for p in last["pieces"] if p["identity"] == "d7")
    assert captured["actual"] is None and captured["actual_probability"] is None

    complete = inspect_head(checkpoint, "1. e4 d5 2. exd5 Qxd5 *", "cpu", all_positions=True)
    assert [r["ply"] for r in complete["timeline"]] == list(range(5))
    assert complete["timeline"][-1]["next_move"] is None
    assert complete["timeline"][-1]["last_move"] == "Qxd5"
    assert complete["timeline"][-1]["input_tokens"] == 5
    assert sum(p["alive"] for p in complete["timeline"][-1]["pieces"]) == 30
    assert complete["involved_count"] == 6
