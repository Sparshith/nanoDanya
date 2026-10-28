import json
import pickle

import chess
import numpy as np
import pytest
import torch

from data_prep.prepare_board_inputs import expand, prepare, verify_replay
from data_prep.prepare_involved_pieces import INITIAL_SQUARES, prepare as prepare_labels
from inference.board_input import BoardInputPlayer, BoardState
from model_loading import load_model
from nanochat.engine import KVCache
from nanochat.gpt import GPT, GPTConfig
from training.board_input import BoardInputModel
from training.common import build_checkpoint


def data(tmp_path, moves):
    vocabulary = ["<bos>", "<eos>", *dict.fromkeys(moves.split())]
    meta = {"itos": vocabulary, "stoi": {s: i for i, s in enumerate(vocabulary)},
            "context_length": 16, "vocab_size": len(vocabulary)}
    tokens = np.array([0, *[meta["stoi"][s] for s in moves.split()], 1] * 5, dtype=np.uint16)
    folder = tmp_path / "data"
    folder.mkdir()
    (folder / "meta.pkl").write_bytes(pickle.dumps(meta))
    for split in ("train", "val"):
        tokens.tofile(folder / f"{split}.bin")
    prepare_labels(folder, tmp_path / "labels")
    return folder, meta, tokens


@pytest.mark.parametrize("moves", [
    "e4 d5 exd5 Qxd5 Nc3 Qe5+",
    "e4 a6 e5 d5 exd6",
    "e4 e5 Nf3 Nc6 Bc4 Nf6 O-O Bc5 d3 O-O",
    "a4 h5 a5 h4 a6 h3 axb7 hxg2 bxa8=Q gxf1=Q+",
])
def test_feature_expansion_matches_board_after_each_input(tmp_path, moves):
    folder, meta, tokens = data(tmp_path, moves)
    manifest = prepare(folder, tmp_path / "labels", tmp_path / "boards")
    features = np.load(tmp_path / "boards/train.npy")
    assert manifest["splits"]["train"]["replay_check"]["games"] == 5
    assert verify_replay(tokens, features, meta)["tokens"] == len(tokens)
    assert features[0, :, 0].tolist() == list(INITIAL_SQUARES)
    board = chess.Board()
    for i, san in enumerate(moves.split(), 1):
        board.push_san(san)
        reconstructed = {int(sq): int(kind) for sq, kind in features[i] if sq != 64}
        assert reconstructed == {sq: p.piece_type for sq, p in board.piece_map().items()}
    if "bxa8=Q" in moves:
        a_pawn = INITIAL_SQUARES.index(chess.A2)
        assert features[9, a_pawn].tolist() == [chess.A8, chess.QUEEN]


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


def test_changing_future_tokens_does_not_change_earlier_board_features(tmp_path):
    folder, meta, tokens = data(tmp_path, "e4 d5 exd5 Qxd5 Nc3 Qe5+")
    involved = np.load(tmp_path / "labels/train.npy")
    full = np.empty((len(tokens), 32, 2), dtype=np.uint8)
    prefix = np.empty((4, 32, 2), dtype=np.uint8)
    expand(tokens, involved, meta, full)
    expand(tokens[:4], involved[:4], meta, prefix)
    np.testing.assert_array_equal(full[:4], prefix)
    assert full[3, INITIAL_SQUARES.index(chess.E2), 0] == chess.D5
    assert full[3, INITIAL_SQUARES.index(chess.D7)].tolist() == [64, 0]


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


@pytest.mark.parametrize("enabled", ["0", "1"])
def test_training_checkpoint_and_inference_roundtrip(tmp_path, monkeypatch, enabled):
    from training import train_board_input

    folder, meta, _ = data(tmp_path, "e4 d5 exd5 Qxd5 Nc3 Qe5+")
    prepare(folder, tmp_path / "labels", tmp_path / "boards")
    config = GPTConfig(sequence_len=16, vocab_size=meta["vocab_size"], n_layer=1,
                       n_head=1, n_kv_head=1, n_embd=8)
    initial = tmp_path / "initial.pt"
    torch.save(build_checkpoint(GPT(config), config, meta, 0), initial)
    for key, value in {"DATASET_DIR": folder, "BOARD_INPUT_DIR": tmp_path / "boards",
                       "CKPT_DIR": tmp_path / "trained", "INIT_CHECKPOINT": initial,
                       "BOARD_INPUT_ENABLED": enabled, "MAX_ITERS": 2, "BATCH_SIZE": 1,
                       "VAL_BATCHES": 2, "EVAL_INTERVAL": 1}.items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.setattr(train_board_input, "get_device", lambda: torch.device("cpu"))
    train_board_input.main()
    metrics = [json.loads(s) for s in (tmp_path / "trained/metrics.jsonl").read_text().splitlines()]
    assert metrics[-1]["step"] == 2
    if enabled == "0":
        assert metrics[-1]["next_move_loss"] == metrics[-1]["blank_board_loss"]
    model, _, _, _ = load_model(str(tmp_path / "trained/final.pt"), "cpu")
    assert torch.isfinite(model(torch.tensor([[0, meta["stoi"]["e4"]]]))).all()
    train_board_input.main()
    assert len((tmp_path / "trained/metrics.jsonl").read_text().splitlines()) == 2
