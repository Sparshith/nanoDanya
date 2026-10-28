import json
import pickle

import chess
import numpy as np
import pytest
import torch

from data_prep.prepare_involved_pieces import (
    CAPTURED, CASTLING_ROOK, INITIAL_SQUARES, MOVER, label_move, prepare, prepare_split,
)
from nanochat.gpt import GPT, GPTConfig
from training.involved_pieces import InvolvedPieceModel, loss_terms, make_batch


def test_collision_penalty_excludes_captured_pieces_and_has_finite_gradients():
    from training.involved_pieces import collision_loss, occupancy_counts

    logits = torch.full((1, 1, 32, 64), -30.0)
    logits[..., 5] = 30
    alive = torch.zeros((1, 1, 32), dtype=torch.bool)
    alive[..., :2] = True
    assert collision_loss(logits, alive).item() == pytest.approx(0.5)
    logits[0, 0, 1, 5], logits[0, 0, 1, 6] = -30, 30
    assert collision_loss(logits, alive).item() == pytest.approx(0, abs=1e-6)
    counts = occupancy_counts(logits, alive, torch.tensor([[[5, 6] + [-1] * 30]]))
    assert counts["collision_positions"] == 0 and counts["exact_boards"] == 1
    random = torch.randn((1, 1, 32, 64), requires_grad=True)
    collision_loss(random, alive).backward()
    assert torch.isfinite(random.grad).all()
    assert random.grad[..., :2, :].abs().sum() > 0
    assert random.grad[..., 2:, :].abs().sum() == 0
    assert collision_loss(random, torch.zeros_like(alive)).item() == 0


def test_occupancy_targets_precede_capture_and_mask_incomplete_prefixes(tmp_path):
    from data_prep.prepare_piece_occupancy import prepare as prepare_occupancy
    from training.involved_pieces import occupancy_batch

    data, labels, _, tokens = dataset(tmp_path)
    output = tmp_path / "occupancy"
    prepare_occupancy(data, labels, output)
    packed = np.load(output / "train_alive.npy")
    truth = np.load(output / "val_locations.npy")
    assert packed[3] == (1 << 32) - 1
    assert not (int(packed[4]) >> identity(chess.D7)) & 1
    assert not (int(packed[5]) >> identity(chess.E2)) & 1
    assert truth[3, identity(chess.E2)] == chess.E4
    assert truth[3, identity(chess.D7)] == chess.D5
    assert truth[4, identity(chess.E2)] == chess.D5
    assert truth[4, identity(chess.D7)] == -1
    x, _, _ = make_batch(tokens, np.load(labels / "train.npy"), [0, 2], 12, 0)
    alive = occupancy_batch(packed, [0, 2], 12, x, 0)
    assert alive[0, 2].sum() == 32 and alive[0, 3].sum() == 31
    assert not alive[1, :6].any() and alive[1, 6].sum() == 32


def replay(moves):
    board = chess.Board()
    identities = dict(zip(INITIAL_SQUARES, range(32)))
    labels = []
    for san in moves.split():
        labels.append(label_move(board, identities, board.parse_san(san)))
    return board, identities, labels


def identity(square):
    return INITIAL_SQUARES.index(square)


def test_capture_targets_current_locations_and_preserves_piece_identity():
    board, identities, labels = replay("e4 d5 exd5 Qxd5 Nc3 Qe5+")
    assert labels[2] == [(identity(chess.E2), chess.E4, MOVER),
                         (identity(chess.D7), chess.D5, CAPTURED)]
    assert labels[3] == [(identity(chess.D8), chess.D8, MOVER),
                         (identity(chess.E2), chess.D5, CAPTURED)]
    assert labels[5] == [(identity(chess.D8), chess.D5, MOVER)]
    assert identities[chess.E5] == identity(chess.D8)
    assert set(identities) == set(board.piece_map())


def test_en_passant_labels_captured_pawn_square_not_destination():
    board, identities, labels = replay("e4 a6 e5 d5 exd6")
    assert labels[-1] == [(identity(chess.E2), chess.E5, MOVER),
                          (identity(chess.D7), chess.D5, CAPTURED)]
    assert chess.D5 not in identities
    assert identities[chess.D6] == identity(chess.E2)
    assert set(identities) == set(board.piece_map())
    _, _, labels = replay("a3 e5 a4 e4 d4 exd3")
    assert labels[-1][1] == (identity(chess.D2), chess.D4, CAPTURED)


@pytest.mark.parametrize("fen, san, king, rook, destination", [
    ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "O-O", chess.E1, chess.H1, chess.F1),
    ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "O-O-O", chess.E8, chess.A8, chess.D8),
])
def test_castling_supervises_both_pre_move_locations(fen, san, king, rook, destination):
    board = chess.Board(fen)
    identities = {sq: identity(sq) for sq in board.piece_map()}
    labels = label_move(board, identities, board.parse_san(san))
    assert labels == [(identity(king), king, MOVER), (identity(rook), rook, CASTLING_ROOK)]
    assert identities[destination] == identity(rook)
    assert set(identities) == set(board.piece_map())


def test_promotion_keeps_pawn_identity_on_later_moves():
    board = chess.Board("7k/P7/8/8/8/8/8/7K w - - 0 1")
    identities = {chess.A7: identity(chess.A2), chess.H1: identity(chess.E1), chess.H8: identity(chess.E8)}
    first = label_move(board, identities, board.parse_san("a8=Q+"))
    assert first == [(identity(chess.A2), chess.A7, MOVER)]
    label_move(board, identities, board.parse_san("Kg7"))
    assert label_move(board, identities, board.parse_san("Qa7+")) == [(identity(chess.A2), chess.A8, MOVER)]


def dataset(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    vocabulary = ["<bos>", "<eos>", "e4", "d5", "exd5", "Qxd5", "Nc3", "Qe5+"]
    meta = {"itos": vocabulary, "stoi": {san: i for i, san in enumerate(vocabulary)},
            "vocab_size": len(vocabulary), "context_length": 12}
    (data / "meta.pkl").write_bytes(pickle.dumps(meta))
    tokens = np.array([0, 2, 3, 4, 5, 6, 7, 1] * 4, dtype=np.uint16)
    for split in ("train", "val"):
        tokens.tofile(data / f"{split}.bin")
    labels = tmp_path / "labels"
    prepare(data, labels)
    return data, labels, meta, tokens


def test_labels_align_with_next_token_and_crop_masks_until_bos(tmp_path):
    _, folder, meta, tokens = dataset(tmp_path)
    labels = np.load(folder / "train.npy")
    x, y, involved = make_batch(tokens, labels, [0, 2], 12, meta["stoi"]["<bos>"])
    assert x[0, 0] == 0 and y[0, 0] == meta["stoi"]["e4"]
    assert involved[0, 0, 0].tolist() == [identity(chess.E2), chess.E2, MOVER]
    assert involved[0, 2].tolist() == [[identity(chess.E2), chess.E4, MOVER],
                                     [identity(chess.D7), chess.D5, CAPTURED]]
    assert torch.all(involved[1, :6] == -1)
    assert involved[1, 6, 0].tolist() == [identity(chess.E2), chess.E2, MOVER]
    assert np.all(labels[tokens == 0] == -1)
    assert np.all(labels[tokens == 1] == -1)


def test_invalid_game_fails_instead_of_supervising_wrong_board(tmp_path):
    path = tmp_path / "bad.bin"
    np.array([0, 2, 2, 1], dtype=np.uint16).tofile(path)
    with pytest.raises(ValueError, match="Invalid SAN"):
        prepare_split(path, tmp_path / "labels.npy", {"stoi": {"<bos>": 0, "<eos>": 1},
                                                      "itos": ["<bos>", "<eos>", "e4"]})


def tiny_model():
    config = GPTConfig(sequence_len=12, vocab_size=8, n_layer=1, n_head=1, n_kv_head=1, n_embd=8)
    return InvolvedPieceModel(GPT(config))


def test_auxiliary_loss_reaches_backbone_and_only_selected_piece_heads(tmp_path):
    _, folder, _, tokens = dataset(tmp_path)
    x, y, involved = make_batch(tokens, np.load(folder / "train.npy"), [0], 3, 0)
    model = tiny_model()
    moves, locations = model(x)
    assert torch.allclose(moves, model.policy(x), atol=1e-6)
    _, piece_loss, _ = loss_terms(moves, locations, y, involved)
    piece_loss.backward()
    assert model.policy.transformer.wte.weight.grad.abs().sum() > 0
    assert model.policy.transformer.h[0].attn.c_q.weight.grad.abs().sum() > 0
    gradients = model.piece_head.weight.grad.view(32, 64, -1).abs().sum((1, 2))
    selected = set(involved[..., 0][involved[..., 0] >= 0].tolist())
    assert set(torch.where(gradients > 0)[0].tolist()) == selected


def test_future_tokens_cannot_change_location_predictions():
    model = tiny_model().eval()
    a = torch.tensor([[0, 2, 3, 4]])
    b = torch.tensor([[0, 2, 6, 7]])
    with torch.no_grad():
        _, first = model(a)
        _, second = model(b)
    assert torch.allclose(first[:, :2], second[:, :2], atol=1e-6)


def test_no_supervised_pieces_has_finite_zero_loss():
    model = tiny_model()
    x = torch.tensor([[0, 2]])
    moves, locations = model(x)
    _, loss, correct = loss_terms(moves, locations, x, torch.full((1, 2, 2, 3), -1))
    assert loss.item() == 0 and not correct.any()
    loss.backward()
    assert model.piece_head.weight.grad.abs().sum() == 0


def test_training_checkpoint_is_playable_and_resume_matches_uninterrupted(tmp_path, monkeypatch):
    from training import train_involved_pieces as trainer
    from model_loading import load_model

    torch.set_num_threads(1)
    data, labels, _, _ = dataset(tmp_path)
    monkeypatch.setattr(trainer, "get_device", lambda: torch.device("cpu"))
    settings = {"DATASET_DIR": str(data), "PIECE_LABELS_DIR": str(labels), "MAX_ITERS": "2",
                "BATCH_SIZE": "2", "VAL_BATCHES": "2", "EVAL_INTERVAL": "1",
                "N_LAYER": "1", "N_HEAD": "1", "N_KV_HEAD": "1", "N_EMBD": "8",
                "PIECE_LOSS_WEIGHT": "0.1", "SEED": "42", "GRAD_ACCUM_STEPS": "1"}
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("INIT_CHECKPOINT", raising=False)
    resumed, full = tmp_path / "resumed", tmp_path / "full"
    monkeypatch.setenv("CKPT_DIR", str(resumed))
    trainer.main()
    policy, _, _, _ = load_model(str(resumed / "best.pt"), "cpu")
    assert policy(torch.tensor([[0, 2]])).shape == (1, 2, 8)
    monkeypatch.setenv("MAX_ITERS", "3")
    trainer.main()
    monkeypatch.setenv("CKPT_DIR", str(full))
    trainer.main()
    a = torch.load(resumed / "latest.pt", weights_only=False)
    b = torch.load(full / "latest.pt", weights_only=False)
    assert a["step"] == b["step"] == 3
    for key in ("model", "piece_head"):
        for name, value in a[key].items():
            assert torch.equal(value, b[key][name])
    metrics = json.loads((resumed / "metrics.jsonl").read_text().splitlines()[-1])
    assert metrics["supervised_pieces"] > 0 and 0 <= metrics["piece_accuracy"] <= 1
    monkeypatch.setenv("PIECE_LOSS_WEIGHT", "0")
    with pytest.raises(ValueError, match="Resume configuration"):
        trainer.main()


def test_head_warmup_preserves_policy_and_joint_training_loads_head(tmp_path, monkeypatch):
    from training import train_involved_pieces as trainer
    from training.common import build_checkpoint

    torch.set_num_threads(1)
    data, labels, meta, _ = dataset(tmp_path)
    original = tiny_model().policy
    initial = tmp_path / "initial.pt"
    torch.save(build_checkpoint(original, original.config, meta, 0), initial)
    monkeypatch.setattr(trainer, "get_device", lambda: torch.device("cpu"))
    settings = {"DATASET_DIR": str(data), "PIECE_LABELS_DIR": str(labels), "MAX_ITERS": "3",
                "BATCH_SIZE": "2", "VAL_BATCHES": "2", "EVAL_INTERVAL": "1",
                "PIECE_LOSS_WEIGHT": "0.1", "SEED": "42", "GRAD_ACCUM_STEPS": "1",
                "INIT_CHECKPOINT": str(initial), "LR": "1e-6", "PIECE_HEAD_LR": "0.01",
                "FREEZE_POLICY": "1", "INIT_PIECE_HEAD": "0", "CKPT_DIR": str(tmp_path / "warmup")}
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    trainer.main()
    warm = torch.load(tmp_path / "warmup/latest.pt", weights_only=False)
    assert all(torch.equal(value, warm["model"][key]) for key, value in original.state_dict().items())
    assert [group["lr"] for group in warm["optimizer"]["param_groups"]] == [1e-6, 0.01]
    assert warm["protocol"]["selection"] == "piece_loss"
    history = [json.loads(line) for line in (tmp_path / "warmup/metrics.jsonl").read_text().splitlines()]
    assert history[-1]["piece_loss"] < history[0]["piece_loss"]
    assert history[-1]["next_move_loss"] == history[0]["next_move_loss"]

    monkeypatch.setenv("INIT_CHECKPOINT", str(tmp_path / "warmup/latest.pt"))
    monkeypatch.setenv("CKPT_DIR", str(tmp_path / "joint"))
    monkeypatch.setenv("INIT_PIECE_HEAD", "1")
    monkeypatch.setenv("FREEZE_POLICY", "0")
    monkeypatch.setenv("PIECE_HEAD_LR", "0.001")
    monkeypatch.setenv("MAX_ITERS", "1")
    trainer.main()
    joint = torch.load(tmp_path / "joint/latest.pt", weights_only=False)
    assert any(not torch.equal(value, joint["model"][key]) for key, value in warm["model"].items())
    for key, value in warm["piece_head"].items():
        assert (value - joint["piece_head"][key]).abs().max() < 0.002
    assert joint["protocol"]["selection"] == "next_move_loss"


@pytest.mark.parametrize("scope", ["involved", "all_living"])
def test_collision_training_keeps_policy_fixed_and_records_full_board_metrics(tmp_path, monkeypatch, scope):
    from data_prep.prepare_piece_occupancy import prepare as prepare_occupancy, prepare_training_locations
    from training import train_involved_pieces as trainer
    from training.common import build_checkpoint

    data, labels, meta, _ = dataset(tmp_path)
    occupancy = tmp_path / "occupancy"
    prepare_occupancy(data, labels, occupancy)
    prepare_training_locations(data, labels, occupancy)
    model = tiny_model()
    initial = tmp_path / "initial.pt"
    torch.save(build_checkpoint(model.policy, model.policy.config, meta, 0,
                               piece_head=model.piece_head.state_dict()), initial)
    monkeypatch.setattr(trainer, "get_device", lambda: torch.device("cpu"))
    settings = {"DATASET_DIR": str(data), "PIECE_LABELS_DIR": str(labels),
                "PIECE_OCCUPANCY_DIR": str(occupancy), "COLLISION_LOSS_WEIGHT": "0.1",
                "CKPT_DIR": str(tmp_path / "run"), "INIT_CHECKPOINT": str(initial),
                "FREEZE_POLICY": "1", "INIT_PIECE_HEAD": "1", "MAX_ITERS": "2",
                "BATCH_SIZE": "2", "VAL_BATCHES": "2", "EVAL_INTERVAL": "1",
                "LR": "3e-6", "PIECE_HEAD_LR": "1e-3", "PIECE_LOSS_WEIGHT": "0.1",
                "SEED": "42", "GRAD_ACCUM_STEPS": "1"}
    settings["PIECE_LOCATION_SCOPE"] = scope
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    trainer.main()
    baseline = json.loads((tmp_path / "run/baseline.json").read_text())
    metrics = json.loads((tmp_path / "run/metrics.jsonl").read_text().splitlines()[-1])
    assert baseline["next_move_loss"] == metrics["next_move_loss"]
    assert metrics["living_count"] > metrics["supervised_pieces"]
    assert 0 <= metrics["collision_position_rate"] <= 1
    location_key = "living_loss" if scope == "all_living" else "piece_loss"
    assert metrics["head_loss"] == pytest.approx(0.1 * (metrics[location_key] + metrics["collision_loss"]))
    saved = torch.load(tmp_path / "run/latest.pt", weights_only=False)
    assert all(torch.equal(value, saved["model"][key]) for key, value in model.policy.state_dict().items())
    assert any(not torch.equal(value, saved["piece_head"][key]) for key, value in model.piece_head.state_dict().items())


def test_all_living_loss_supervises_uninvolved_identities_and_excludes_captured():
    from training.involved_pieces import living_piece_loss

    logits = torch.randn(1, 2, 32, 64, requires_grad=True)
    truth = torch.full((1, 2, 32), -1)
    truth[0, 0, :3] = torch.tensor([0, 8, 63])
    alive = truth >= 0
    loss = living_piece_loss(logits, truth, alive)
    expected = torch.nn.functional.cross_entropy(logits[0, 0, :3], truth[0, 0, :3])
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert (logits.grad[0, 0, :3].abs().sum(-1) > 0).all()
    assert logits.grad[0, 0, 3:].abs().sum() == 0
    assert logits.grad[0, 1].abs().sum() == 0
    assert living_piece_loss(logits, truth, torch.zeros_like(alive)).item() == 0


def test_next_move_only_continuation_updates_policy_but_preserves_fitted_head(tmp_path, monkeypatch):
    from training import train_involved_pieces as trainer
    from training.common import build_checkpoint

    data, labels, meta, _ = dataset(tmp_path)
    model = tiny_model()
    initial = tmp_path / "initial.pt"
    torch.save(build_checkpoint(model.policy, model.policy.config, meta, 0,
                               piece_head=model.piece_head.state_dict()), initial)
    monkeypatch.setattr(trainer, "get_device", lambda: torch.device("cpu"))
    settings = {"DATASET_DIR": str(data), "PIECE_LABELS_DIR": str(labels),
                "CKPT_DIR": str(tmp_path / "control"), "INIT_CHECKPOINT": str(initial),
                "PIECE_LOSS_WEIGHT": "0", "COLLISION_LOSS_WEIGHT": "0", "FREEZE_POLICY": "0",
                "INIT_PIECE_HEAD": "1", "MAX_ITERS": "1", "BATCH_SIZE": "2", "VAL_BATCHES": "1",
                "EVAL_INTERVAL": "1", "LR": "3e-6", "PIECE_HEAD_LR": "1e-3"}
    for key, value in settings.items():
        monkeypatch.setenv(key, value)
    trainer.main()
    saved = torch.load(tmp_path / "control/latest.pt", weights_only=False)
    assert any(not torch.equal(value, saved["model"][key]) for key, value in model.policy.state_dict().items())
    assert all(torch.equal(value, saved["piece_head"][key]) for key, value in model.piece_head.state_dict().items())


@pytest.mark.parametrize("sans", [
    "e4 a6 e5 d5 exd6 cxd6",
    "e4 e5 Nf3 Nc6 Bc4 Nf6 O-O Be7 d3 O-O",
    "a4 h5 a5 h4 a6 h3 axb7 hxg2 bxa8=Q gxh1=Q Qa7",
])
def test_full_training_locations_match_board_replay(tmp_path, sans):
    from data_prep.prepare_piece_occupancy import prepare as prepare_occupancy, prepare_training_locations

    data = tmp_path / "data"
    data.mkdir()
    vocabulary = list(dict.fromkeys(["<bos>", "<eos>", *sans.split()]))
    stoi = {s: i for i, s in enumerate(vocabulary)}
    meta = {"stoi": stoi, "itos": vocabulary}
    (data / "meta.pkl").write_bytes(pickle.dumps(meta))
    tokens = np.array([0, *[stoi[s] for s in sans.split()], 1], dtype=np.uint16)
    for split in ("train", "val"):
        tokens.tofile(data / f"{split}.bin")
    labels, occupancy = tmp_path / "labels", tmp_path / "occupancy"
    prepare(data, labels)
    prepare_occupancy(data, labels, occupancy)
    prepare_training_locations(data, labels, occupancy)
    np.testing.assert_array_equal(np.load(occupancy / "train_locations.npy"),
                                  np.load(occupancy / "val_locations.npy"))
