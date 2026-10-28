import json
import pickle

import numpy as np
import pytest
import torch

from data_prep.prepare_board_inputs import prepare as prepare_boards
from data_prep.prepare_involved_pieces import prepare as prepare_labels
from inference.board_input import BoardInputPlayer
from nanochat.gpt import GPT, GPTConfig
from probes.board_input_readout import FrozenReadout, counts, run_readout, sequence_batch, state_hash, targets
from training.board_input import BoardInputModel
from training.common import build_checkpoint


@pytest.mark.parametrize("arm", ["champion", "control", "oracle"])
@pytest.mark.parametrize("classes,slots", [(64, 32), (71, 32), (13, 64)])
def test_readout_uses_actual_final_state_and_never_updates_backbone(arm, classes, slots):
    vocabulary = ["<bos>", "<eos>", "e4", "d5", "exd5"]
    config = GPTConfig(sequence_len=16, vocab_size=5, n_layer=2, n_head=2, n_kv_head=2, n_embd=16)
    policy = GPT(config)
    player = policy
    if arm != "champion":
        model = BoardInputModel(policy, enabled=arm == "oracle", piece_dim=4)
        torch.nn.init.normal_(model.encoder.projection.weight, std=0.1)
        player = BoardInputPlayer(model, vocabulary)
    x, boards = sequence_batch([[0, 2, 3, 4]], vocabulary, 1, "cpu")
    captured = []
    hook = policy.lm_head.register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach()))
    with torch.no_grad():
        player(x)
    hook.remove()
    readout = FrozenReadout(player, classes_per_piece=classes, slots=slots)
    before = state_hash(readout.backbone)
    actual = readout(x, boards)
    expected = readout.head(captured[0]).unflatten(-1, (slots, classes))
    torch.testing.assert_close(actual, expected)
    optimizer = torch.optim.AdamW(readout.head.parameters(), lr=1e-3)
    actual.square().mean().backward()
    assert readout.head.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in readout.backbone.parameters())
    optimizer.step()
    assert before == state_hash(readout.backbone)
    if arm != "oracle":
        modified = boards.clone()
        modified[..., 0], modified[..., 1] = 64, 0
        torch.testing.assert_close(readout(x, boards), readout(x, modified))


def test_masks_exclude_incomplete_histories_captured_pieces_and_padding():
    vocabulary = ["<bos>", "<eos>", "e4", "d5", "exd5"]
    x, boards = sequence_batch([[0, 2, 3, 4], [0, 2]], vocabulary, 1, "cpu")
    truth, alive = targets(x, boards, 0)
    assert alive[0].sum(1).tolist() == [32, 32, 32, 31]
    assert alive[1].sum(1).tolist() == [32, 32, 0, 0]
    _, cropped = targets(x[:, 1:], boards[:, 1:], 0)
    assert not cropped.any()
    logits = torch.zeros(2, 4, 32, 64)
    logits.scatter_(-1, truth.clamp_min(0).unsqueeze(-1), 20)
    metrics = counts(logits, truth, alive)
    assert metrics["living_correct"] == metrics["living_count"] == 191
    assert metrics["exact_boards"] == metrics["scored_positions"] == 6
    assert metrics["collision_positions"] == 0


@pytest.mark.parametrize("mode", ["living", "identity", "square"])
def test_complete_frozen_readout_training_scoring_and_resume(tmp_path, monkeypatch, mode):
    full_board, square_board = mode == "identity", mode == "square"
    vocabulary = ["<bos>", "<eos>", "e4", "d5", "exd5", "Qxd5"]
    stoi = {s: i for i, s in enumerate(vocabulary)}
    meta = {"itos": vocabulary, "stoi": stoi, "context_length": 16, "vocab_size": len(vocabulary)}
    data = tmp_path / "data"
    data.mkdir()
    (data / "meta.pkl").write_bytes(pickle.dumps(meta))
    for split in ("train", "val", "probe"):
        np.array([0, 2, 3, 4, 5, 1] * 5, dtype=np.uint16).tofile(data / f"{split}.bin")
    prepare_labels(data, tmp_path / "labels")
    prepare_boards(data, tmp_path / "labels", tmp_path / "boards")
    config = GPTConfig(sequence_len=16, vocab_size=6, n_layer=1, n_head=1, n_kv_head=1, n_embd=8)
    model = BoardInputModel(GPT(config))
    torch.nn.init.normal_(model.encoder.projection.weight, std=0.1)
    path = tmp_path / "policy.pt"
    torch.save(build_checkpoint(model.policy, config, meta, 1, board_input=model.checkpoint_metadata()), path)
    monkeypatch.setattr("probes.board_input_readout.get_device", lambda: torch.device("cpu"))
    args = (str(path), data, tmp_path / "boards", "1. e4 d5 2. exd5 Qxd5 *", tmp_path / "readout")
    result = run_readout(*args, steps=2, batch_size=1, val_batches=2, full_board=full_board,
                         square_board=square_board)
    assert result["backbone_bitwise_unchanged"]
    assert len(result["trace"]["timeline"]) == 5
    if full_board or square_board:
        assert result["trace"]["aggregate"]["counts"]["actual"] == 125
        assert result["probe"]["aggregate"]["counts"]["actual"] == 5 * 125
        predictions = np.load(tmp_path / "readout/probe-predictions.npz")
        assert predictions['actual'].shape == predictions['predicted'].shape == ((25, 64) if square_board else (25, 32, 2))
    else:
        assert result["trace"]["aggregate"]["living_count"] == 157
        assert result["probe"]["aggregate"]["living_count"] == 5 * 157
    assert result["trace"]["timeline"][3]["next_move"] == "Qxd5"
    resumed = run_readout(*args, steps=2, batch_size=1, val_batches=2, full_board=full_board,
                          square_board=square_board)
    assert resumed["probe"] == result["probe"]
    assert len((tmp_path / "readout/metrics.jsonl").read_text().splitlines()) == (3 if square_board else 2)
    assert json.loads((tmp_path / "readout/result.json").read_text())["protocol"]["steps"] == 2
