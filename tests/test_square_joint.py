import copy
import pickle

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from data_prep.prepare_board_inputs import prepare as prepare_boards
from data_prep.prepare_involved_pieces import prepare as prepare_labels
from inference.board_input import BoardInputPlayer
from model_loading import load_model
from nanochat.gpt import GPT, GPTConfig
from probes.board_input_readout import FrozenReadout, sequence_batch, state_hash
from probes.square_readout import square_loss
from training.board_input import BoardInputModel
from training.common import build_checkpoint
from training.train_square_joint import JointSquareModel, train


def models(joint):
    vocab = ['<bos>', '<eos>', 'e4', 'd5', 'exd5']
    config = GPTConfig(sequence_len=16, vocab_size=5, n_layer=2, n_head=2, n_kv_head=2, n_embd=16)
    policy = BoardInputModel(GPT(config))
    torch.nn.init.normal_(policy.encoder.projection.weight, std=0.1)
    player = BoardInputPlayer(policy, vocab)
    head = FrozenReadout(player, classes_per_piece=13, slots=64)
    model = JointSquareModel(player, head.head.state_dict(), joint)
    x, boards = sequence_batch([[0, 2, 3, 4]], vocab, 1, 'cpu')
    return model, player, x, boards


@pytest.mark.parametrize('joint', [False, True])
def test_joint_move_logits_match_inference_and_board_gradient_routing(joint):
    model, player, x, boards = models(joint)
    moves, squares = model(x, boards)
    torch.testing.assert_close(moves, player(x))
    assert squares.shape == (1, 4, 64, 13)
    square_loss(squares, x, boards, {'<bos>': 0, '<eos>': 1}).backward()
    assert model.head.weight.grad.abs().sum() > 0
    if joint:
        assert model.policy.transformer.wte.weight.grad.abs().sum() > 0
        assert model.backbone.encoder.projection.weight.grad.abs().sum() > 0
    else:
        assert all(p.grad is None for p in model.backbone.parameters())


def test_control_policy_update_is_identical_to_next_move_only_update():
    control, _, x, boards = models(False)
    reference = copy.deepcopy(control)
    y = torch.tensor([[2, 3, 4, 1]])
    for model, use_board in ((control, True), (reference, False)):
        optimizer = torch.optim.AdamW([{'params': model.backbone.parameters(), 'lr': 3e-6},
                                       {'params': model.head.parameters(), 'lr': 1e-4}])
        moves, squares = model(x, boards)
        loss = F.cross_entropy(moves.flatten(0, 1), y.flatten())
        if use_board:
            loss = loss + 0.1 * square_loss(squares, x, boards, {'<bos>': 0, '<eos>': 1})
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1)
        optimizer.step()
    assert state_hash(control.backbone) == state_hash(reference.backbone)
    assert state_hash(control.head) != state_hash(reference.head)


def test_tiny_training_resume_and_inference_checkpoint(tmp_path, monkeypatch):
    from data_prep.prepare_involved_pieces import file_hash
    vocab = ['<bos>', '<eos>', 'e4', 'd5', 'exd5', 'Qxd5']
    stoi = {s: i for i, s in enumerate(vocab)}
    meta = {'itos': vocab, 'stoi': stoi, 'context_length': 16, 'vocab_size': len(vocab)}
    data = tmp_path / 'dataset'
    data.mkdir()
    (data / 'meta.pkl').write_bytes(pickle.dumps(meta))
    for split in ('train', 'val', 'probe'):
        np.array([0, 2, 3, 4, 5, 1] * 5, dtype=np.uint16).tofile(data / f'{split}.bin')
    prepare_labels(data, tmp_path / 'labels')
    prepare_boards(data, tmp_path / 'labels', tmp_path / 'boards')
    config = GPTConfig(sequence_len=16, vocab_size=6, n_layer=1, n_head=1, n_kv_head=1, n_embd=8)
    policy = BoardInputModel(GPT(config))
    checkpoint = tmp_path / 'initial.pt'
    torch.save(build_checkpoint(policy.policy, config, meta, 1, board_input=policy.checkpoint_metadata()), checkpoint)
    player, _, _, _ = load_model(str(checkpoint), 'cpu')
    readout = FrozenReadout(player, classes_per_piece=13, slots=64)
    head_path = tmp_path / 'head.pt'
    torch.save({'protocol': {'version': 3, 'checkpoint_sha256': file_hash(checkpoint)},
                'head': readout.head.state_dict()}, head_path)
    monkeypatch.setattr('training.train_square_joint.get_device', lambda: torch.device('cpu'))
    args = (str(checkpoint), str(head_path), data, tmp_path / 'boards', tmp_path / 'joint', True)
    result = train(*args, steps=2, batch_size=1, val_batches=1, interval=1)
    final, _, _, _ = load_model(result['checkpoint'], 'cpu')
    assert state_hash(final.model) == result['final_backbone_state_hash']
    assert result['final_backbone_state_hash'] != state_hash(player.model)
    resumed = train(*args, steps=2, batch_size=1, val_batches=1, interval=1)
    assert resumed['metrics'] == result['metrics']
    assert resumed['final_backbone_state_hash'] == result['final_backbone_state_hash']
