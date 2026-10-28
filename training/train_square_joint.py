"""Matched continuation with a balanced square loss, or a detached control head."""
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data_prep.prepare_involved_pieces import file_hash
from inference.board_input import BoardInputPlayer
from model_loading import load_model
from probes.board_input_readout import state_hash
from probes.square_readout import square_loss, square_loss_parts, square_targets
from training.board_input import BoardInputModel
from training.common import build_checkpoint, forward_hidden, get_device, move_logits


class JointSquareModel(nn.Module):
    def __init__(self, player, head_state, joint):
        super().__init__()
        self.backbone = player.model if isinstance(player, BoardInputPlayer) else player
        self.backbone.requires_grad_(True)
        self.policy = self.backbone.policy if isinstance(self.backbone, BoardInputModel) else self.backbone
        self.config = self.policy.config
        self.head = nn.Linear(self.config.n_embd, 64 * 13).to(self.policy.transformer.wte.weight.device)
        self.head.load_state_dict(head_state)
        self.joint = joint

    def forward(self, x, boards):
        additions = None
        if isinstance(self.backbone, BoardInputModel):
            if not self.backbone.enabled:
                boards = torch.stack((torch.full_like(boards[..., 0], 64), torch.zeros_like(boards[..., 1])), -1)
            additions = self.backbone.encoder(boards)
        hidden = forward_hidden(self.policy, x, additions)
        squares = self.head(hidden if self.joint else hidden.detach()).unflatten(-1, (64, 13))
        return move_logits(self.policy, hidden), squares


def train(checkpoint, head_path, data_dir, board_dir, output, joint, steps=1000,
          batch_size=32, weight=0.1, lr=3e-6, head_lr=1e-4, val_batches=10, interval=100):
    data_dir, board_dir, output = map(Path, (data_dir, board_dir, output))
    output.mkdir(parents=True, exist_ok=True)
    device = get_device()
    torch.manual_seed(42)
    player, config, stoi, _ = load_model(checkpoint, str(device))
    saved_head = torch.load(head_path, map_location='cpu', weights_only=False)
    assert saved_head['protocol']['version'] == 3
    assert saved_head['protocol']['checkpoint_sha256'] == file_hash(checkpoint)
    model = JointSquareModel(player, saved_head['head'], joint).to(device)
    del player, saved_head
    meta = pickle.loads((data_dir / 'meta.pkl').read_bytes())
    manifest = json.loads((board_dir / 'manifest.json').read_text())
    assert meta['stoi'] == stoi
    assert file_hash(data_dir / 'meta.pkl') == manifest['source_labels']['meta_sha256']
    splits = {}
    for split in ('train', 'val'):
        tokens_path, boards_path = data_dir / f'{split}.bin', board_dir / f'{split}.npy'
        assert file_hash(tokens_path) == manifest['source_labels']['splits'][split]['source_sha256']
        assert file_hash(boards_path) == manifest['splits'][split]['features_sha256']
        splits[split] = (np.memmap(tokens_path, dtype=np.uint16, mode='r'), np.load(boards_path, mmap_mode='r'))
    protocol = {'version': 1, 'joint': joint, 'board_weight': weight, 'steps': steps, 'batch_size': batch_size,
                'sequence_len': config.sequence_len, 'lr': lr, 'encoder_lr': lr, 'head_lr': head_lr,
                'seed': 42, 'val_batches': val_batches, 'interval': interval, 'weight_decay': 0.01,
                'initial_checkpoint': checkpoint, 'initial_sha256': file_hash(checkpoint),
                'initial_head': head_path, 'initial_head_sha256': file_hash(head_path),
                'initial_backbone_state_hash': state_hash(model.backbone),
                'initial_head_state_hash': state_hash(model.head), 'data': manifest,
                'loss': 'next-move CE + 0.1 * per-board (0.5 occupied CE + 0.5 empty CE)',
                'control': 'same head loss and optimizer, hidden detached before board head',
                'clipping': 'backbone and head separately, max norm 1',
                'sampling': 'uniform token-window starts, seed 42; no stage resampling',
                'selection': 'fixed final step, no best-checkpoint selection'}
    optimizer = torch.optim.AdamW([{'params': model.backbone.parameters(), 'lr': lr},
                                  {'params': model.head.parameters(), 'lr': head_lr}])
    rng = torch.Generator().manual_seed(42)
    start = 1
    if (output / 'latest.pt').exists():
        saved = torch.load(output / 'latest.pt', map_location='cpu', weights_only=False)
        assert saved['protocol'] == protocol
        model.policy.load_state_dict(saved['model'])
        if isinstance(model.backbone, BoardInputModel):
            model.backbone.encoder.load_state_dict(saved['board_input']['encoder'])
        model.head.load_state_dict(saved['square_head'])
        optimizer.load_state_dict(saved['optimizer'])
        rng.set_state(saved['train_rng'])
        start = saved['step'] + 1
        del saved
    compiled = torch.compile(model) if device.type == 'cuda' else model

    def batch(split, generator):
        tokens, boards = splits[split]
        starts = torch.randint(len(tokens) - config.sequence_len, (batch_size,), generator=generator).tolist()
        x = np.stack([tokens[i:i+config.sequence_len] for i in starts]).astype(np.int64)
        y = np.stack([tokens[i+1:i+config.sequence_len+1] for i in starts]).astype(np.int64)
        b = np.stack([boards[i:i+config.sequence_len] for i in starts]).astype(np.int64)
        return [torch.from_numpy(a).to(device) for a in (x, y, b)]

    def autocast():
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda')

    @torch.no_grad()
    def evaluate():
        model.eval()
        generator = torch.Generator().manual_seed(43)
        move_sum = complete_sum = occupied_sum = empty_sum = 0.0
        count = complete_count = board_count = exact = 0
        for _ in range(val_batches):
            x, y, boards = batch('val', generator)
            with autocast():
                moves, squares = compiled(x, boards)
                move_ce = F.cross_entropy(moves.flatten(0, 1), y.flatten(), reduction='none').view_as(y)
            occupied, empty, valid = square_loss_parts(squares.float(), x, boards, stoi)
            complete = valid & (y != stoi['<bos>']) & (y != stoi['<eos>'])
            move_sum += float(move_ce.sum())
            count += y.numel()
            complete_sum += float(move_ce[complete].sum())
            complete_count += int(complete.sum())
            occupied_sum += float(occupied[valid].sum())
            empty_sum += float(empty[valid].sum())
            board_count += int(valid.sum())
            exact += int(((squares.argmax(-1) == square_targets(boards)).all(-1) & valid).sum())
        return {'move_loss': move_sum / count, 'complete_prefix_move_loss': complete_sum / complete_count,
                'balanced_board_loss': (occupied_sum + empty_sum) / (2 * board_count),
                'occupied_loss': occupied_sum / board_count, 'empty_loss': empty_sum / board_count,
                'attached_head_exact_board_rate': exact / board_count, 'board_positions': board_count,
                'complete_prefix_moves': complete_count}

    def save(name, step, resume):
        extras = {'square_head': model.head.state_dict(), 'protocol': protocol}
        if isinstance(model.backbone, BoardInputModel):
            extras['board_input'] = model.backbone.checkpoint_metadata()
        if resume:
            extras.update(optimizer=optimizer.state_dict(), train_rng=rng.get_state())
        path = output / name
        temporary = path.with_suffix('.tmp')
        torch.save(build_checkpoint(model.policy, config, meta, step, **extras), temporary)
        temporary.replace(path)

    print(json.dumps({'joint': joint, 'protocol': protocol}), flush=True)
    if start == 1:
        baseline = {'step': 0, **evaluate()}
        (output / 'baseline.json').write_text(json.dumps(baseline, indent=2) + '\n')
        print(json.dumps({'joint': joint, **baseline}), flush=True)
    for step in range(start, steps + 1):
        model.train()
        x, y, boards = batch('train', rng)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            moves, squares = compiled(x, boards)
            move_loss = F.cross_entropy(moves.flatten(0, 1), y.flatten())
            board_loss = square_loss(squares, x, boards, stoi)
            loss = move_loss + weight * board_loss
        loss.backward()
        backbone_norm = torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), 1.0, error_if_nonfinite=True)
        head_norm = torch.nn.utils.clip_grad_norm_(model.head.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % interval == 0 or step == steps:
            row = {'step': step, 'train_move_loss': float(move_loss.detach()),
                   'train_board_loss': float(board_loss.detach()), 'train_total_loss': float(loss.detach()),
                   'backbone_grad_norm': float(backbone_norm), 'head_grad_norm': float(head_norm), **evaluate()}
            with (output / 'metrics.jsonl').open('a') as log:
                log.write(json.dumps(row) + '\n')
            print(json.dumps({'joint': joint, **row}), flush=True)
            save('latest.pt', step, True)
    save('final.pt', steps, False)
    final_hash = state_hash(model.backbone)
    assert final_hash != protocol['initial_backbone_state_hash']
    result = {'protocol': protocol, 'checkpoint': str(output / 'final.pt'),
              'sha256': file_hash(output / 'final.pt'), 'final_backbone_state_hash': final_hash,
              'baseline': json.loads((output / 'baseline.json').read_text()),
              'metrics': [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]}
    (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
