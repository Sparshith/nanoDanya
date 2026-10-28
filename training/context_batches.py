import numpy as np
import torch


def context_targets(blocks, bos_id):
    x = blocks[:, :-1].copy()
    y = blocks[:, 1:].copy()
    has_start = np.logical_or.accumulate(x == bos_id, axis=1)
    y[~has_start | (y == bos_id)] = -1
    return x, y


def sample_context_batch(data, rng, batch_size, block_size, bos_id, device):
    starts = rng.integers(0, len(data) - block_size, size=batch_size)
    blocks = np.stack([data[i:i + block_size + 1] for i in starts]).astype(np.int64)
    x, y = context_targets(blocks, bos_id)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
