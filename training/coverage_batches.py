import numpy as np
import torch


def coverage_index(tokens, bos, eos, width=512, chunk_size=16 * 1024 * 1024):
    starts, eos_count = [], 0
    for offset in range(0, len(tokens), chunk_size):
        block = tokens[offset:offset + chunk_size]
        starts.append(np.flatnonzero(block == bos) + offset)
        eos_count += int(np.count_nonzero(block == eos))
    starts = np.concatenate(starts)
    assert starts[0] == 0 and tokens[-1] == eos and len(starts) == eos_count
    assert np.all(tokens[starts[1:] - 1] == eos)
    boundaries = np.append(starts, len(tokens))
    lengths = np.diff(boundaries)
    assert lengths.min() >= 2
    extra = int(np.maximum(0, (lengths - width - 1 + width // 2 - 1) // (width // 2)).sum())
    windows = np.empty((len(starts) + extra, 3), dtype=np.int64)
    game, row = 0, 0
    while game < len(starts):
        start, end = int(boundaries[game]), int(boundaries[game + 1])
        if end - start > width + 1:
            target = start + 1
            stop = start + width + 1
            windows[row] = start, stop, target
            row += 1
            target = stop
            while target < end:
                stop = min(end, target + width // 2)
                windows[row] = target - width // 2, stop, target
                row += 1
                target = stop
            game += 1
        else:
            following = int(np.searchsorted(boundaries, start + width + 1, side='right')) - 1
            windows[row] = start, boundaries[following], start + 1
            row += 1
            game = following
    windows = windows[:row].copy()
    gaps = windows[1:, 2] - windows[:-1, 1]
    assert np.all((gaps == 0) | (gaps == 1))
    assert np.all(tokens[windows[1:, 2][gaps == 1] - 1] == bos)
    assert windows[0, 2] == 1 and windows[-1, 1] == len(tokens)
    assert np.all(windows[:, 1] - windows[:, 0] - 1 <= width)
    # Intervals partition every target except the BOS at a new window boundary.
    omitted_bos = 1 + int(np.count_nonzero(gaps))
    assert int((windows[:, 1] - windows[:, 2]).sum()) == len(tokens) - omitted_bos
    summary = dict(corpus_tokens=len(tokens), games=len(starts), windows=len(windows),
        supervised_tokens=len(tokens) - len(starts), excluded_games=0,
        long_games=int(np.count_nonzero(lengths > width + 1)),
        context_only_tokens=int((windows[:, 2] - windows[:, 0] - 1).sum()),
        width=width, interval_partition_verified=True)
    return windows, summary


class CoverageSampler:
    def __init__(self, windows, seed, cursor=0):
        self.windows = windows
        self.cursor = cursor
        self.order = np.random.default_rng(seed).permutation(len(windows)).astype(np.int32)

    def take(self, count):
        selected = self.order[self.cursor:self.cursor + count]
        self.cursor += len(selected)
        return selected


def coverage_batch(tokens, windows, selected, batch_size, width, bos, eos):
    x = np.full((batch_size, width), eos, dtype=np.int64)
    y = np.full((batch_size, width), -1, dtype=np.int64)
    for row, window in enumerate(windows[selected]):
        start, stop, target = map(int, window)
        length = stop - start - 1
        x[row, :length] = tokens[start:stop - 1]
        y[row, :length] = tokens[start + 1:stop]
        y[row, :target - start - 1] = -1
    y[y == bos] = -1
    return torch.from_numpy(x), torch.from_numpy(y)
