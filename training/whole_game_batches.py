import numpy as np
import torch


def game_index(tokens, bos, eos, max_inputs=512, chunk_size=16 * 1024 * 1024):
    starts, ends = [], 0
    for offset in range(0, len(tokens), chunk_size):
        block = tokens[offset:offset + chunk_size]
        starts.append(np.flatnonzero(block == bos) + offset)
        ends += int(np.count_nonzero(block == eos))
    starts = np.concatenate(starts)
    assert starts[0] == 0 and tokens[-1] == eos and len(starts) == ends
    assert np.all(tokens[starts[1:] - 1] == eos)
    lengths = np.diff(np.append(starts, len(tokens))).astype(np.int32)
    assert lengths.min() >= 2
    keep = lengths <= max_inputs + 1
    return dict(starts=starts[keep], lengths=lengths[keep], excluded=int((~keep).sum()))


class GameSampler:
    def __init__(self, index, seed, state=None):
        self.index = index
        self.seed = seed
        self.epoch = 0 if state is None else state['epoch']
        self.cursor = 0 if state is None else state['cursor']
        self.shuffle()

    def shuffle(self):
        self.order = np.arange(len(self.index['lengths']), dtype=np.int32)
        np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch])).shuffle(self.order)

    def state(self):
        return dict(epoch=self.epoch, cursor=self.cursor)

    def take(self, minimum_tokens):
        selected, count = [], 0
        while count < minimum_tokens:
            if self.cursor == len(self.order):
                self.epoch += 1
                self.cursor = 0
                self.shuffle()
            candidates = self.order[self.cursor:self.cursor + 1024]
            cumulative = np.cumsum(self.index['lengths'][candidates] - 1)
            take = min(len(candidates), int(np.searchsorted(cumulative, minimum_tokens - count)) + 1)
            selected.append(candidates[:take].copy())
            count += int(cumulative[take - 1])
            self.cursor += take
        return np.concatenate(selected), count


def collate_games(tokens, index, selected, width, eos):
    starts = index['starts'][selected]
    lengths = index['lengths'][selected] - 1
    positions = np.arange(width)[None, :]
    valid = positions < lengths[:, None]
    offsets = starts[:, None] + np.where(valid, positions, 0)
    x = np.where(valid, tokens[offsets], eos).astype(np.int64)
    y = tokens[offsets + 1].astype(np.int64)
    y[~valid] = -1
    return x, y


def microbatches(data, indices, selected, eos, buckets=(64, 128, 256, 512), max_tokens=32768):
    assignments = {d: np.searchsorted(buckets, indices[d]['lengths'][selected[d]] - 1)
                   for d in selected}
    for bucket, width in enumerate(buckets):
        rows, domains = [], []
        for domain_id, domain in enumerate(['human', 'leela']):
            games = selected[domain][assignments[domain] == bucket]
            if len(games):
                rows.append(collate_games(data[domain + '_train'], indices[domain], games, width, eos))
                domains.append(np.full(len(games), domain_id, dtype=np.int64))
        if not rows:
            continue
        x, y = [np.concatenate([row[i] for row in rows]) for i in range(2)]
        domain_ids = np.concatenate(domains)
        batch_size = max_tokens // width
        for start in range(0, len(x), batch_size):
            xx, yy, dd = [a[start:start + batch_size] for a in [x, y, domain_ids]]
            padding = (-len(xx)) % 8
            if padding:
                xx = np.pad(xx, ((0, padding), (0, 0)), constant_values=eos)
                yy = np.pad(yy, ((0, padding), (0, 0)), constant_values=-1)
                dd = np.pad(dd, (0, padding))
            yield tuple(torch.from_numpy(a) for a in [xx, yy, dd])


def domain_weighted_loss(logits, targets, domain_ids, domain_tokens):
    losses = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1), ignore_index=-1, reduction='none').reshape(targets.shape).sum(dim=1)
    weights = torch.where(domain_ids == 0, .5 / domain_tokens['human'], .5 / domain_tokens['leela'])
    return (losses * weights).sum()
