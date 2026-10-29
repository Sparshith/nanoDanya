import hashlib
import math
from pathlib import Path
import pickle


def resolve_recipe(settings):
    defaults = dict(model_type='gpt', seed=42, batch_size=64, microbatch_size=64,
        sampling={'mode': 'coverage'}, loss_weights=None, learning_rate=3e-4,
        min_learning_rate=3e-5, warmup_steps=2000, learning_rate_schedule='cosine',
        weight_decay=.1, betas=[.9, .95], grad_clip=1., eval_interval=5000,
        val_batches=32, checkpoint_interval=5000, device='cuda', compile=True,
        compile_mode='default')
    unknown = settings.keys() - defaults.keys() - {'datasets', 'model_config'}
    if unknown:
        raise ValueError(f'Unknown recipe settings: {sorted(unknown)}')
    recipe = defaults | settings
    sources = recipe['datasets']
    if not sources:
        raise ValueError('At least one dataset is required')
    recipe['datasets'] = {name: sources[name] for name in sorted(sources)}
    for key in ['batch_size', 'microbatch_size', 'eval_interval', 'val_batches', 'checkpoint_interval']:
        if not isinstance(recipe[key], int) or recipe[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if recipe['model_type'] not in ('gpt', 'student'):
        raise ValueError('model_type must be gpt or student')
    if recipe['device'] not in ('cpu', 'cuda'):
        raise ValueError('device must be cpu or cuda')
    if recipe['compile_mode'] not in ('default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'):
        raise ValueError('Unknown compile_mode')
    if recipe['learning_rate_schedule'] not in ('cosine', 'constant'):
        raise ValueError('learning_rate_schedule must be cosine or constant')
    if not 0 <= recipe['min_learning_rate'] <= recipe['learning_rate'] or recipe['learning_rate'] <= 0:
        raise ValueError('Require 0 <= min_learning_rate <= learning_rate and learning_rate > 0')
    if recipe['warmup_steps'] < 0:
        raise ValueError('warmup_steps cannot be negative')
    sampling = recipe['sampling']
    if sampling['mode'] == 'coverage':
        if set(sampling) != {'mode'}:
            raise ValueError('coverage visits every dataset once; it takes no quotas or step limit')
        if recipe['loss_weights'] is not None:
            raise ValueError('Source-mean loss weights require mixture sampling; coverage uses equal token weights')
    elif sampling['mode'] == 'mixture':
        if sampling.keys() - {'mode', 'windows_per_source', 'steps', 'coverage_source'}:
            raise ValueError('Unknown mixture sampling settings')
        quotas = sampling['windows_per_source']
        if set(quotas) != set(sources) or any(not isinstance(n, int) or n < 1 for n in quotas.values()):
            raise ValueError('windows_per_source must give every dataset a positive integer quota')
        if sum(quotas.values()) != recipe['batch_size']:
            raise ValueError('windows_per_source must sum to batch_size')
        if 'coverage_source' in sampling:
            if sampling['coverage_source'] not in sources or 'steps' in sampling:
                raise ValueError('coverage_source must name a dataset and replaces the steps setting')
        elif not isinstance(sampling.get('steps'), int) or sampling['steps'] < 1:
            raise ValueError('Random mixture sampling requires positive steps')
    else:
        raise ValueError('sampling.mode must be coverage or mixture')
    weights = recipe['loss_weights']
    if weights is not None:
        if set(weights) != set(sources) or any(not math.isfinite(w) or w <= 0 for w in weights.values()):
            raise ValueError('loss_weights must give every dataset a finite positive weight')
        total = sum(weights.values())
        recipe['loss_weights'] = {name: weight / total for name, weight in weights.items()}
    return recipe


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_data(recipe):
    import numpy as np

    data, fingerprints, meta = {}, {}, None
    for name, folder in recipe['datasets'].items():
        root = Path(folder)
        current = pickle.loads((root / 'meta.pkl').read_bytes())
        if meta is None:
            meta = current
        if current['stoi'] != meta['stoi'] or current['vocab_size'] != recipe['model_config']['vocab_size']:
            raise ValueError(f'{name}: tokenizer does not match the model and other datasets')
        data[name], fingerprints[name] = {}, {'tokenizer_sha256': file_hash(root / 'meta.pkl')}
        for split in ('train', 'val'):
            path = root / (split + '.bin')
            if not path.stat().st_size or path.stat().st_size % 2:
                raise ValueError(f'{path}: expected nonempty uint16 tokens')
            tokens = np.memmap(path, mode='r', dtype='<u2')
            targets = 0
            for start in range(0, len(tokens), 8 * 1024 * 1024):
                chunk = tokens[start:start + 8 * 1024 * 1024]
                if chunk.max() >= current['vocab_size']:
                    raise ValueError(f'{path}: token outside vocabulary')
                targets += int(np.count_nonzero(chunk != meta['stoi']['<bos>']))
            data[name][split] = tokens
            fingerprints[name][split] = dict(sha256=file_hash(path), tokens=len(tokens), targets=targets)
    return data, meta, fingerprints


class BatchStream:
    def __init__(self, data, meta, recipe):
        import numpy as np
        from training.coverage_batches import coverage_index

        self.data, self.recipe = data, recipe
        self.names = list(recipe['datasets'])
        self.bos, self.eos = meta['stoi']['<bos>'], meta['stoi']['<eos>']
        self.width = recipe['model_config']['sequence_len']
        self.rng = np.random.default_rng(recipe['seed'])
        self.step = 0
        self.sampling = recipe['sampling']
        self.indices, self.coverage = {}, {}
        covered = self.names if self.sampling['mode'] == 'coverage' else [self.sampling.get('coverage_source')]
        for name in covered:
            if name is not None:
                self.indices[name], self.coverage[name] = coverage_index(
                    data[name]['train'], self.bos, self.eos, self.width)
        for name in self.names:
            if name not in self.indices and len(data[name]['train']) <= self.width:
                raise ValueError(f'{name}: random sampling needs more than sequence_len tokens')
        if self.sampling['mode'] == 'coverage':
            self.ends = np.cumsum([len(self.indices[name]) for name in self.names])
            self.order = self.rng.permutation(int(self.ends[-1]))
            self.total_steps = math.ceil(len(self.order) / recipe['batch_size'])
        elif self.sampling.get('coverage_source'):
            name = self.sampling['coverage_source']
            self.order = self.rng.permutation(len(self.indices[name]))
            self.total_steps = math.ceil(len(self.order) / self.sampling['windows_per_source'][name])
        else:
            self.total_steps = self.sampling['steps']

    def state_dict(self):
        return dict(step=self.step, rng=self.rng.bit_generator.state)

    def load_state_dict(self, state):
        self.step = state['step']
        self.rng.bit_generator.state = state['rng']

    def take(self):
        import numpy as np
        import torch
        from training.context_batches import sample_context_batch
        from training.coverage_batches import coverage_batch

        if self.step >= self.total_steps:
            raise StopIteration
        x = torch.full((self.recipe['batch_size'], self.width), self.eos, dtype=torch.long)
        y = torch.full_like(x, -1)
        source_ids = torch.full((len(x),), -1, dtype=torch.long)
        if self.sampling['mode'] == 'coverage':
            start = self.step * len(x)
            chosen = self.order[start:start + len(x)]
            owners = np.searchsorted(self.ends, chosen, side='right')
            for source, name in enumerate(self.names):
                rows = np.flatnonzero(owners == source)
                offset = 0 if source == 0 else self.ends[source - 1]
                bx, by = coverage_batch(self.data[name]['train'], self.indices[name],
                    chosen[rows] - offset, len(rows), self.width, self.bos, self.eos)
                x[rows], y[rows], source_ids[rows] = bx, by, source
        else:
            offset = 0
            for source, name in enumerate(self.names):
                count = self.sampling['windows_per_source'][name]
                if name in self.indices:
                    start = self.step * count
                    bx, by = coverage_batch(self.data[name]['train'], self.indices[name],
                        self.order[start:start + count], count, self.width, self.bos, self.eos)
                else:
                    bx, by = sample_context_batch(self.data[name]['train'], self.rng,
                        count, self.width, self.bos, 'cpu')
                x[offset:offset + count], y[offset:offset + count] = bx, by
                source_ids[offset:offset + count] = source
                offset += count
        self.step += 1
        return x, y, source_ids


def target_counts(y, source_ids, names):
    return {name: int((y[source_ids == i] != -1).sum()) for i, name in enumerate(names)}


def loss_scales(counts, weights):
    if weights is None:
        total = sum(counts.values())
        if not total:
            raise ValueError('Batch has no supervised targets')
        return {name: 1 / total for name in counts}
    if any(counts[name] == 0 for name in weights):
        raise ValueError('Source-mean loss needs supervised targets from every source in the batch')
    return {name: weights[name] / counts[name] for name in counts}
