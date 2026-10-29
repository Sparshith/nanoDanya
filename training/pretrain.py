import argparse
from functools import partial
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from nanochat.gpt import GPT, GPTConfig
from training.context_batches import sample_context_batch
from training.coverage_batches import coverage_batch, coverage_index
from training.recipe import BatchStream, load_data, loss_scales, resolve_recipe, target_counts


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def learning_rate(step, total_steps, recipe):
    peak = recipe['learning_rate']
    warmup = min(recipe['warmup_steps'], max(0, total_steps - 1))
    if step <= warmup:
        return peak * step / warmup
    if recipe['learning_rate_schedule'] == 'constant':
        return peak
    progress = (step - warmup) / (total_steps - warmup)
    return recipe['min_learning_rate'] + .5 * (peak - recipe['min_learning_rate']) * (1 + math.cos(math.pi * progress))


def token_losses(model, x, y):
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=x.device.type == 'cuda'):
        logits = model(x)
        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
            ignore_index=-1, reduction='none').reshape(y.shape)


def training_loss(model, x, y, row_scales):
    return (token_losses(model, x, y).sum(dim=1) * row_scales).sum()


def make_loss(model, recipe):
    loss = partial(training_loss, model)
    if recipe['compile'] and recipe['device'] == 'cuda':
        return torch.compile(loss, fullgraph=True, mode=recipe['compile_mode'])
    return loss


def backward_batch(model, batch, recipe, loss_fn=None):
    if (loss_fn is not None and recipe.get('compile') and recipe['device'] == 'cuda'
            and recipe['compile_mode'] in ('reduce-overhead', 'max-autotune')):
        torch.compiler.cudagraph_mark_step_begin()
    loss_fn = loss_fn or partial(training_loss, model)
    x, y, source_ids = batch
    names = list(recipe['datasets'])
    counts = target_counts(y, source_ids, names)
    scales = loss_scales(counts, recipe['loss_weights'])
    row_scales = torch.tensor([scales[name] for name in names])[source_ids.clamp_min(0)]
    total_loss = 0.
    for start in range(0, len(x), recipe['microbatch_size']):
        stop = start + recipe['microbatch_size']
        xb, yb = x[start:stop].to(recipe['device']), y[start:stop].to(recipe['device'])
        # Normalize against the entire update, including all accumulated microbatches.
        loss = loss_fn(xb, yb, row_scales[start:stop].to(recipe['device']))
        loss.backward()
        total_loss += float(loss.detach())
    return total_loss, counts


def evaluate(model, data, meta, recipe, fingerprints):
    model.eval()
    metrics = {}
    bos, eos = meta['stoi']['<bos>'], meta['stoi']['<eos>']
    width = recipe['model_config']['sequence_len']
    with torch.no_grad():
        for source, name in enumerate(recipe['datasets']):
            rng = np.random.default_rng(recipe['seed'] + 100000 + source)
            tokens = data[name]['val']
            windows = coverage_index(tokens, bos, eos, width)[0] if len(tokens) <= width else None
            loss_sum, count, move_loss, moves = 0., 0, 0., 0
            for _ in range(recipe['val_batches']):
                if windows is None:
                    x, y = sample_context_batch(tokens, rng, recipe['batch_size'], width, bos, 'cpu')
                else:
                    chosen = rng.integers(len(windows), size=recipe['batch_size'])
                    x, y = coverage_batch(tokens, windows, chosen, len(chosen), width, bos, eos)
                for start in range(0, len(x), recipe['microbatch_size']):
                    stop = start + recipe['microbatch_size']
                    xb, yb = x[start:stop].to(recipe['device']), y[start:stop].to(recipe['device'])
                    losses = token_losses(model, xb, yb)
                    valid, move = yb != -1, (yb != -1) & (yb != eos)
                    loss_sum += float(losses.sum())
                    count += int(valid.sum())
                    move_loss += float(losses[move].sum())
                    moves += int(move.sum())
            if not count or not moves:
                raise ValueError(f'{name}: validation sample contains no supervised moves')
            metrics[name] = dict(ce=loss_sum / count, move_ce=move_loss / moves, targets=count)
    weights = recipe['loss_weights']
    if weights is None:
        if recipe['sampling']['mode'] == 'coverage':
            weights = {name: fingerprints[name]['train']['targets'] for name in metrics}
        else:
            weights = {name: recipe['sampling']['windows_per_source'][name] * metrics[name]['targets']
                       for name in metrics}
    metrics['selection_ce'] = sum(weights[name] * metrics[name]['ce'] for name in weights) / sum(weights.values())
    if not math.isfinite(metrics['selection_ce']):
        raise FloatingPointError('Non-finite validation loss')
    model.train()
    return metrics


def train(settings, output_dir, stop_after_step=None, deadline=None, commit=lambda: None):
    recipe = resolve_recipe(settings)
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('This trainer currently supports one process; do not launch with torchrun')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / 'recipe.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != recipe:
        raise ValueError('Recipe differs from this run; use a new output directory')
    write_json(protocol_path, recipe)
    data, meta, fingerprints = load_data(recipe)
    stream = BatchStream(data, meta, recipe)
    torch.manual_seed(recipe['seed'])
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('high')
    config = GPTConfig(**recipe['model_config'])
    if recipe['model_type'] == 'student':
        from training.student import StudentGPT
        model = StudentGPT(config)
    else:
        model = GPT(config)
        model.init_weights()
    model = model.to(recipe['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['learning_rate'],
        betas=tuple(recipe['betas']), weight_decay=recipe['weight_decay'], fused=recipe['device'] == 'cuda')
    history, trained, best, best_step = [], dict.fromkeys(recipe['datasets'], 0), float('inf'), 0
    latest = output / 'latest.pt'
    if latest.exists():
        state = torch.load(latest, map_location='cpu', weights_only=False)
        if state['recipe'] != recipe or state['fingerprints'] != fingerprints or state['total_steps'] != stream.total_steps:
            raise ValueError('Checkpoint recipe or dataset contents differ from this run')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        stream.load_state_dict(state['sampler'])
        torch.set_rng_state(state['torch_rng'])
        if recipe['device'] == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        history, trained = state['history'], state['trained_tokens']
        best, best_step = state['best_metric'], state['best_step']
        del state
    compiled = torch.compile(model) if recipe['compile'] and recipe['device'] == 'cuda' else model
    loss_fn = make_loss(model, recipe)
    checkpoint_meta = dict(model_config=config.__dict__, tokenizer=meta)
    if recipe['model_type'] == 'student':
        checkpoint_meta.update(model_type='student', model_family='student_tied', qat=False)

    def save(name, resumable=False):
        state = dict(model=model.state_dict(), meta=checkpoint_meta, recipe=recipe,
            fingerprints=fingerprints, step=stream.step, total_steps=stream.total_steps,
            sampler=stream.state_dict(), best_metric=best, best_step=best_step,
            trained_tokens=trained.copy(), history=history)
        if resumable:
            state.update(optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if recipe['device'] == 'cuda' else [])
        temporary = output / (name + '.partial')
        torch.save(state, temporary)
        temporary.replace(output / name)

    def validation():
        nonlocal best, best_step
        metrics = evaluate(compiled, data, meta, recipe, fingerprints)
        history.append(dict(step=stream.step, **metrics))
        if metrics['selection_ce'] < best:
            best, best_step = metrics['selection_ce'], stream.step
            save('best.pt')
        write_json(output / 'metrics.json', history)
        print('VALIDATION=' + json.dumps(history[-1]), flush=True)

    if not history:
        validation()
        save('latest.pt', resumable=True)
        commit()
    started = time.monotonic()
    while stream.step < stream.total_steps:
        if (stop_after_step is not None and stream.step >= stop_after_step) or (deadline is not None and time.time() >= deadline):
            break
        lr = learning_rate(stream.step + 1, stream.total_steps, recipe)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        loss, counts = backward_batch(model, stream.take(), recipe, loss_fn=loss_fn)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), recipe['grad_clip'])
        if not math.isfinite(loss) or not torch.isfinite(norm):
            raise FloatingPointError(f'Non-finite update at step {stream.step}')
        optimizer.step()
        for name in trained:
            trained[name] += counts[name]
        if stream.step == 1 or stream.step % 1000 == 0:
            progress = dict(step=stream.step, total_steps=stream.total_steps, loss=loss,
                learning_rate=lr, trained_tokens=trained.copy(), elapsed_seconds=time.monotonic() - started)
            write_json(output / 'progress.json', progress)
            print('TRAIN_PROGRESS=' + json.dumps(progress), flush=True)
        if stream.step % recipe['eval_interval'] == 0:
            validation()
        if stream.step % recipe['checkpoint_interval'] == 0:
            save('latest.pt', resumable=True)
            commit()
    if history[-1]['step'] != stream.step:
        validation()
    complete = stream.step == stream.total_steps
    if complete:
        for name, coverage in stream.coverage.items():
            if trained[name] != coverage['supervised_tokens']:
                raise ValueError(f'{name}: coverage target count mismatch')
        save('final.pt')
    save('latest.pt', resumable=True)
    report = dict(status='complete' if complete else 'stopped', step=stream.step,
        total_steps=stream.total_steps, trained_tokens=trained, coverage=stream.coverage,
        best_step=best_step, best_metric=best, final_metrics=history[-1])
    write_json(output / 'report.json', report)
    commit()
    return report


def commit_volume():
    if request_file := os.getenv('MODAL_CHECKPOINT_COMMIT_FILE'):
        request = Path(request_file)
        request.write_text('commit')
        deadline = time.monotonic() + 300
        while request.exists():
            if time.monotonic() > deadline:
                raise TimeoutError('Modal checkpoint commit was not acknowledged')
            time.sleep(.5)


def main():
    parser = argparse.ArgumentParser(description='Next-move training from a JSON recipe')
    parser.add_argument('--recipe', default=os.getenv('TRAIN_RECIPE'))
    parser.add_argument('--output', default=os.getenv('CKPT_DIR'))
    parser.add_argument('--sources', default=os.getenv('TRAIN_SOURCES'), help='Comma-separated subset, coverage mode only')
    parser.add_argument('--stop-after-step', type=int)
    parser.add_argument('--time-limit', type=float, default=float(os.environ['MODAL_TIMEOUT']) - 600 if 'MODAL_TIMEOUT' in os.environ else None)
    args = parser.parse_args()
    if not args.recipe or not args.output:
        parser.error('--recipe and --output (or TRAIN_RECIPE and CKPT_DIR) are required')
    settings = json.loads(Path(args.recipe).read_text())
    if args.sources:
        if settings.get('sampling', {'mode': 'coverage'})['mode'] != 'coverage':
            parser.error('--sources applies only to coverage recipes')
        names = args.sources.split(',')
        if not set(names) <= settings['datasets'].keys():
            parser.error('--sources contains an unknown dataset')
        settings['datasets'] = {name: settings['datasets'][name] for name in names}
    deadline = time.time() + args.time_limit if args.time_limit is not None else None
    print(json.dumps(train(settings, args.output, args.stop_after_step, deadline, commit_volume), indent=2))


if __name__ == '__main__':
    main()
