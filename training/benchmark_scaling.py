"""Fixed-global-batch benchmark, launched with torchrun for multiple GPUs."""
import argparse
import json
from pathlib import Path
import time
import shutil
import tempfile

import torch

from nanochat.gpt import GPT, GPTConfig
from training import distributed
from training.pretrain import backward_batch, make_loss, write_json
from training.recipe import BatchStream, load_data, resolve_recipe


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    distributed.barrier()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--operational', action='store_true')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if min(args.warmup, args.steps, args.repeats) < 1:
        parser.error('warmup, steps and repeats must be positive')
    settings = json.loads(Path(args.recipe).read_text())
    settings['datasets'] = {args.source: settings['datasets'][args.source]}
    recipe = resolve_recipe(settings)
    if recipe['sampling']['mode'] != 'coverage' or recipe['model_type'] != 'gpt':
        parser.error('Benchmark requires a GPT coverage recipe')
    distributed.initialize(recipe['device'])
    try:
        if args.operational:
            from training.pretrain import _train, commit_volume
            if distributed.rank() == 0:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            distributed.barrier()
            output = distributed.broadcast(tempfile.mkdtemp(prefix='operational-',
                dir=Path(args.output).parent) if distributed.rank() == 0 else None)
            synchronize()
            started = time.perf_counter()
            report = _train(recipe, output, 20, None, commit_volume)
            synchronize()
            elapsed = time.perf_counter() - started
            if distributed.rank() == 0:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                write_json(args.output, dict(world_size=distributed.world_size(), recipe=recipe,
                    operational_seconds=elapsed, updates=report['step'], report=report,
                    torch_version=torch.__version__, gpu=torch.cuda.get_device_name()))
                shutil.rmtree(output)
        else:
            run(recipe, args)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def run(recipe, args):
    started = time.perf_counter()
    if recipe['batch_size'] % distributed.world_size():
        raise ValueError('Global batch must divide evenly across GPUs')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.manual_seed(recipe['seed'])
    data, meta, fingerprints = load_data(recipe)
    stream = BatchStream(data, meta, recipe)
    if distributed.rank() == 0:
        print(f'Data ready after {time.perf_counter() - started:.1f}s; starting model warmup', flush=True)
    if stream.total_steps < args.warmup + args.steps * args.repeats:
        raise ValueError('Dataset is too small for this benchmark')
    model = GPT(GPTConfig(**recipe['model_config']))
    model.init_weights()
    model.to(recipe['device'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['learning_rate'],
        betas=tuple(recipe['betas']), weight_decay=recipe['weight_decay'], fused=recipe['device'] == 'cuda')
    loss_fn = make_loss(model, recipe)

    def update():
        optimizer.zero_grad(set_to_none=True)
        loss, counts = backward_batch(model, stream.take(), recipe, loss_fn)
        torch.nn.utils.clip_grad_norm_(model.parameters(), recipe['grad_clip'], error_if_nonfinite=True)
        optimizer.step()
        return loss, sum(counts.values())

    for _ in range(args.warmup):
        update()
    synchronize()
    startup = time.perf_counter() - started
    if distributed.rank() == 0:
        print(f'Warmup complete after {startup:.1f}s; starting timed updates', flush=True)
    if recipe['device'] == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    results = []
    for repeat in range(args.repeats):
        synchronize()
        start = time.perf_counter()
        targets = 0
        for _ in range(args.steps):
            loss, count = update()
            targets += count
        synchronize()
        elapsed = time.perf_counter() - start
        results.append(dict(seconds=elapsed, seconds_per_update=elapsed / args.steps,
            supervised_tokens_per_second=targets / elapsed, supervised_tokens=targets,
            final_loss=loss))
        if distributed.rank() == 0:
            print('BENCHMARK_REPEAT=' + json.dumps(results[-1]), flush=True)
    peak = torch.tensor(torch.cuda.max_memory_allocated() if recipe['device'] == 'cuda' else 0,
                        device=recipe['device'])
    if distributed.world_size() > 1:
        torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
    report = dict(world_size=distributed.world_size(), recipe=recipe, fingerprints=fingerprints,
        startup_seconds=startup, warmup_steps=args.warmup, measured_steps_per_repeat=args.steps,
        optimizer_schedule='constant', optimizer_learning_rate=recipe['learning_rate'],
        ddp_graph_partitioning=False,
        peak_allocated_bytes_per_gpu=int(peak), repeats=results, torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        gpu=torch.cuda.get_device_name() if recipe['device'] == 'cuda' else 'cpu')
    if distributed.rank() == 0:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, report)


if __name__ == '__main__':
    main()
