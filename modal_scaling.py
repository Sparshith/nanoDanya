"""Budget-bounded, sequential H100 scaling benchmark on the selected Modal profile."""
import json
from pathlib import Path
import statistics
import time

import modal
from modal_train import base_image, image, ignore, volume

app = modal.App('nanodanya-scaling')


@app.function(image=base_image.pip_install('pytest==9.0.3').add_local_dir('.', '/root/project', ignore=ignore), cpu=(4, 4), memory=(8192, 8192),
              timeout=240, retries=0, max_containers=1, scaledown_window=2)
def verify():
    import subprocess
    subprocess.run(['python', '-m', 'pytest', 'tests/pretraining', '-q'],
                   cwd='/root/project', check=True, timeout=210)


@app.cls(image=image, volumes={'/data': volume}, gpu='H100!', cpu=(32, 32),
              memory=(65536, 65536), timeout=900, retries=0, max_containers=1,
              scaledown_window=2, startup_timeout=120)
class Benchmark:
    @modal.method()
    def run(self, count: int, run_id: str, source: str, operational: bool = False):
        import os
        import subprocess
        import signal
        import shutil

        started = time.monotonic()
        volume.reload()
        output = f'/data/benchmarks/scaling/{run_id}/{count}.json'
        env = dict(os.environ, OMP_NUM_THREADS='2',
                   TORCHINDUCTOR_CACHE_DIR='/data/compile-cache/scaling-torch-2.5.1-h100')
        command = ['python', '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={count}',
                   '-m', 'training.benchmark_scaling', '--recipe', 'training/recipes/l16-all-once.json',
                   '--source', source, '--output', output]
        request = Path(f'/tmp/scaling-commit-{count}')
        if operational:
            command.append('--operational')
            request.unlink(missing_ok=True)
            env['MODAL_CHECKPOINT_COMMIT_FILE'] = str(request)
        telemetry = Path(f'/tmp/gpu-{count}.csv')
        with telemetry.open('w') as log:
            monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,index,utilization.gpu,memory.used',
                                        '--format=csv,nounits', '-l', '1'], stdout=log)
            try:
                if operational:
                    with subprocess.Popen(command, cwd='/root/project', env=env, start_new_session=True) as process:
                        try:
                            deadline = time.monotonic() + 350
                            while process.poll() is None:
                                if request.exists():
                                    volume.commit()
                                    request.unlink()
                                if time.monotonic() >= deadline:
                                    raise subprocess.TimeoutExpired(command, 350)
                                time.sleep(.25)
                            if process.returncode:
                                raise subprocess.CalledProcessError(process.returncode, command)
                        finally:
                            if process.poll() is None:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait()
                else:
                    subprocess.run(command, cwd='/root/project', env=env, check=True, timeout=800)
            finally:
                monitor.terminate()
                monitor.wait()
                if operational:
                    for temporary in Path(output).parent.glob('operational-*'):
                        shutil.rmtree(temporary)
                    volume.commit()
        result = json.loads(Path(output).read_text())
        result['container_seconds'] = time.monotonic() - started
        Path(output).write_text(json.dumps(result, indent=2) + '\n')
        Path(output).with_suffix('.csv').write_text(telemetry.read_text())
        volume.commit()
        return result

@app.local_entrypoint()
def main(source: str = 'human', counts: str = '1,2,4,8', budget: float = 0, operational: bool = False):
    if budget <= 0:
        raise ValueError('An explicit positive --budget is required')
    prices = json.loads(Path('ops/modal-pricing.json').read_text())
    sizes = [int(value) for value in counts.split(',')]
    if not sizes or any(n not in (1, 2, 4, 8) for n in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError('counts must be distinct values from 1,2,4,8')
    rates = {n: n * prices['gpu_per_second']['H100'] + 32 * prices['cpu_physical_core_per_second']
             + 64 * prices['memory_gib_per_second'] for n in sizes}
    # Reserve startup, timeout, shutdown, and a separate margin for image build costs.
    ceiling = sum(rates.values()) * (600 if operational else 1100) + 3
    if ceiling > budget:
        raise ValueError(f'Runtime caps reserve ${ceiling:.2f}, exceeding ${budget:.2f}')
    run_id = time.strftime('%Y%m%d-%H%M%S')
    folder = Path('benchmark/results') / f'scaling-{run_id}'
    folder.mkdir(parents=True)
    (folder / 'budget.json').write_text(json.dumps(dict(budget=budget, reserved=ceiling,
        prices=prices, counts=sizes, source=source, operational=operational), indent=2))
    print(f'Runtime reservation: ${ceiling:.2f}; outputs: {folder}', flush=True)
    call = sweep.spawn(sizes, run_id, source, rates, operational)
    (folder / 'call.json').write_text(json.dumps(dict(call_id=call.object_id, run_id=run_id)))
    print(f'Sweep call: {call.object_id}; volume: benchmarks/scaling/{run_id}', flush=True)


@app.function(image=image, volumes={'/data': volume}, cpu=(1, 1), memory=(1024, 1024),
              timeout=6000, retries=0, max_containers=1, scaledown_window=2, nonpreemptible=True)
def sweep(sizes, run_id, source, rates, operational):
    verify.remote()
    volume.reload()
    folder = Path('/data/benchmarks/scaling') / run_id
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    for size in sizes:
        call = Benchmark.with_options(gpu=f'H100!:{size}', timeout=400 if operational else 900)().run.spawn(size, run_id, source, operational)
        try:
            result = call.get(timeout=520 if operational else 1020)
        except BaseException:
            call.cancel(terminate_containers=True)
            raise
        volume.reload()
        result['estimated_compute_usd'] = result['container_seconds'] * rates[size]
        timing_key = 'seconds_per_update_including_overhead' if operational else 'median_seconds_per_update'
        result[timing_key] = (result['container_seconds'] / result['updates'] if operational
            else statistics.median(r['seconds_per_update'] for r in result['repeats']))
        results.append(result)
        if sizes[0] == 1:
            result['speedup'] = results[0][timing_key] / result[timing_key]
            result['gpu_hour_ratio'] = size / result['speedup']
        (folder / f'{size}.json').write_text(json.dumps(result, indent=2))
        volume.commit()
        print(json.dumps({k: result[k] for k in ['world_size', timing_key,
                                               'estimated_compute_usd', 'speedup', 'gpu_hour_ratio'] if k in result}), flush=True)
    (folder / 'summary.json').write_text(json.dumps(results, indent=2))
    volume.commit()
    return results
