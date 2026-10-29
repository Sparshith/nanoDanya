import json
import os
import modal

app = modal.App("nanodanya-train")

# A100 now requires a payment method on file (credits alone no longer unlock it).
# Set MODAL_GPU (e.g. A10G, L4, L40S) to run on a credit-eligible GPU.
GPU = os.environ.get("MODAL_GPU", "A100")
torch_version, cuda_wheels = ("2.7.1+cu128", "cu128") if GPU == "B200" else ("2.5.1+cu124", "cu124")

ignore = modal.FilePatternMatcher.from_file(".modalignore")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ca-certificates", "curl")
    .run_commands(
        "curl -L --fail -o /usr/local/bin/fairy-stockfish "
        "https://github.com/fairy-stockfish/Fairy-Stockfish/releases/download/fairy_sf_14/fairy-stockfish-largeboard_x86-64",
        "echo '41b8b4d539adfd9924929ee4a948d1a37dd1e9beaa535a811cb5e7fee9e4cb99  /usr/local/bin/fairy-stockfish' | sha256sum -c",
        "chmod +x /usr/local/bin/fairy-stockfish",
    )
    .pip_install(
        "numpy",
        f"torch=={torch_version}",
        "python-chess",
        extra_index_url=f"https://download.pytorch.org/whl/{cuda_wheels}",
    )
    .pip_install("requests==2.32.5")
    .pip_install("onnx==1.23.0", "onnxruntime==1.23.2")
    .env({"PYTHONPATH": "/root/project:/root/project/nanochat", "PYTHONUNBUFFERED": "1"})
    .add_local_dir(os.getenv("MODAL_PROJECT_DIR", "."), "/root/project", ignore=ignore)
)

volume = modal.Volume.from_name("nanodanya-data")
budgeted = os.getenv("MODAL_BUDGETED", "0") == "1"
cpu_count = float(os.environ.get("MODAL_CPU", "1"))
memory_mib = int(os.environ.get("MODAL_MEMORY", "8192"))


def parse_env_overrides(raw: str) -> dict[str, str]:
    raw = raw.strip()
    if not raw:
        return {}
    if raw.startswith("{"):
        parsed = json.loads(raw)
        return {str(k): str(v) for k, v in parsed.items()}
    pairs = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        key, value = item.split("=", 1)
        pairs[key.strip()] = value.strip()
    return pairs


@app.function(
    image=image,
    gpu=None if GPU.lower() == "none" else GPU,
    cpu=(cpu_count, cpu_count) if budgeted else cpu_count,
    memory=(memory_mib, memory_mib) if budgeted else memory_mib,
    timeout=int(os.environ.get("MODAL_TIMEOUT", str(60 * 60 * 12))),
    **({"max_containers": 1, "retries": 0, "scaledown_window": 2,
        "startup_timeout": 120} if budgeted else {}),
    volumes={"/data": volume},
)
def train(datasets: str, script: str, env_overrides: str = "", resources: dict | None = None):
    import os
    import subprocess
    import shutil
    import time
    from pathlib import Path

    started = time.monotonic()
    volume.reload()
    project_root = Path("/root/project")
    volume_data = Path("/data")

    for dataset in filter(None, datasets.split(",")):
        local_data = project_root / "data" / dataset
        local_data.parent.mkdir(parents=True, exist_ok=True)
        if local_data.exists():
            if local_data.is_symlink():
                local_data.unlink()
            elif local_data.is_dir():
                shutil.rmtree(local_data)
            else:
                raise FileExistsError(f"{local_data} already exists and is not a symlink")
        local_data.symlink_to(volume_data / dataset, target_is_directory=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(project_root) + ":" + str(project_root / "nanochat")
    env["NANODANYA_MODAL_TRAIN"] = "1"
    env.update(parse_env_overrides(env_overrides))
    if env_overrides:
        print(f"Using env overrides: {env_overrides}")
    try:
        command = ["python", script]
        if request_file := env.get("MODAL_CHECKPOINT_COMMIT_FILE"):
            request = Path(request_file)
            request.unlink(missing_ok=True)
            with subprocess.Popen(command, cwd=project_root, env=env) as process:
                try:
                    while process.poll() is None:
                        if request.exists():
                            volume.commit()
                            request.unlink()
                        time.sleep(0.5)
                    if process.returncode:
                        raise subprocess.CalledProcessError(process.returncode, command)
                except BaseException:
                    process.terminate()
                    raise
        else:
            subprocess.run(command, check=True, cwd=project_root, env=env)
    finally:
        volume.commit()
        print("RESOURCE_RECEIPT=" + json.dumps({"seconds": time.monotonic() - started,
              **(resources or {})}), flush=True)
    print("Script complete, volume changes committed")


@app.local_entrypoint()
def main(datasets: str = "processed", script: str = "training/train.py", gpu: str = "", env_overrides: str = "",
         recipe: str = "", checkpoint_dir: str = "", sources: str = ""):
    if recipe:
        from pathlib import Path
        from training.recipe import resolve_recipe

        settings = resolve_recipe(json.loads(Path(recipe).read_text()))
        if not checkpoint_dir:
            raise ValueError('--checkpoint-dir is required with --recipe')
        if sources and (settings['sampling']['mode'] != 'coverage' or
                        not set(sources.split(',')) <= settings['datasets'].keys()):
            raise ValueError('--sources must name datasets in a coverage recipe')
        if not all(Path(folder).is_relative_to('/data') for folder in settings['datasets'].values()):
            raise ValueError('Modal recipes must use absolute dataset paths on /data')
        if not Path(checkpoint_dir).is_relative_to('/data'):
            raise ValueError('--checkpoint-dir must be on /data')
        recipe = str(Path(recipe).resolve().relative_to(Path(__file__).resolve().parent))
        env = parse_env_overrides(env_overrides)
        env.update(TRAIN_RECIPE=recipe, CKPT_DIR=checkpoint_dir,
                   MODAL_CHECKPOINT_COMMIT_FILE=checkpoint_dir + '/commit-request')
        env.setdefault('TORCHINDUCTOR_CACHE_DIR',
                       f'/data/compile-cache/torch-{torch_version.replace("+", "-")}-{GPU}')
        env.setdefault('MODAL_TIMEOUT', os.getenv('MODAL_TIMEOUT', str(60 * 60 * 12)))
        if sources:
            env['TRAIN_SOURCES'] = sources
        env_overrides = json.dumps(env)
        datasets = ''
        script = 'training/train.py'
    print(f"GPU: {GPU} (set via MODAL_GPU env var)")
    resources = {"gpu": GPU, "cpu_limit": cpu_count if budgeted else None,
                 "memory_mib": memory_mib}
    if budgeted:
        call = train.spawn(datasets, script, env_overrides, resources)
        try:
            # A wall-clock limit also covers automatic retries after preemption.
            call.get(timeout=int(os.environ["MODAL_TIMEOUT"]) + 120)
        except BaseException:
            call.cancel(terminate_containers=True)
            raise
    else:
        train.remote(datasets, script, env_overrides, resources)
