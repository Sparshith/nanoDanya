import json
import modal

app = modal.App("nanodanya-train")

ignore = modal.FilePatternMatcher.from_file(".modalignore")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy",
        "torch==2.5.1+cu124",
        "tokenizers",
        "tiktoken",
        "python-chess",
        "pyarrow",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .add_local_dir(".", "/root/project", ignore=ignore)
)

volume = modal.Volume.from_name("nanodanya-data")


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
    gpu="A100",
    timeout=60 * 60 * 12,
    volumes={"/data": volume},
)
def train(datasets: str, script: str, env_overrides: str = ""):
    import os
    import subprocess
    import shutil
    from pathlib import Path

    project_root = Path("/root/project")
    volume_data = Path("/data")

    for dataset in datasets.split(","):
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
    env["PYTHONPATH"] = str(project_root / "nanochat")
    env["NANODANYA_MODAL_TRAIN"] = "1"
    env.update(parse_env_overrides(env_overrides))
    if env_overrides:
        print(f"Using env overrides: {env_overrides}")
    subprocess.run(
        ["python", script],
        check=True,
        cwd=project_root,
        env=env,
    )

    volume.commit()
    print("Training complete, checkpoint saved to volume")


@app.local_entrypoint()
def main(datasets: str = "processed", script: str = "training/train.py", gpu: str = "A100", env_overrides: str = ""):
    if gpu != "A100":
        print("Ignoring gpu override; training is pinned to A100 in @app.function.")
    train.remote(datasets, script, env_overrides)
