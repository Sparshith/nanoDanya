import modal

app = modal.App("nanochess")

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
    .add_local_dir(".", "/root/project")
)

volume = modal.Volume.from_name("nanochess-data")


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 6,
    volumes={"/data": volume},
)
def train():
    import os
    import subprocess
    import shutil
    from pathlib import Path

    project_root = Path("/root/project")
    volume_data = Path("/data")
    local_data = project_root / "data" / "processed"

    local_data.parent.mkdir(parents=True, exist_ok=True)
    if local_data.exists():
        if local_data.is_symlink():
            local_data.unlink()
        elif local_data.is_dir():
            shutil.rmtree(local_data)
        else:
            raise FileExistsError(f"{local_data} already exists and is not a symlink")
    local_data.symlink_to(volume_data, target_is_directory=True)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    subprocess.run(
        ["python", "Train.py"],
        check=True,
        cwd=project_root,
        env=env,
    )

    volume.commit()
    print("Training complete, checkpoint saved to volume")


@app.local_entrypoint()
def main():
    train.remote()
