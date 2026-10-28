"""Compare independently fitted, frozen-model identity-location readouts."""
import json
from pathlib import Path

import modal

from modal_board_benchmark import image

image = image.add_local_file("modal_board_benchmark.py", "/root/project/modal_board_benchmark.py")
app = modal.App("nanodanya-board-readouts")
volume = modal.Volume.from_name("nanodanya-data")
SOURCE = "/data/experiments/involved-pieces-20260921"
RUN = "/data/experiments/board-input-20260922"
MODELS = {
    "champion": ("/data/checkpoints/plain/games-15m/l16_best.pt",
                 "7e72148ed260564d671a67f5e0bc9bc4d82fd11695a34d9f4708858033955b0e"),
    "control": (f"{RUN}/control/final.pt", "d30440939a8e64634169cefecc6f7254b3e646c893b6b1f2f378fd6a0d9402e8"),
    "oracle": (f"{RUN}/oracle/final.pt", "b07f4d6ac0a7330faad2937954e9dee6f4dca4a0251191cda073196da34828be"),
}


@app.function(image=image, gpu="H100", cpu=8, memory=32768, max_containers=3,
              volumes={"/data": volume}, timeout=7200)
def fit(arm, full_board=False, square_board=False):
    import os
    import sys
    import time
    import torch

    os.chdir("/root/project")
    sys.path[:0] = ["/root/project", "/root/project/nanochat"]
    from data_prep.prepare_involved_pieces import file_hash
    from probes.board_input_readout import run_readout

    volume.reload()
    checkpoint, digest = MODELS[arm]
    assert file_hash(checkpoint) == digest
    assert not (full_board and square_board)
    output = Path(RUN) / ("square-readouts" if square_board else "full-board-readouts" if full_board else "readouts") / arm
    if (output / "result.json").exists():
        result = json.loads((output / "result.json").read_text())
        assert result["protocol"]["checkpoint_sha256"] == digest and result["protocol"]["steps"] == 3000
        assert result["protocol"]["version"] == (3 if square_board else 2 if full_board else 1)
        return {"arm": arm, **result}
    torch.set_num_threads(8)
    started = time.monotonic()
    try:
        result = run_readout(checkpoint, f"{SOURCE}/dataset", f"{RUN}/boards",
                             (Path(SOURCE) / "game.pgn").read_text(), output, full_board=full_board,
                             square_board=square_board)
        result["seconds"] = time.monotonic() - started
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        return {"arm": arm, **result}
    finally:
        volume.commit()


@app.function(image=image, cpu=0.25, memory=2048, timeout=14400)
def pipeline(full_board=False):
    return {row["arm"]: row for row in fit.starmap([(arm, full_board) for arm in MODELS], order_outputs=False)}


@app.local_entrypoint()
def main(full_board: bool = False, square_board: bool = False, arm: str = ""):
    assert not (full_board and square_board)
    if square_board or arm:
        arm = arm or "oracle"
        assert arm in MODELS
        result = {arm: fit.remote(arm, full_board, square_board)}
    else:
        result = pipeline.remote(full_board)
    output = Path("data/probes/board-input-20260922")
    output.mkdir(parents=True, exist_ok=True)
    name = "square-readout-result.json" if square_board else "full-board-readout-result.json" if full_board else "readout-result.json"
    (output / name).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({arm: {"probe": r["probe"]["aggregate"], "trace": r["trace"]["aggregate"]}
                      for arm, r in result.items()}, indent=2))
