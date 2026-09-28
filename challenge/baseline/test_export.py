import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime
import pytest
import torch

STARTER = Path(__file__).resolve().parents[1] / "starter"
SERVED = Path(__file__).resolve().parents[2] / "dashboard/assets/baseline.onnx"
sys.path.insert(0, str(STARTER))
from model import Model, load_weights  # noqa: E402


@pytest.mark.skipif(not (STARTER / "baseline.pt").exists(), reason="baseline weights not on this machine")
def test_int4_export_matches_pytorch_and_served_file(tmp_path):
    output = tmp_path / "baseline.onnx"
    subprocess.run([sys.executable, str(STARTER / "export.py"), "--int4", "--output", str(output)], check=True, capture_output=True)
    assert output.stat().st_size <= 10_000_000
    session = onnxruntime.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    props = session.get_modelmeta().custom_metadata_map
    vocabulary = json.loads(props["vocabulary"])
    assert vocabulary[0] == "<bos>" and props["context_length"] == "512"
    config, state, _ = load_weights(STARTER / "baseline.pt")
    model = Model(config, qat=False).eval()
    model.load_state_dict(state)
    served = onnxruntime.InferenceSession(str(SERVED), providers=["CPUExecutionProvider"]) if SERVED.exists() else None
    rng = np.random.default_rng(3)
    for length in (1, 8, 64):
        tokens = np.concatenate([[0], rng.integers(2, len(vocabulary), size=length - 1)]).astype(np.int64)[None]
        with torch.no_grad():
            expected = model(torch.from_numpy(tokens))[0, -1].numpy()
        actual = session.run(None, {"input_ids": tokens})[0].reshape(-1)
        assert actual.shape == (len(vocabulary),)
        np.testing.assert_allclose(actual, expected, atol=2e-3, rtol=1e-3)
        if served:
            np.testing.assert_allclose(served.run(None, {"input_ids": tokens})[0].reshape(-1), actual, atol=1e-4)
