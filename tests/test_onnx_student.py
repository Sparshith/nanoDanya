import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnxruntime
import torch

from inference.onnx_policy import load_torch_from_onnx
from inference.onnx_student import Config, Model


def test_research_export_and_reload_work_without_the_challenge_starter(tmp_path):
    config = Config(vocab_size=128, sequence_len=32, n_layer=1, n_head=1, n_embd=64)
    model = Model(config, qat=False).eval()
    vocabulary = ['<bos>', '<eos>', *[f'token-{i}' for i in range(126)]]
    checkpoint = tmp_path / 'student.pt'
    torch.save({'model': model.state_dict(), 'meta': {
        'model_config': asdict(config), 'tokenizer': {'itos': dict(enumerate(vocabulary))},
    }}, checkpoint)
    artifact = tmp_path / 'student.onnx'
    subprocess.run([sys.executable, '-m', 'inference.export_onnx', '--checkpoint', str(checkpoint),
                    '--output', str(artifact), '--int4'], cwd=Path(__file__).resolve().parents[1],
                   check=True, capture_output=True)
    loaded, stoi, itos = load_torch_from_onnx(str(artifact))
    assert itos == vocabulary and stoi == dict(zip(vocabulary, range(len(vocabulary))))
    session = onnxruntime.InferenceSession(str(artifact), providers=['CPUExecutionProvider'])
    assert json.loads(session.get_modelmeta().custom_metadata_map['vocabulary']) == vocabulary
    for length in (1, 8, 24):
        tokens = torch.randint(config.vocab_size, (1, length))
        with torch.no_grad():
            expected = loaded(tokens)[0, -1].numpy()
        actual = session.run(None, {'input_ids': tokens.numpy()})[0].reshape(-1)
        np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=1e-4)
