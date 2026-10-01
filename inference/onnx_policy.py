"""Run an ONNX policy offline with the same interface the benchmarks use for PyTorch models."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import onnxruntime
import torch


class OnnxPolicy:
    def __init__(self, path: str, device: str = "cpu"):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.startswith("cuda") else ["CPUExecutionProvider"]
        self.session = onnxruntime.InferenceSession(path, providers=providers)
        props = self.session.get_modelmeta().custom_metadata_map
        self.itos = json.loads(props["vocabulary"])
        self.stoi = {token: index for index, token in enumerate(self.itos)}
        self.name = props.get("name", path)
        self.config = SimpleNamespace(sequence_len=int(props.get("context_length", 512)), vocab_size=len(self.itos))
        self.input_name = self.session.get_inputs()[0].name

    def last_logits(self, prefix: list[int]) -> torch.Tensor:
        tokens = np.asarray(prefix, dtype=np.int64)[None]
        logits = self.session.run(None, {self.input_name: tokens})[0]
        return torch.from_numpy(np.ascontiguousarray(logits.reshape(-1, logits.shape[-1])[-1]))


def load_torch_from_onnx(path: str, device: str = "cpu"):
    """Rebuild a student INT4 export from its packed weights and config metadata."""
    import onnx
    from onnx import numpy_helper

    from inference.onnx_student import Config, Model, unpack_int4

    proto = onnx.load(path)
    props = {entry.key: entry.value for entry in proto.metadata_props}
    tokens = json.loads(props["vocabulary"])
    tensors = {tensor.name: torch.from_numpy(numpy_helper.to_array(tensor).copy()) for tensor in proto.graph.initializer}
    state = {}
    for name in [name[:-len(".packed")] for name in tensors if name.endswith(".packed")]:
        scales = tensors[name + ".scales"][:, :, 0]
        state[name.removeprefix("model.")] = unpack_int4(tensors[name + ".packed"], scales)
    config = Config(**json.loads(props["config"]))
    model = Model(config, qat=False)
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model, {token: index for index, token in enumerate(tokens)}, tokens
