from __future__ import annotations

import torch

from model_registry import resolve_model_ref
from nanochat.gpt import GPT, GPTConfig


def load_model(path: str, device: str):
    model_path, _ = resolve_model_ref(path)
    if model_path.endswith(".onnx"):
        from inference.onnx_policy import load_torch_from_onnx

        model, stoi, itos = load_torch_from_onnx(model_path, device)
        return model, model.config, stoi, itos
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    config = GPTConfig(**ckpt["meta"]["model_config"])
    if ckpt["meta"].get("model_family") == "student_tied":
        from training.student import StudentGPT

        model = StudentGPT(config, qat=ckpt["meta"].get("qat", False)).eval()
    else:
        model = GPT(config).eval()
    model.load_state_dict(ckpt["model"])
    stoi = ckpt["meta"]["tokenizer"]["stoi"]
    itos = ckpt["meta"]["tokenizer"]["itos"]
    if "board_input" in ckpt:
        from inference.board_input import BoardInputPlayer
        from training.board_input import BoardInputModel

        board_input = ckpt["board_input"]
        if board_input["version"] != 1:
            raise ValueError("Unsupported board-input checkpoint version")
        model = BoardInputModel(model, board_input["enabled"], board_input["piece_dim"])
        model.encoder.load_state_dict(board_input["encoder"])
        model = BoardInputPlayer(model, itos).eval()
    model = model.to(device)
    return model, config, stoi, itos
