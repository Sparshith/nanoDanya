from __future__ import annotations

import torch

from model_registry import resolve_model_ref
from nanochat.gpt import GPT, GPTConfig


def load_model(path: str, device: str):
    model_path, _ = resolve_model_ref(path)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    config = GPTConfig(**ckpt["meta"]["model_config"])
    model = GPT(config).eval()
    model.load_state_dict(ckpt["model"])
    stoi = ckpt["meta"]["tokenizer"]["stoi"]
    itos = ckpt["meta"]["tokenizer"]["itos"]
    model = model.to(device)
    return model, config, stoi, itos
