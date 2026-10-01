"""ONNX-exportable student architecture used by research export and evaluation."""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 14750
    sequence_len: int = 512
    n_layer: int = 9
    n_head: int = 5
    n_embd: int = 320


def norm(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1.1920928955078125e-07)


def fake_quant_int4(weight, group_size=64):
    # Rounds weights to 4 bits in the forward pass but lets gradients through, so the
    # model learns to live with the rounding that int4 export applies. On by default in Model.
    groups = weight.reshape(weight.size(0), -1, group_size)
    scale = groups.detach().abs().amax(dim=-1, keepdim=True) / 7
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    rounded = (groups / scale).round().clamp(-7, 7) * scale
    return (groups + (rounded - groups).detach()).reshape_as(weight)


def rotary(x, cos, sin):
    half = x.size(-1) // 2
    a, b = x[..., :half], x[..., half:]
    return torch.cat((a * cos + b * sin, b * cos - a * sin), dim=-1)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head, self.head_dim = config.n_head, config.n_embd // config.n_head
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, cos, sin, linear):
        batch, tokens, dim = x.shape
        shape = (batch, tokens, self.n_head, self.head_dim)
        q = norm(rotary(linear(x, self.c_q).view(shape), cos, sin)).transpose(1, 2)
        k = norm(rotary(linear(x, self.c_k).view(shape), cos, sin)).transpose(1, 2)
        v = linear(x, self.c_v).view(shape).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        mask = torch.ones(tokens, tokens, dtype=torch.bool, device=x.device).tril()
        y = scores.masked_fill(~mask, float("-inf")).softmax(-1) @ v
        return linear(y.transpose(1, 2).reshape(batch, tokens, dim), self.c_proj)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x, linear):
        return linear(F.relu(linear(x, self.c_fc)).square(), self.c_proj)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn, self.mlp = Attention(config), MLP(config)

    def forward(self, x, cos, sin, linear):
        x = x + self.attn(norm(x), cos, sin, linear)
        return x + self.mlp(norm(x), linear)


class Model(nn.Module):
    def __init__(self, config=Config(), qat=True):
        super().__init__()
        self.config, self.qat = config, qat
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList(Block(config) for _ in range(config.n_layer)),
        })
        head_dim = config.n_embd // config.n_head
        positions = torch.arange(config.sequence_len, dtype=torch.float32)
        frequencies = torch.outer(positions, 10000 ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("cos", frequencies.cos().bfloat16().float()[None, :, None, :], persistent=False)
        self.register_buffer("sin", frequencies.sin().bfloat16().float()[None, :, None, :], persistent=False)
        for name, parameter in self.named_parameters():
            nn.init.normal_(parameter, std=0.02 if "wte" in name else 1 / math.sqrt(parameter.size(1)))
            if name.endswith("c_proj.weight"):
                parameter.data.div_(math.sqrt(2 * config.n_layer))

    def linear(self, x, layer):
        weight = fake_quant_int4(layer.weight) if self.qat and self.training else layer.weight
        return F.linear(x, weight)

    def forward(self, input_ids):
        tokens = input_ids.size(1)
        embedding = self.transformer.wte.weight
        if self.qat and self.training:
            embedding = fake_quant_int4(embedding)
        x = norm(F.embedding(input_ids, embedding))
        cos, sin = self.cos[:, :tokens], self.sin[:, :tokens]
        for block in self.transformer.h:
            x = block(x, cos, sin, self.linear)
        return 15 * torch.tanh(F.linear(norm(x), embedding) / 15)


def unpack_int4(packed, scales):
    # packed: uint8 [rows, cols/2], two 4-bit codes per byte (low nibble first). scales: float32 [rows, cols/64].
    rows = packed.size(0)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(rows, -1).to(torch.int8)
    codes = torch.where(codes >= 8, codes - 16, codes).float()
    return (codes.view(rows, scales.size(1), 64) * scales[:, :, None]).reshape(rows, -1)


def load_weights(path):
    """Returns (config, state_dict, saved int4 scales or None) from a starter weights file or a plain state_dict."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if "packed" in saved:
        state = {name: unpack_int4(saved["packed"][name], saved["scales"][name]) for name in saved["packed"]}
        return Config(**saved["config"]), state, saved["scales"]
    if "model" in saved and "meta" in saved:
        state = {key.removeprefix("_orig_mod."): value for key, value in saved["model"].items()}
        config = {key: saved["meta"]["model_config"][key] for key in Config.__dataclass_fields__ if key in saved["meta"]["model_config"]}
        return Config(**config), state, None
    return None, saved, None
