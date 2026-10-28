import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def fake_quant_int4(weight, group_size=64):
    if weight.size(1) % group_size:
        raise ValueError("weight width must be divisible by INT4 group size")
    groups = weight.reshape(weight.size(0), -1, group_size)
    scale = groups.detach().abs().amax(dim=-1, keepdim=True) / 7
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    rounded = (groups / scale).round().clamp(-7, 7) * scale
    return (groups + (rounded - groups).detach()).reshape_as(weight)


def linear(x, layer, qat):
    weight = fake_quant_int4(layer.weight) if qat else layer.weight
    return F.linear(x, weight)


def rotary(x, cos, sin):
    half = x.size(-1) // 2
    a, b = x[..., :half], x[..., half:]
    return torch.cat((a * cos + b * sin, b * cos - a * sin), dim=-1).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=False)
        self.c_k = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(config.n_embd, config.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x, cos, sin, qat):
        batch, length, _ = x.shape
        q = linear(x, self.c_q, qat).view(batch, length, self.n_head, self.head_dim)
        k = linear(x, self.c_k, qat).view(batch, length, self.n_kv_head, self.head_dim)
        v = linear(x, self.c_v, qat).view(batch, length, self.n_kv_head, self.head_dim)
        q = norm(rotary(q, cos, sin)).transpose(1, 2)
        k = norm(rotary(k, cos, sin)).transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, enable_gqa=self.n_head != self.n_kv_head
        )
        y = y.transpose(1, 2).contiguous().view(batch, length, -1)
        return linear(y, self.c_proj, qat)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x, qat):
        return linear(F.relu(linear(x, self.c_fc, qat)).square(), self.c_proj, qat)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = Attention(config)
        self.mlp = MLP(config)

    def forward(self, x, cos, sin, qat):
        x = x + self.attn(norm(x), cos, sin, qat)
        return x + self.mlp(norm(x), qat)


class StudentGPT(nn.Module):
    def __init__(self, config, qat=False):
        super().__init__()
        if config.n_embd % config.n_head or config.n_head % config.n_kv_head:
            raise ValueError("invalid attention head configuration")
        if (config.n_embd // config.n_head) % 2:
            raise ValueError("RoPE head dimension must be even")
        if config.n_embd % 64 or (4 * config.n_embd) % 64:
            raise ValueError("INT4 export requires matrix widths divisible by 64")
        self.config = config
        self.qat = qat
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        })
        positions = torch.arange(config.sequence_len * 10, dtype=torch.float32)
        channels = torch.arange(0, config.n_embd // config.n_head, 2, dtype=torch.float32)
        frequencies = torch.outer(positions, 10000 ** (-channels / (config.n_embd // config.n_head)))
        self.register_buffer("cos", frequencies.cos().to(torch.bfloat16)[None, :, None, :], persistent=False)
        self.register_buffer("sin", frequencies.sin().to(torch.bfloat16)[None, :, None, :], persistent=False)
        self.apply(self._init_weights)
        for block in self.transformer.h:
            block.attn.c_proj.weight.data.div_(math.sqrt(2 * config.n_layer))
            block.mlp.c_proj.weight.data.div_(math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=1 / math.sqrt(module.weight.size(1)))

    def forward(self, idx):
        length = idx.size(1)
        if length > self.cos.size(1):
            raise ValueError("sequence exceeds RoPE cache")
        embedding = self.transformer.wte.weight
        if self.qat:
            embedding = fake_quant_int4(embedding)
        x = norm(F.embedding(idx, embedding))
        cos, sin = self.cos[:, :length], self.sin[:, :length]
        for block in self.transformer.h:
            x = block(x, cos, sin, self.qat)
        logits = F.linear(norm(x), embedding)
        return 15 * torch.tanh(logits / 15)
