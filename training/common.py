import torch

from nanochat.gpt import norm

SOFTCAP = 15


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    return device


def load_model_state(model, state):
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state["model"].items()})


def clean_state_dict(module):
    return {k.replace("_orig_mod.", ""): v for k, v in module.state_dict().items()}


def build_checkpoint(model, config, meta, step, **extra):
    ckpt = {
        "model": clean_state_dict(model),
        "meta": {"model_config": config.__dict__, "tokenizer": meta},
        "step": step,
    }
    ckpt.update(extra)
    return ckpt


def forward_hidden(model, xb, input_additions=None, kv_cache=None):
    T = xb.size(1)
    start = 0 if kv_cache is None else kv_cache.get_pos()
    cos_sin = model.cos[:, start:start + T], model.sin[:, start:start + T]
    x = model.transformer.wte(xb)
    if input_additions is not None:
        x = x + input_additions
    x = norm(x)
    for block in model.transformer.h:
        x = block(x, cos_sin, kv_cache)
    return norm(x)


def move_logits(model, x):
    logits = model.lm_head(x)
    return (SOFTCAP * torch.tanh(logits / SOFTCAP)).float()
