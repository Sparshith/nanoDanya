from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from model_loading import load_model


def parse_layers(value: str, n_layer: int) -> list[int]:
    if value == "all":
        return list(range(n_layer))
    layers = [int(x) for x in value.split(",") if x.strip()]
    for layer in layers:
        if layer < 0 or layer >= n_layer:
            raise ValueError(f"Layer {layer} outside model range 0..{n_layer - 1}")
    return layers


def device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.inference_mode()
def cache_hiddens(args) -> dict:
    device = device_name()
    positions = torch.load(args.positions, map_location="cpu", weights_only=False)
    model, config, _, _ = load_model(args.model, device)
    layers = parse_layers(args.layers, config.n_layer)

    hook_out: dict[int, torch.Tensor] = {}
    hooks = []
    for layer in layers:
        def make_hook(layer_idx):
            def hook(_module, _inp, out):
                hook_out[layer_idx] = out.detach().float().cpu()
            return hook
        hooks.append(model.transformer.h[layer].register_forward_hook(make_hook(layer)))

    prefixes: list[list[int]] = positions["prefixes"]
    n = len(prefixes)
    hiddens = {
        layer: torch.empty((n, config.n_embd), dtype=torch.float32)
        for layer in layers
    }

    eos_pad = 1
    for start in range(0, n, args.batch_size):
        batch = prefixes[start:start + args.batch_size]
        lengths = torch.tensor([len(x) for x in batch], dtype=torch.long)
        max_len = int(lengths.max().item())
        x = torch.full((len(batch), max_len), eos_pad, dtype=torch.long, device=device)
        for row, prefix in enumerate(batch):
            x[row, :len(prefix)] = torch.tensor(prefix, dtype=torch.long, device=device)

        hook_out.clear()
        model(x)
        rows = torch.arange(len(batch))
        cols = lengths - 1
        for layer in layers:
            hiddens[layer][start:start + len(batch)] = hook_out[layer][rows, cols]

        done = min(start + len(batch), n)
        print(f"cached {done}/{n} positions")

    for hook in hooks:
        hook.remove()

    return {
        "model": args.model,
        "positions": str(args.positions),
        "layers": layers,
        "hiddens": hiddens,
        "n_embd": config.n_embd,
        "vocab_size": positions["vocab_size"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache model hidden states for probe positions.")
    parser.add_argument("--model", default="plain/games-500k")
    parser.add_argument("--positions", default="data/probes/positions_val_small.pt")
    parser.add_argument("--out", default="data/probes/hiddens_plain_games_500k_val_small.pt")
    parser.add_argument("--layers", default="0,3,6,9,11")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    out = cache_hiddens(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
