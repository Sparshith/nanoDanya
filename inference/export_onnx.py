"""Export a research checkpoint to standalone ONNX with optional INT8 or INT4 weights."""
import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

from inference.onnx_student import Config, Model, load_weights

LIMIT = 10_000_000
GROUP = 64

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=Path, default=os.environ.get("STUDENT_CHECKPOINT"),
                    help="training checkpoint or state_dict to export")
parser.add_argument("--output", type=Path, default=os.environ.get("STUDENT_ARTIFACT", "model.onnx"))
parser.add_argument("--name", default=os.environ.get("EXPORT_NAME", "nanoDanya student"))
parser.add_argument("--vocabulary", type=Path, help="JSON token list, required for weights without tokenizer metadata")
precision = parser.add_mutually_exclusive_group()
precision.add_argument("--int8", action="store_true", help="8-bit matrix weights, about 4x smaller than fp32")
precision.add_argument("--int4", action="store_true", help="4-bit matrix weights in groups of 64, about 8x smaller")
args = parser.parse_args()
if os.environ.get("EXPORT_PRECISION") == "int4":
    args.int4 = True

if not args.checkpoint:
    parser.error("--checkpoint or STUDENT_CHECKPOINT is required")
weights = args.checkpoint
saved = torch.load(weights, map_location="cpu", weights_only=False)
if args.vocabulary:
    vocabulary = json.loads(args.vocabulary.read_text())
elif "meta" in saved and "tokenizer" in saved["meta"]:
    itos = saved["meta"]["tokenizer"]["itos"]
    vocabulary = [itos[i] if i in itos else itos[str(i)] for i in range(len(itos))] if isinstance(itos, dict) else list(itos)
else:
    parser.error("checkpoint has no tokenizer metadata; provide --vocabulary")
config, state, saved_scales = load_weights(weights)
model = Model(config or Config(), qat=False).eval()
state.pop("lm_head.weight", None)
model.load_state_dict(state)
if len(vocabulary) != model.config.vocab_size:
    raise SystemExit(f"vocabulary has {len(vocabulary):,} tokens but the model has {model.config.vocab_size:,}")


class LastToken(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        return self.model(input_ids)[:, -1]


with torch.no_grad():
    torch.onnx.export(LastToken(model), (torch.zeros(1, 8, dtype=torch.int64),), str(args.output), dynamo=False,
                      opset_version=18, do_constant_folding=args.int8, input_names=["input_ids"], output_names=["logits"],
                      dynamic_axes={"input_ids": {1: "tokens"}})

if args.int8:
    from onnxruntime.quantization import quantize_dynamic
    quantize_dynamic(str(args.output), str(args.output))


def pack_int4(matrix, scales=None):
    rows, columns = matrix.shape
    groups = matrix.reshape(rows, columns // GROUP, GROUP)
    if scales is None:
        scales = np.abs(groups).max(axis=-1) / 7
        scales[scales == 0] = 1
    codes = np.clip(np.rint(groups / scales[..., None]), -7, 7).astype(np.int8).reshape(rows, columns)
    nibbles = (codes & 15).astype(np.uint8)
    return nibbles[:, ::2] | (nibbles[:, 1::2] << 4), scales.astype(np.float32)


def int4_pass(exported):
    """Replaces every large 2-D float initializer with packed 4-bit codes, scales, and the ops that unpack them."""
    graph = exported.graph
    consts = {"nibble_mask": np.array(15, np.uint8), "nibble_shift": np.array(4, np.uint8), "axis_2": np.array([2], np.int64),
              "seven_half": np.array(7.5, np.float32), "sixteen": np.array(16, np.float32)}
    nodes, keep = [], []
    for tensor in graph.initializer:
        array = numpy_helper.to_array(tensor)
        if tensor.data_type != TensorProto.FLOAT or array.ndim != 2 or array.shape[1] % GROUP or array.size < 4096:
            keep.append(tensor)
            continue
        name = tensor.name.removeprefix("model.")
        packed, scales = pack_int4(array, saved_scales[name].numpy() if saved_scales and name in saved_scales else None)
        rows, columns = array.shape
        p, s = tensor.name + ".packed", tensor.name + ".scales"
        keep += [numpy_helper.from_array(packed, p), numpy_helper.from_array(scales[:, :, None], s),
                 numpy_helper.from_array(np.array([rows, columns // GROUP, GROUP], np.int64), tensor.name + ".groups"),
                 numpy_helper.from_array(np.array([rows, columns], np.int64), tensor.name + ".shape")]
        t = tensor.name + "."
        nodes += [
            helper.make_node("BitwiseAnd", [p, "nibble_mask"], [t + "low"]),
            helper.make_node("BitShift", [p, "nibble_shift"], [t + "high"], direction="RIGHT"),
            helper.make_node("Unsqueeze", [t + "low", "axis_2"], [t + "low3"]),
            helper.make_node("Unsqueeze", [t + "high", "axis_2"], [t + "high3"]),
            helper.make_node("Concat", [t + "low3", t + "high3"], [t + "pairs"], axis=2),
            helper.make_node("Cast", [t + "pairs"], [t + "codes"], to=TensorProto.FLOAT),
            helper.make_node("Greater", [t + "codes", "seven_half"], [t + "negative"]),
            helper.make_node("Sub", [t + "codes", "sixteen"], [t + "wrapped"]),
            helper.make_node("Where", [t + "negative", t + "wrapped", t + "codes"], [t + "signed"]),
            helper.make_node("Reshape", [t + "signed", tensor.name + ".groups"], [t + "grouped"]),
            helper.make_node("Mul", [t + "grouped", s], [t + "scaled"]),
            helper.make_node("Reshape", [t + "scaled", tensor.name + ".shape"], [tensor.name]),
        ]
    keep += [numpy_helper.from_array(value, key) for key, value in consts.items()]
    del graph.initializer[:]
    graph.initializer.extend(keep)
    old = list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes + old)
    return exported


exported = onnx.load(str(args.output))
if args.int4:
    exported = int4_pass(exported)
helper.set_model_props(exported, {"name": args.name, "vocabulary": json.dumps(vocabulary, separators=(",", ":")),
                                  "context_length": str(model.config.sequence_len), "config": json.dumps(asdict(model.config))})
onnx.checker.check_model(exported)
onnx.save(exported, str(args.output))

size = args.output.stat().st_size
print(f"{args.output.name}: {size:,} bytes ({size / 1e6:.2f} MB) {'OK' if size <= LIMIT else 'TOO BIG, try --int8 or --int4'}")
