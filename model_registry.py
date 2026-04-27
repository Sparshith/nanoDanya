from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class ModelSpec:
    aliases: tuple[str, ...]
    path: str
    family: str
    lineage: str
    role: str
    objective: str
    dataset: str
    size: str
    tags: tuple[str, ...] = ()
    notes: str = ""

    @property
    def primary_alias(self) -> str:
        return self.aliases[0]

    @property
    def filename(self) -> str:
        return Path(self.path).name

    @property
    def absolute_path(self) -> Path:
        return PROJECT_ROOT / self.path


MODEL_REGISTRY = (
    ModelSpec(
        aliases=("baseline/l4/cpu",),
        path="models/chess_L4_H4_E256.pt",
        family="baseline",
        lineage="baseline",
        role="small reference checkpoint",
        objective="plain next-token SAN prediction",
        dataset="processed",
        size="L4 H4 E256",
        tags=("small", "legacy", "cpu-era"),
        notes="Earliest small baseline checkpoint.",
    ),
    ModelSpec(
        aliases=("baseline/l4/gpu",),
        path="models/chess_L4_H4_E256_gpu.pt",
        family="baseline",
        lineage="baseline",
        role="small GPU-trained checkpoint",
        objective="plain next-token SAN prediction",
        dataset="processed",
        size="L4 H4 E256",
        tags=("small", "legacy", "gpu"),
        notes="GPU variant of the small baseline line.",
    ),
    ModelSpec(
        aliases=("baseline/l8/reference",),
        path="models/chess_L8_H8_E512.pt",
        family="baseline",
        lineage="baseline",
        role="mid-size baseline reference",
        objective="plain next-token SAN prediction",
        dataset="processed",
        size="L8 H8 E512",
        tags=("medium", "legacy", "reference"),
        notes="Scale-up baseline between the small and large plain next-token models.",
    ),
    ModelSpec(
        aliases=("baseline/l12/reference", "baseline/reference"),
        path="models/chess_L12_H6_E768.pt",
        family="baseline",
        lineage="baseline",
        role="large plain next-token reference",
        objective="plain next-token SAN prediction",
        dataset="processed",
        size="L12 H6 E768",
        tags=("large", "reference", "legacy"),
        notes="Main plain next-token baseline used by several older scripts.",
    ),
    ModelSpec(
        aliases=("weighted/l12/reference", "weighted/reference"),
        path="models/chess_weighted_L12_H6_E768.pt",
        family="weighted",
        lineage="weighted",
        role="weighted tactical reference",
        objective="weighted next-token SAN prediction",
        dataset="puzzle_weighted",
        size="L12 H6 E768",
        tags=("weighted", "tactical", "dirtier-legality"),
        notes="Weights emphasize high-swing puzzle-derived positions.",
    ),
    ModelSpec(
        aliases=("puzzle-plain/l12/reference", "puzzle-plain/reference"),
        path="models/chess_puzzle_plain_L12_H6_E768.pt",
        family="puzzle-plain",
        lineage="puzzle-plain",
        role="large-data plain next-token control",
        objective="plain next-token SAN prediction on the puzzle-derived corpus",
        dataset="puzzle_weighted",
        size="L12 H6 E768",
        tags=("ablation", "large-data", "plain-objective"),
        notes="Control line for separating larger puzzle-derived data from the effect of puzzle weighting.",
    ),
    ModelSpec(
        aliases=("puzzle-plain/l12/500k", "puzzle-plain/500k"),
        path="models/chess_puzzle_plain_500k_L12_H6_E768.pt",
        family="puzzle-plain",
        lineage="puzzle-plain-500k",
        role="500k-game puzzle-derived plain ablation",
        objective="plain next-token SAN prediction on a 500k-game puzzle-derived subset",
        dataset="puzzle_weighted_500k",
        size="L12 H6 E768",
        tags=("ablation", "500k", "plain-objective"),
        notes="Planned size-matched ablation against the larger puzzle-derived plain run.",
    ),
    ModelSpec(
        aliases=("eval-head/l12/reference", "eval/reference"),
        path="models/chess_eval_L12_H6_E768.pt",
        family="eval-head",
        lineage="weighted-eval-head",
        role="weighted backbone plus eval head",
        objective="move loss plus eval-head fine-tuning",
        dataset="eval",
        size="L12 H6 E768",
        tags=("eval-head", "experimental"),
        notes="Backbone checkpoint with an attached eval head.",
    ),
    ModelSpec(
        aliases=("scratch/v1/best", "scratch/clean-baseline"),
        path="models/chess_scratch_L12_H6_E768_best.pt",
        family="scratch",
        lineage="scratch-v1",
        role="cleaner scratch checkpoint",
        objective="from-scratch move+eval training, v1 best checkpoint",
        dataset="eval",
        size="L12 H6 E768",
        tags=("cleaner-legality", "scratch", "best"),
        notes="Older scratch checkpoint; cleaner raw legality than scratch/v2 in saved logs.",
    ),
    ModelSpec(
        aliases=("scratch/v2/best", "scratch/latest"),
        path="models/chess_scratch_v2_L12_H6_E768_best.pt",
        family="scratch",
        lineage="scratch-v2",
        role="latest local scratch checkpoint",
        objective="from-scratch move+eval training, v2 best combined checkpoint",
        dataset="eval",
        size="L12 H6 E768",
        tags=("latest-local", "eval-aware", "scratch", "best"),
        notes="Latest local scratch checkpoint; stronger in some settings, worse raw legality than scratch/v1.",
    ),
)


ALIAS_INDEX = {alias: spec for spec in MODEL_REGISTRY for alias in spec.aliases}
FILENAME_INDEX = {spec.filename: spec for spec in MODEL_REGISTRY}
PATH_INDEX = {str(spec.absolute_path.resolve()): spec for spec in MODEL_REGISTRY}


def list_models(*, family: str | None = None, tag: str | None = None) -> list[ModelSpec]:
    specs = list(MODEL_REGISTRY)
    if family is not None:
        specs = [spec for spec in specs if spec.family == family]
    if tag is not None:
        specs = [spec for spec in specs if tag in spec.tags]
    return specs


def get_model_spec(ref: str | Path) -> ModelSpec | None:
    ref_str = str(ref)
    if ref_str in ALIAS_INDEX:
        return ALIAS_INDEX[ref_str]

    ref_path = Path(ref_str)
    if ref_path.name in FILENAME_INDEX:
        return FILENAME_INDEX[ref_path.name]

    if ref_path.is_absolute():
        return PATH_INDEX.get(str(ref_path.resolve()))

    direct = PROJECT_ROOT / ref_path
    if direct.exists():
        return PATH_INDEX.get(str(direct.resolve()))

    models_relative = MODELS_DIR / ref_path
    if models_relative.exists():
        return PATH_INDEX.get(str(models_relative.resolve()))

    return None


def resolve_model_ref(ref: str | Path) -> tuple[str, ModelSpec | None]:
    spec = get_model_spec(ref)
    if spec is not None:
        return str(spec.absolute_path), spec

    ref_path = Path(ref)
    if ref_path.is_absolute():
        return str(ref_path), None

    direct = PROJECT_ROOT / ref_path
    if direct.exists():
        return str(direct), None

    models_relative = MODELS_DIR / ref_path
    if models_relative.exists():
        return str(models_relative), None

    if str(ref_path).startswith(("models/", "./", "../")):
        return str(direct), None

    return str(models_relative), None


def model_ref_help() -> str:
    return (
        "Model reference can be a registry alias "
        "(e.g. scratch/v2/best, scratch/clean-baseline, weighted/reference), "
        "a filename, or a filesystem path."
    )


def _spec_to_dict(spec: ModelSpec) -> dict:
    data = asdict(spec)
    data["primary_alias"] = spec.primary_alias
    data["absolute_path"] = str(spec.absolute_path)
    return data


def _print_table(specs: list[ModelSpec]) -> None:
    if not specs:
        print("No models matched.")
        return

    alias_width = max(len(spec.primary_alias) for spec in specs)
    family_width = max(len(spec.family) for spec in specs)
    size_width = max(len(spec.size) for spec in specs)
    header = (
        f"{'Alias':<{alias_width}}  "
        f"{'Family':<{family_width}}  "
        f"{'Size':<{size_width}}  "
        f"File"
    )
    print(header)
    print("-" * len(header))
    for spec in specs:
        print(
            f"{spec.primary_alias:<{alias_width}}  "
            f"{spec.family:<{family_width}}  "
            f"{spec.size:<{size_width}}  "
            f"{spec.filename}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="List and resolve local nanoDanya model aliases.")
    parser.add_argument("ref", nargs="?", help="Optional alias, filename, or path to resolve.")
    parser.add_argument("--family", help="Filter listed models by family.")
    parser.add_argument("--tag", help="Filter listed models by tag.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    args = parser.parse_args()

    if args.ref:
        resolved_path, spec = resolve_model_ref(args.ref)
        if args.json:
            print(json.dumps({"ref": args.ref, "resolved_path": resolved_path, "spec": _spec_to_dict(spec) if spec else None}, indent=2))
        else:
            print(f"ref: {args.ref}")
            print(f"resolved_path: {resolved_path}")
            if spec is None:
                print("spec: <unregistered>")
            else:
                print(f"primary_alias: {spec.primary_alias}")
                print(f"family: {spec.family}")
                print(f"tags: {', '.join(spec.tags)}")
                print(f"notes: {spec.notes}")
        return

    specs = list_models(family=args.family, tag=args.tag)
    if args.json:
        print(json.dumps([_spec_to_dict(spec) for spec in specs], indent=2))
    else:
        _print_table(specs)


if __name__ == "__main__":
    main()
