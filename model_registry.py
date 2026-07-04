from __future__ import annotations

import argparse
from collections import Counter
import json
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"
VOLUME_ROOT = Path("/data")


@dataclass(frozen=True)
class ModelSpec:
    aliases: tuple[str, ...]
    path: str
    family: str
    data_source: str
    scale: str
    arch: str
    dataset: str
    status: str
    init: str = "scratch"
    base_model: str | None = None
    notes: str = ""

    @property
    def primary_alias(self) -> str:
        return self.aliases[0]

    @property
    def filename(self) -> str:
        return Path(self.path).name

    @property
    def absolute_path(self) -> Path:
        path = Path(self.path)
        if path.is_absolute():
            return path

        project_path = PROJECT_ROOT / path
        if project_path.exists():
            return project_path

        volume_path = VOLUME_ROOT / path
        if volume_path.exists():
            return volume_path

        return project_path


MODEL_REGISTRY = (
    ModelSpec(
        aliases=("plain/games-500k", "plain/games-500k/l12"),
        path="checkpoints/plain/games-500k/l12.pt",
        family="plain",
        data_source="games",
        scale="500k",
        arch="L12_H6_E768",
        dataset="datasets/games/500k",
        status="ready",
        notes="Main normal-game baseline. The unsuffixed alias resolves to the L12 architecture.",
    ),
    ModelSpec(
        aliases=("plain/games-500k/l8",),
        path="checkpoints/plain/games-500k/l8.pt",
        family="plain",
        data_source="games",
        scale="500k",
        arch="L8_H8_E512",
        dataset="datasets/games/500k",
        status="ready",
        notes="Older L8 normal-game checkpoint, historically named chess_min.pt.",
    ),
    ModelSpec(
        aliases=("plain/games-5m", "plain/games-5m/l12"),
        path="checkpoints/plain/games-5m/l12_best.pt",
        family="plain",
        data_source="games",
        scale="5m",
        arch="L12_H6_E768",
        dataset="datasets/games/5m",
        status="ready",
        notes="Normal-game 5M best checkpoint.",
    ),
    ModelSpec(
        aliases=("plain/games-15m", "plain/games-15m/l16"),
        path="checkpoints/plain/games-15m/l16_best.pt",
        family="plain",
        data_source="games",
        scale="15m",
        arch="L16_H8_E1024",
        dataset="datasets/games/15m",
        status="ready",
        notes="Champion default (step 200k). Beats the size-matched L12 only 54.45% H2H over 1000 games.",
    ),
    ModelSpec(
        aliases=("plain/games-15m/l12",),
        path="checkpoints/plain/games-15m/l12_best.pt",
        family="plain",
        data_source="games",
        scale="15m",
        arch="L12_H6_E768",
        dataset="datasets/games/15m",
        status="ready",
        notes="Size-matched 15M L12; source of the H2H matrix, Stockfish ladder, and blog numbers.",
    ),
    ModelSpec(
        aliases=("plain/puzzles-5m", "plain/puzzles-5m/l12"),
        path="checkpoints/plain/puzzles-5m/l12.pt",
        family="plain",
        data_source="puzzles",
        scale="5m",
        arch="L12_H6_E768",
        dataset="datasets/puzzles/5m",
        status="ready",
        notes="Plain objective on the full puzzle-derived corpus.",
    ),
    ModelSpec(
        aliases=("plain/puzzles-500k", "plain/puzzles-500k/l12"),
        path="checkpoints/plain/puzzles-500k/l12_best.pt",
        family="plain",
        data_source="puzzles",
        scale="500k",
        arch="L12_H6_E768",
        dataset="datasets/puzzles/500k",
        status="ready",
        notes="Rebuilt 500k puzzle slice. Replaces the old arbitrary 500k puzzle checkpoint.",
    ),
    ModelSpec(
        aliases=("weighted/puzzles-5m", "weighted/puzzles-5m/l12"),
        path="checkpoints/weighted/puzzles-5m/l12.pt",
        family="weighted",
        data_source="puzzles",
        scale="5m",
        arch="L12_H6_E768",
        dataset="datasets/puzzles/5m",
        status="ready",
        notes="Weighted objective on the full puzzle-derived corpus.",
    ),
)


ALIAS_INDEX = {alias: spec for spec in MODEL_REGISTRY for alias in spec.aliases}
FILENAME_INDEX = {spec.filename: spec for spec in MODEL_REGISTRY}
PATH_INDEX = {str(spec.absolute_path.resolve()): spec for spec in MODEL_REGISTRY}


def list_models(
    *,
    family: str | None = None,
    status: str | None = None,
) -> list[ModelSpec]:
    specs = list(MODEL_REGISTRY)
    if family is not None:
        specs = [spec for spec in specs if spec.family == family]
    if status is not None:
        specs = [spec for spec in specs if spec.status == status]
    return specs


def get_model_spec(ref: str | Path) -> ModelSpec | None:
    ref_str = str(ref)
    if ref_str in ALIAS_INDEX:
        return ALIAS_INDEX[ref_str]

    ref_path = Path(ref_str)
    if ref_path.is_absolute():
        return PATH_INDEX.get(str(ref_path.resolve()))

    direct = PROJECT_ROOT / ref_path
    if direct.exists():
        return PATH_INDEX.get(str(direct.resolve()))

    models_relative = MODELS_DIR / ref_path
    if models_relative.exists():
        return PATH_INDEX.get(str(models_relative.resolve()))

    if ref_path.name in FILENAME_INDEX:
        return FILENAME_INDEX[ref_path.name]

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

    if str(ref_path).startswith(("models/", "checkpoints/", "datasets/", "./", "../")):
        return str(direct), None

    return str(models_relative), None


def model_ref_help() -> str:
    return (
        "Model reference can be a registry alias "
        "(e.g. plain/games-500k, plain/puzzles-5m, weighted/puzzles-5m), "
        "a filename, or a filesystem path."
    )


def _spec_to_dict(spec: ModelSpec) -> dict:
    data = asdict(spec)
    data["primary_alias"] = spec.primary_alias
    data["absolute_path"] = str(spec.absolute_path)
    return data


def _print_table(specs: list[ModelSpec], *, details: bool = False) -> None:
    if not specs:
        print("No models matched.")
        return

    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        _print_plain_table(specs)
        return

    counts = Counter(spec.family for spec in specs)
    summary = Table(title="Family Summary")
    summary.add_column("Family", style="magenta", no_wrap=True)
    summary.add_column("Checkpoints", justify="right", style="green")
    for family, count in sorted(counts.items()):
        summary.add_row(family, str(count))
    summary.add_row("[bold]Total[/bold]", f"[bold]{len(specs)}[/bold]")

    table = Table(title="Registered Checkpoints")
    table.add_column("Alias", style="cyan", no_wrap=True)
    table.add_column("Family", style="magenta", no_wrap=True)
    table.add_column("Data", style="yellow", no_wrap=True)
    table.add_column("Scale", style="green", no_wrap=True)
    table.add_column("Arch", style="green", no_wrap=True)
    table.add_column("Status", style="yellow", no_wrap=True)
    if details:
        table.add_column("Dataset", style="yellow")
        table.add_column("Init", style="cyan")
        table.add_column("File", style="dim")
        table.add_column("Notes")

    for spec in specs:
        row = [
            spec.primary_alias,
            spec.family,
            spec.data_source,
            spec.scale,
            spec.arch,
            spec.status,
        ]
        if details:
            row.extend([spec.dataset, spec.init, spec.filename, spec.notes])
        table.add_row(*row)

    console = Console()
    console.print(summary)
    console.print(table)


def _print_plain_table(specs: list[ModelSpec]) -> None:
    alias_width = max(len(spec.primary_alias) for spec in specs)
    family_width = max(len(spec.family) for spec in specs)
    data_width = max(len(spec.data_source) for spec in specs)
    scale_width = max(len(spec.scale) for spec in specs)
    arch_width = max(len(spec.arch) for spec in specs)
    status_width = max(len(spec.status) for spec in specs)
    header = (
        f"{'Alias':<{alias_width}}  "
        f"{'Family':<{family_width}}  "
        f"{'Data':<{data_width}}  "
        f"{'Scale':<{scale_width}}  "
        f"{'Arch':<{arch_width}}  "
        f"{'Status':<{status_width}}  "
        f"File"
    )
    print(header)
    print("-" * len(header))
    for spec in specs:
        print(
            f"{spec.primary_alias:<{alias_width}}  "
            f"{spec.family:<{family_width}}  "
            f"{spec.data_source:<{data_width}}  "
            f"{spec.scale:<{scale_width}}  "
            f"{spec.arch:<{arch_width}}  "
            f"{spec.status:<{status_width}}  "
            f"{spec.filename}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="List and resolve nanoDanya model aliases.")
    parser.add_argument("ref", nargs="?", help="Optional alias, filename, or path to resolve.")
    parser.add_argument("--family", help="Filter listed models by family.")
    parser.add_argument("--status", help="Filter listed models by status.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a table.")
    parser.add_argument("--details", action="store_true", help="Include dataset, init, file, and notes columns.")
    args = parser.parse_args()

    if args.ref:
        resolved_path, spec = resolve_model_ref(args.ref)
        if args.json:
            print(
                json.dumps(
                    {
                        "ref": args.ref,
                        "resolved_path": resolved_path,
                        "spec": _spec_to_dict(spec) if spec else None,
                    },
                    indent=2,
                )
            )
        else:
            print(f"ref: {args.ref}")
            print(f"resolved_path: {resolved_path}")
            if spec is None:
                print("spec: <unregistered>")
            else:
                print(f"primary_alias: {spec.primary_alias}")
                print(f"family: {spec.family}")
                print(f"data_source: {spec.data_source}")
                print(f"scale: {spec.scale}")
                print(f"arch: {spec.arch}")
                print(f"status: {spec.status}")
                print(f"init: {spec.init}")
                print(f"base_model: {spec.base_model or '<none>'}")
                print(f"notes: {spec.notes}")
        return

    specs = list_models(family=args.family, status=args.status)
    if args.json:
        print(json.dumps([_spec_to_dict(spec) for spec in specs], indent=2))
    else:
        _print_table(specs, details=args.details)


if __name__ == "__main__":
    main()
