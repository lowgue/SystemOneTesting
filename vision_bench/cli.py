"""Single entry point for validation, benchmarks, reports, data and Clef."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

import psutil

from .core import DEFAULT_CONTRACT, ROOT, load_json, pack_dataset, unpack_dataset


def _add_common_inputs(parser, default_config, default_manifest):
    """Add the shared config, manifest, contract, data-root, and model options to a subcommand."""
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--manifest", type=Path, default=default_manifest)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--data-root", type=Path, default=ROOT)
    parser.add_argument("--models", nargs="+", help="Model IDs from config")


def _parse_threads(value):
    """Accept automatic CPU sizing or a positive explicit Torch thread count."""
    if value == "auto":
        return value
    try:
        thread_count = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("use 'auto' or a positive integer") from None
    if thread_count < 1:
        raise argparse.ArgumentTypeError("thread count must be positive")
    return thread_count


def _build_parser():
    """Define the command-line interface and its command-specific options."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    default_config = ROOT / "benchmarks/configs/local.json"
    default_manifest = ROOT / "benchmarks/datasets/vehicles/portable.json"

    doctor = commands.add_parser(
        "doctor",
        help="Validate files, dependencies and environment without inference",
    )
    _add_common_inputs(doctor, default_config, default_manifest)

    run = commands.add_parser("run", help="Run models sequentially with frozen configuration")
    _add_common_inputs(run, default_config, default_manifest)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--repetitions", type=int, default=3)
    run.add_argument("--threads", type=_parse_threads, default="auto")
    run.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    run.add_argument("--offline", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--limit", type=int)
    run.add_argument("--machine-id")
    run.add_argument("--max-edge", type=int, default=1024)
    run.add_argument("--seed", type=int, default=20261026)

    report = commands.add_parser(
        "report", help="Report runs separately, without pooling different machines"
    )
    report.add_argument("runs", type=Path, nargs="+")
    report.add_argument("--output", type=Path, required=True)

    pack = commands.add_parser(
        "pack-data", help="Export a verified image and manifest ZIP without model weights"
    )
    pack.add_argument("--manifest", type=Path, required=True)
    pack.add_argument("--data-root", type=Path, default=ROOT)
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)

    unpack = commands.add_parser(
        "unpack-data", help="Verify image hashes before importing a bundle"
    )
    unpack.add_argument("bundle", type=Path)
    unpack.add_argument("--destination", type=Path, required=True)
    unpack.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)

    serve = commands.add_parser(
        "clef-server", help="Download pinned weights and start a Clef-capable llama-server"
    )
    serve.add_argument("--weights", type=Path, default=ROOT / "benchmarks/configs/clef_weights.json")
    serve.add_argument("--llama-server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=11437)
    serve.add_argument("--context", type=int, default=4096)
    serve.add_argument("--threads", type=int, default=8)
    serve.add_argument("--gpu-layers", type=int, default=0)
    serve.add_argument("--metadata", type=Path, default=ROOT / ".artifacts/clef-server.json")
    serve.add_argument("--dry-run", action="store_true")

    return parser


def _doctor(args):
    """Validate the selected models, dataset, dependencies, and environment without inference."""
    from .runner import environment, preflight

    specs, manifest, issues = preflight(
        load_json(args.config),
        args.manifest,
        load_json(args.contract),
        args.data_root,
        args.models,
    )
    return {
        "status": "ready" if not issues else "needs-setup",
        "models": [spec["id"] for spec in specs],
        "events": len(manifest["items"]),
        "issues": issues,
        "environment": environment(),
        "note": (
            "No model load or inference; HTTP endpoint availability and protocol "
            "support require a smoke run."
        ),
    }


def _run_benchmark(args):
    """Validate run-specific options and start the benchmark runner."""
    if args.max_edge < 1:
        raise ValueError("max-edge must be positive")

    from .runner import run

    return run(args)


def _dispatch(args):
    """Route parsed arguments to the requested command implementation."""
    if args.command == "doctor":
        return _doctor(args), 0
    if args.command == "run":
        return _run_benchmark(args), 0
    if args.command == "report":
        from .reporting.report import generate

        return {"report": generate(args.runs, args.output)}, 0
    if args.command == "pack-data":
        result = pack_dataset(args.manifest, args.data_root, args.output, load_json(args.contract))
        return result, 0
    if args.command == "unpack-data":
        manifest = unpack_dataset(args.bundle, args.destination, load_json(args.contract))
        return {"manifest": str(manifest)}, 0
    if args.command == "clef-server":
        from .clef import serve

        return serve(args), 0
    raise ValueError(f"Unknown command: {args.command}")


def main(argv=None):
    """Parse command-line arguments, execute a command, and return its process status."""
    args = _build_parser().parse_args(argv)
    try:
        result, exit_code = _dispatch(args)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.command == "doctor" and result["status"] != "ready":
            return 1
        return exit_code
    except (
        ValueError,
        RuntimeError,
        OSError,
        KeyError,
        ImportError,
        subprocess.SubprocessError,
        psutil.Error,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
