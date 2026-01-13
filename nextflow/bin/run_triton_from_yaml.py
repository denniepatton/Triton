#!/usr/bin/env python

import argparse
import os
import pathlib
import subprocess
import sys
from typing import Any, Sequence, cast

try:
    import yaml  # type: ignore
except Exception:
    sys.stderr.write(
        "ERROR: PyYAML is required for Nextflow support. "
        "Add 'pyyaml' to containers/environment.yml or install it in your local env.\n"
    )
    raise


def _load_yaml(path: pathlib.Path) -> dict[str, Any]:
    with path.open("r") as f:
        data: Any = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected YAML structure in {path}")
    return cast(dict[str, Any], data)


def _resolve_path(repo_dir: pathlib.Path, maybe_path: str) -> str:
    # Keep absolute paths as-is; resolve relative paths against repo root.
    p = pathlib.Path(maybe_path)
    if p.is_absolute():
        return str(p)

    # Prefer resolving relative paths against the current working directory.
    # Nextflow stages project inputs (e.g. config/) into the task work dir.
    cwd_candidate = (pathlib.Path.cwd() / p)
    if cwd_candidate.exists():
        return str(cwd_candidate.resolve())

    repo_candidate = (repo_dir / p)
    if repo_candidate.exists():
        return str(repo_candidate.resolve())

    # Fall back to repo_dir-based resolution even if the path doesn't exist yet.
    return str((repo_dir / p).resolve())


def _parse_size_range(value: Any) -> tuple[int, int]:
    if value is None:
        return (15, 500)
    if isinstance(value, (list, tuple)):
        seq = cast(Sequence[Any], value)
        if len(seq) == 2:
            return (int(seq[0]), int(seq[1]))
    if isinstance(value, str):
        parts = value.split()
        if len(parts) == 2:
            return (int(parts[0]), int(parts[1]))
    raise ValueError(f"Invalid size_range value: {value!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Triton/Triton.py using values from config/config.yaml")
    parser.add_argument("--repo-dir", required=True, help="Path to Triton repo root")
    parser.add_argument("--config-yaml", required=True, help="Path to config/config.yaml")
    parser.add_argument("--sample", required=True, help="Sample name")
    parser.add_argument("--bam", required=True, help="BAM/CRAM path")
    parser.add_argument("--gc-bias", required=True, help="GC bias file path")
    parser.add_argument("--results-dir", required=True, help="Output directory for Triton (within the task work dir)")
    parser.add_argument("--cpus", required=True, type=int, help="CPUs to pass to Triton.py")

    # Optional overrides (if you want to avoid editing config/config.yaml)
    parser.add_argument("--annotation", required=False)
    parser.add_argument("--reference-genome", required=False)
    parser.add_argument("--map-quality", required=False, type=int)
    parser.add_argument("--run-mode", required=False)
    parser.add_argument("--size-range", required=False, nargs=2, type=int)

    args = parser.parse_args()

    repo_dir = pathlib.Path(args.repo_dir).resolve()
    config_yaml = pathlib.Path(args.config_yaml).resolve()

    cfg = _load_yaml(config_yaml)

    annotation = args.annotation or cfg.get("annotation")
    reference_genome = args.reference_genome or cfg.get("reference_genome")
    map_quality = args.map_quality if args.map_quality is not None else int(cfg.get("map_quality", 20))
    run_mode = args.run_mode or cfg.get("run_mode")

    if not annotation:
        raise ValueError("Missing 'annotation' in config/config.yaml (or pass --annotation)")
    if not reference_genome:
        raise ValueError("Missing 'reference_genome' in config/config.yaml (or pass --reference-genome)")
    if not run_mode:
        raise ValueError("Missing 'run_mode' in config/config.yaml (or pass --run-mode)")

    size_range = (args.size_range[0], args.size_range[1]) if args.size_range else _parse_size_range(cfg.get("size_range"))

    triton_py = repo_dir / "Triton" / "Triton.py"
    frag_dict = repo_dir / "nc_fitting" / "NCDict.pkl"

    cmd: list[str] = [
        sys.executable,
        str(triton_py),
        "--sample_name",
        args.sample,
        "--input",
        _resolve_path(repo_dir, args.bam),
        "--bias",
        _resolve_path(repo_dir, args.gc_bias),
        "--annotation",
        _resolve_path(repo_dir, str(annotation)),
        "--reference_genome",
        _resolve_path(repo_dir, str(reference_genome)),
        "--results_dir",
        args.results_dir,
        "--map_quality",
        str(map_quality),
        "--size_range",
        str(size_range[0]),
        str(size_range[1]),
        "--cpus",
        str(args.cpus),
        "--run_mode",
        str(run_mode),
        "--frag_dict",
        str(frag_dict),
    ]

    # Make sure the working directory is writable and predictable.
    os.makedirs(args.results_dir, exist_ok=True)

    sys.stderr.write("Running: " + " ".join(cmd) + "\n")
    sys.stderr.flush()

    subprocess.check_call(cmd)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
