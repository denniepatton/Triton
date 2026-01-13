#!/usr/bin/env python

import argparse
import pathlib
import sys
from typing import Any, Optional, cast

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover
    sys.stderr.write(
        "ERROR: PyYAML is required for Nextflow support. "
        "Add 'pyyaml' to containers/environment.yml or install it in your local env.\n"
    )
    raise


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Triton config/samples.yaml into a TSV manifest")
    parser.add_argument("--samples-yaml", required=True, help="Path to samples.yaml")
    parser.add_argument("--out", required=True, help="Output TSV path")
    args = parser.parse_args()

    samples_yaml = pathlib.Path(args.samples_yaml)
    out_path = pathlib.Path(args.out)

    with samples_yaml.open("r") as f:
        data_any: Any = yaml.safe_load(f)

    if not isinstance(data_any, dict):
        raise ValueError(f"Unexpected YAML structure in {samples_yaml}")

    data = cast(dict[str, Any], data_any)

    samples_any: Any = data.get("samples", data)
    if not isinstance(samples_any, dict):
        raise ValueError(f"Expected top-level 'samples:' mapping in {samples_yaml}")
    samples = cast(dict[str, Any], samples_any)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as out:
        out.write("sample\tbam\tgc_bias\n")
        for sample, info_any in samples.items():
            if not isinstance(info_any, dict):
                raise ValueError(f"Sample '{sample}' must map to a dict")
            info = cast(dict[str, Any], info_any)
            bam = cast(Optional[str], info.get("bam"))
            gc_bias = cast(Optional[str], info.get("GC_bias"))
            if not bam or not gc_bias:
                raise ValueError(f"Sample '{sample}' missing 'bam' or 'GC_bias'")
            out.write(f"{sample}\t{bam}\t{gc_bias}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
