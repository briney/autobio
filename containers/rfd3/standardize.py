#!/usr/bin/env python3
"""Standardize RFDiffusion3 outputs into autobio schema format.

Walks the raw output directory, reads per-design CIF and JSON files, copies
structures to ``outputs/standardized/``, and produces ``result_data.json``
conforming to the ``StructureDesignOutput`` schema.

RFD3 output structure is flat in out_dir::

    <out_dir>/spec_{spec_name}_{batch_index}_model_{model_index}.cif.gz
    <out_dir>/spec_{spec_name}_{batch_index}_model_{model_index}.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
from pathlib import Path

# Matches: spec_{spec_name}_{batch_index}_model_{model_index}
_FILENAME_RE = re.compile(
    r"^spec_(?P<spec_name>.+)_(?P<batch>\d+)_model_(?P<model>\d+)$"
)


def _find_designs(raw_dir: Path) -> list[dict]:
    """Find all design outputs in the flat raw output directory."""
    designs: list[dict] = []

    for cif_path in sorted(raw_dir.glob("*.cif.gz")):
        stem = cif_path.name.removesuffix(".cif.gz")

        m = _FILENAME_RE.match(stem)
        if m:
            spec_name = m.group("spec_name")
            batch_index = int(m.group("batch"))
            design_index = int(m.group("model"))
        else:
            # Fallback: treat entire stem as spec_name with batch=0, design=0
            spec_name = stem
            batch_index = 0
            design_index = 0

        # Read companion JSON metadata if it exists
        json_path = cif_path.with_name(stem + ".json")
        metadata = None
        if json_path.exists():
            try:
                metadata = json.loads(json_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        # Prefer the canonical spec name from the metadata ("example" field)
        # over the filename-derived name, which may include a file prefix.
        if metadata:
            example_name = (
                metadata.get("specification", {}).get("extra", {}).get("example")
            )
            if example_name:
                spec_name = example_name

        designs.append({
            "spec_name": spec_name,
            "batch_index": batch_index,
            "design_index": design_index,
            "raw_cif_path": cif_path,
            "diffusion_metadata": metadata,
        })

    return designs


def _copy_structure(
    raw_cif_path: Path,
    std_dir: Path,
    spec_name: str,
    batch_index: int,
    design_index: int,
) -> Path:
    """Decompress a gzipped CIF file to the standardized dir.

    Returns the path within the standardized directory.
    """
    dest_name = f"{spec_name}_b{batch_index}_d{design_index}.cif"
    dest_path = std_dir / dest_name

    if raw_cif_path.name.endswith(".gz"):
        with gzip.open(raw_cif_path, "rb") as f_in:
            dest_path.write_bytes(f_in.read())
    else:
        shutil.copy2(raw_cif_path, dest_path)

    return dest_path


def standardize(workspace: Path) -> None:
    """Transform raw RFD3 outputs into standardized result_data.json."""
    raw_dir = workspace / "outputs" / "raw"
    std_dir = workspace / "outputs" / "standardized"
    std_dir.mkdir(parents=True, exist_ok=True)

    raw_designs = _find_designs(raw_dir)

    if not raw_designs:
        raise RuntimeError(
            f"No RFD3 design outputs (.cif.gz) found in {raw_dir}. "
            f"Check logs/tool.log for execution errors."
        )

    designs = []
    spec_summary: dict[str, int] = {}

    for d in raw_designs:
        # Copy structure to standardized dir (decompressed)
        structure_path = _copy_structure(
            d["raw_cif_path"],
            std_dir,
            d["spec_name"],
            d["batch_index"],
            d["design_index"],
        )

        designs.append({
            "spec_name": d["spec_name"],
            "batch_index": d["batch_index"],
            "design_index": d["design_index"],
            "structure_path": str(structure_path),
            "diffusion_metadata": d["diffusion_metadata"],
        })

        spec_summary[d["spec_name"]] = spec_summary.get(d["spec_name"], 0) + 1

    result_data = {
        "designs": designs,
        "spec_summary": spec_summary,
    }
    (std_dir / "result_data.json").write_text(json.dumps(result_data, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardize RFD3 outputs.")
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    standardize(args.workspace)
