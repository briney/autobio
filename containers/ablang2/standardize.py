#!/usr/bin/env python3
"""Standardize AbLang2 outputs into autobio schema format.

Handles both embedding mode (copies .npy files) and PLL mode (copies scores).
Reads metadata produced by run_ablang2.py from outputs/raw/ and writes
result_data.json to outputs/standardized/.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def standardize(workspace: Path) -> None:
    raw_dir = workspace / "outputs" / "raw"
    std_dir = workspace / "outputs" / "standardized"

    metadata_path = raw_dir / "metadata.json"
    if not metadata_path.exists():
        raise RuntimeError(
            f"metadata.json not found in {raw_dir}. "
            f"Files present: {[f.name for f in raw_dir.iterdir()]}"
        )
    metadata = json.loads(metadata_path.read_text())

    # Read config to determine mode
    config = json.loads((workspace / "config.json").read_text())
    mode = config["mode"]

    # Use model name as-is (no org prefix to strip for ablang2)
    model_name = metadata["model_name"]
    if "/" in model_name:
        model_name = model_name.split("/", 1)[1]

    if mode == "embedding":
        _standardize_embeddings(metadata, model_name, raw_dir, std_dir)
    elif mode == "pll":
        _standardize_pll(metadata, model_name, std_dir)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _standardize_embeddings(
    metadata: dict,
    model_name: str,
    raw_dir: Path,
    std_dir: Path,
) -> None:
    """Copy .npy files and write embedding result_data.json."""
    embeddings = []
    for result in metadata["results"]:
        raw_path = Path(result["embedding_path"])
        std_path = std_dir / raw_path.name
        shutil.copy2(raw_path, std_path)

        embeddings.append({
            "sequence_id": result["sequence_id"],
            "embedding_path": f"/workspace/outputs/standardized/{raw_path.name}",
            "dimension": result["dimension"],
            "layer": result["layer"],
            "pooling": result["pooling"],
        })

    result_data = {
        "embeddings": embeddings,
        "model_name": model_name,
        "embedding_dimension": metadata["embedding_dimension"],
    }
    (std_dir / "result_data.json").write_text(json.dumps(result_data, indent=2))


def _standardize_pll(
    metadata: dict,
    model_name: str,
    std_dir: Path,
) -> None:
    """Write PLL result_data.json."""
    scores = []
    for result in metadata["results"]:
        entry = {
            "sequence_id": result["sequence_id"],
            "pll": result["pll"],
            "sequence_length": result["sequence_length"],
        }
        if "per_position_pll" in result:
            entry["per_position_pll"] = result["per_position_pll"]
        scores.append(entry)

    result_data = {
        "scores": scores,
        "model_name": model_name,
    }
    (std_dir / "result_data.json").write_text(json.dumps(result_data, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    standardize(args.workspace)
