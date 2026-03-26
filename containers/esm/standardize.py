#!/usr/bin/env python3
"""Standardize ESM embedding outputs into autobio schema format.

Reads the metadata and .npy files produced by run_esm.py from outputs/raw/
and writes result_data.json plus copied .npy files to outputs/standardized/.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def standardize(workspace: Path) -> None:
    raw_dir = workspace / "outputs" / "raw"
    std_dir = workspace / "outputs" / "standardized"

    # Read metadata written by run_esm.py
    metadata_path = raw_dir / "metadata.json"
    if not metadata_path.exists():
        raise RuntimeError(
            f"metadata.json not found in {raw_dir}. "
            f"Files present: {[f.name for f in raw_dir.iterdir()]}"
        )
    metadata = json.loads(metadata_path.read_text())

    # Copy .npy files to standardized/ and rewrite paths
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

    # Strip the "facebook/" prefix for a cleaner model_name in output
    model_name = metadata["model_name"]
    if model_name.startswith("facebook/"):
        model_name = model_name[len("facebook/"):]

    result_data = {
        "embeddings": embeddings,
        "model_name": model_name,
        "embedding_dimension": metadata["embedding_dimension"],
    }

    (std_dir / "result_data.json").write_text(json.dumps(result_data, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    standardize(args.workspace)
