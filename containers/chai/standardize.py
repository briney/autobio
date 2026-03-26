#!/usr/bin/env python3
"""Standardize Chai-1 outputs into autobio schema format.

Chai-1 writes predictions to ``<output_dir>/``:

- ``pred.model_idx_<N>.cif`` — mmCIF structure files (one per candidate)
- ``scores.model_idx_<N>.npz`` — scoring files with confidence metrics

The ``.npz`` files contain 1D arrays:
- ``aggregate_score``: shape (1,) — overall ranking score
- ``ptm``: shape (1,) — predicted TM-score
- ``iptm``: shape (1,) — interface predicted TM-score
- ``per_chain_ptm``: shape (1, N_chains)
- ``per_chain_pair_iptm``: shape (1, N_chains, N_chains)

pLDDT values are stored as B-factors in the mmCIF structure files.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


def _find_structure_files(raw_dir: Path) -> list[Path]:
    """Find all Chai-1 structure output files.

    Chai-1 names them ``pred.model_idx_<N>.cif``. We use a glob pattern
    and filter to avoid matching non-structure files.
    """
    structures = sorted(raw_dir.glob("pred.model_idx_*.cif"))

    if not structures:
        # Chai-1 may nest output in a subdirectory — search recursively
        structures = sorted(raw_dir.rglob("pred.model_idx_*.cif"))

    if not structures:
        raise RuntimeError(
            f"No Chai-1 structure files (pred.model_idx_*.cif) found in {raw_dir}. "
            f"Contents: {[f.name for f in raw_dir.iterdir()]}"
        )
    return structures


def _read_scores(structure_path: Path) -> dict:
    """Read the companion scoring .npz file for a structure.

    Chai-1 stores scores as 1D numpy arrays (shape (1,)) for scalar metrics.
    """
    try:
        import numpy as np
    except ImportError:
        return {}

    # pred.model_idx_0.cif → scores.model_idx_0.npz
    stem = structure_path.stem  # "pred.model_idx_0"
    idx_part = stem.replace("pred.", "")  # "model_idx_0"
    scores_name = f"scores.{idx_part}.npz"
    scores_path = structure_path.parent / scores_name

    if not scores_path.exists():
        for candidate in structure_path.parent.glob(f"scores.*{idx_part}*.npz"):
            scores_path = candidate
            break
        else:
            return {}

    try:
        data = np.load(scores_path, allow_pickle=True)
        scores: dict = {}

        # Chai-1 v0.6 stores metrics as 1D arrays with shape (1,)
        if "aggregate_score" in data.files:
            val = data["aggregate_score"]
            scores["aggregate_score"] = float(val.flat[0])

        if "ptm" in data.files:
            val = data["ptm"]
            scores["ptm"] = float(val.flat[0])

        if "iptm" in data.files:
            val = data["iptm"]
            scores["iptm"] = float(val.flat[0])

        return scores
    except (OSError, ValueError, KeyError, IndexError):
        return {}


def _read_plddt_from_cif(cif_path: Path) -> tuple[list[float] | None, float | None]:
    """Extract per-residue pLDDT from CIF B-factors (Cα atoms only).

    Chai-1 writes pLDDT values as B-factors in the mmCIF ``_atom_site`` loop.
    We extract Cα atoms for proteins and C1' for nucleotides.

    Returns:
        (per_residue_plddt, mean_plddt) or (None, None) on failure.
    """
    try:
        bfactor_col = None
        atom_name_col = None
        cols: list[str] = []
        in_atom_site = False
        ca_bfactors: list[float] = []
        all_bfactors: list[float] = []

        with open(cif_path) as f:
            for line in f:
                if line.startswith("_atom_site."):
                    in_atom_site = True
                    col_name = line.strip().split(".")[1]
                    cols.append(col_name)
                    if col_name == "B_iso_or_equiv":
                        bfactor_col = len(cols) - 1
                    elif col_name == "label_atom_id":
                        atom_name_col = len(cols) - 1
                elif in_atom_site and (line.startswith("ATOM") or line.startswith("HETATM")):
                    parts = line.split()
                    if bfactor_col is not None and len(parts) > bfactor_col:
                        bfactor = float(parts[bfactor_col])
                        all_bfactors.append(bfactor)
                        # Extract Cα (protein) and C1' (nucleotide) atoms
                        if atom_name_col is not None and len(parts) > atom_name_col:
                            atom_name = parts[atom_name_col]
                            if atom_name in ("CA", "C1'"):
                                ca_bfactors.append(bfactor)
                elif in_atom_site and line.strip() == "#":
                    break

        if ca_bfactors:
            mean_plddt = sum(ca_bfactors) / len(ca_bfactors)
            return ca_bfactors, mean_plddt
        elif all_bfactors:
            mean_plddt = sum(all_bfactors) / len(all_bfactors)
            return None, mean_plddt
        return None, None
    except (OSError, ValueError):
        return None, None


def standardize(workspace: Path) -> None:
    """Transform raw Chai-1 outputs into standardized result_data.json."""
    raw_dir = workspace / "outputs" / "raw"
    std_dir = workspace / "outputs" / "standardized"
    std_dir.mkdir(parents=True, exist_ok=True)

    structure_files = _find_structure_files(raw_dir)

    # Parse each structure with its scores and pLDDT
    models: list[dict] = []
    for struct_path in structure_files:
        scores = _read_scores(struct_path)
        plddt_per_residue, plddt_mean = _read_plddt_from_cif(struct_path)

        models.append({
            "raw_path": struct_path,
            "aggregate_score": scores.get("aggregate_score"),
            "ptm": scores.get("ptm"),
            "iptm": scores.get("iptm"),
            "plddt_per_residue": plddt_per_residue,
            "plddt_mean": plddt_mean,
        })

    # Rank by aggregate_score (highest first), with None values last
    models.sort(
        key=lambda m: (m["aggregate_score"] is not None, m["aggregate_score"] or 0),
        reverse=True,
    )

    # Copy structures to standardized dir and build result
    structures = []
    for rank, model in enumerate(models, start=1):
        suffix = model["raw_path"].suffix
        dest_name = f"model_{rank}{suffix}"
        dest_path = std_dir / dest_name
        shutil.copy2(model["raw_path"], dest_path)

        structures.append({
            "model_rank": rank,
            "structure_path": str(dest_path),
            "plddt_per_residue": model["plddt_per_residue"],
            "plddt_mean": model["plddt_mean"],
            "ptm": model["ptm"],
            "iptm": model["iptm"],
            "chain_mapping": None,
        })

    # Build summary confidence metrics from best model
    best = models[0] if models else {}
    confidence_metrics = {
        "best_plddt_mean": best.get("plddt_mean"),
        "best_ptm": best.get("ptm"),
        "best_iptm": best.get("iptm"),
    }

    result_data = {
        "structures": structures,
        "confidence": confidence_metrics,
    }
    (std_dir / "result_data.json").write_text(json.dumps(result_data, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardize Chai-1 outputs.")
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    standardize(args.workspace)
