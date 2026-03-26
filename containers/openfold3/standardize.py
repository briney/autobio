#!/usr/bin/env python3
"""Standardize OpenFold3 outputs into autobio schema format.

OpenFold3 writes predictions to ``<output_dir>/<query_name>/seed_<N>/``:

- ``<query>_seed_<N>_sample_<M>_model.cif`` — mmCIF structure files
- ``<query>_seed_<N>_sample_<M>_confidences.json`` — per-atom pLDDT and PDE
- ``<query>_seed_<N>_sample_<M>_confidences_aggregated.json`` — summary metrics:
  - ``avg_plddt`` — average pLDDT over all atoms
  - ``gpde`` — global predicted distance error
  - ``ptm`` — predicted TM-score (requires PAE enabled)
  - ``iptm`` — interface predicted TM-score (requires PAE enabled)
  - ``sample_ranking_score`` — weighted ranking score (requires PAE enabled)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _find_structure_files(raw_dir: Path) -> list[Path]:
    """Find all OpenFold3 structure output files.

    Searches recursively across query/seed directories for files matching
    ``*_model.cif`` or ``*_model.pdb``.
    """
    structures = sorted(raw_dir.rglob("*_model.cif"))
    if not structures:
        structures = sorted(raw_dir.rglob("*_model.pdb"))
    if not structures:
        raise RuntimeError(
            f"No OpenFold3 structure files (*_model.cif or *_model.pdb) found in {raw_dir}. "
            f"Contents: {[str(f.relative_to(raw_dir)) for f in raw_dir.rglob('*') if f.is_file()]}"
        )
    return structures


def _read_aggregated_confidences(structure_path: Path) -> dict:
    """Read the companion aggregated confidences JSON for a structure.

    OpenFold3 names them ``<prefix>_confidences_aggregated.json`` where
    the prefix matches the structure file minus ``_model``.
    """
    # <query>_seed_<N>_sample_<M>_model.cif → <query>_seed_<N>_sample_<M>_confidences_aggregated.json
    prefix = structure_path.name.replace("_model.cif", "").replace("_model.pdb", "")
    conf_name = f"{prefix}_confidences_aggregated.json"
    conf_path = structure_path.parent / conf_name

    if not conf_path.exists():
        return {}

    try:
        data = json.loads(conf_path.read_text())
        return {
            "avg_plddt": data.get("avg_plddt"),
            "ptm": data.get("ptm"),
            "iptm": data.get("iptm"),
            "gpde": data.get("gpde"),
            "sample_ranking_score": data.get("sample_ranking_score"),
        }
    except (OSError, ValueError, KeyError):
        return {}


def _read_per_residue_plddt(structure_path: Path) -> tuple[list[float] | None, float | None]:
    """Read per-atom pLDDT from the confidences JSON and aggregate to per-residue.

    OpenFold3 stores per-atom pLDDT in ``*_confidences.json`` under the
    ``plddt`` key as a flat list.  We return the raw per-atom list and its
    mean.  For a more precise per-residue aggregation, we would need the
    atom-to-residue mapping, but the per-atom mean is a good approximation
    and matches the ``avg_plddt`` from the aggregated file.
    """
    prefix = structure_path.name.replace("_model.cif", "").replace("_model.pdb", "")
    conf_name = f"{prefix}_confidences.json"
    conf_path = structure_path.parent / conf_name

    if not conf_path.exists():
        return None, None

    try:
        data = json.loads(conf_path.read_text())
        plddt_values = data.get("plddt")
        if plddt_values and isinstance(plddt_values, list):
            mean_plddt = sum(plddt_values) / len(plddt_values)
            return plddt_values, mean_plddt
        return None, None
    except (OSError, ValueError, KeyError):
        return None, None


def standardize(workspace: Path) -> None:
    """Transform raw OpenFold3 outputs into standardized result_data.json."""
    raw_dir = workspace / "outputs" / "raw"
    std_dir = workspace / "outputs" / "standardized"
    std_dir.mkdir(parents=True, exist_ok=True)

    structure_files = _find_structure_files(raw_dir)

    # Parse each structure with its confidence metrics
    models: list[dict] = []
    for struct_path in structure_files:
        agg = _read_aggregated_confidences(struct_path)
        plddt_per_residue, plddt_mean_from_atoms = _read_per_residue_plddt(struct_path)

        # Prefer avg_plddt from aggregated file, fall back to per-atom mean
        plddt_mean = agg.get("avg_plddt") or plddt_mean_from_atoms

        models.append({
            "raw_path": struct_path,
            "sample_ranking_score": agg.get("sample_ranking_score"),
            "avg_plddt": plddt_mean,
            "ptm": agg.get("ptm"),
            "iptm": agg.get("iptm"),
            "plddt_per_residue": plddt_per_residue,
            "plddt_mean": plddt_mean,
        })

    # Rank by sample_ranking_score (highest first).  Falls back to avg_plddt
    # if PAE was disabled (sample_ranking_score will be None).
    def _sort_key(m: dict) -> tuple[bool, float]:
        score = m["sample_ranking_score"]
        if score is not None:
            return (True, score)
        plddt = m["avg_plddt"]
        if plddt is not None:
            return (True, plddt)
        return (False, 0.0)

    models.sort(key=_sort_key, reverse=True)

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

    print(f"[openfold3] Standardized {len(structures)} structure(s).")
    if best:
        print(f"[openfold3] Best model: pLDDT={best.get('plddt_mean')}, "
              f"pTM={best.get('ptm')}, ipTM={best.get('iptm')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardize OpenFold3 outputs.")
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    standardize(args.workspace)
