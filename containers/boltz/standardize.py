#!/usr/bin/env python3
"""Standardize Boltz outputs into autobio schema format.

Boltz writes predictions to::

    <out_dir>/boltz_results_<input_name>/predictions/<input_name>/

Since the runner names its input ``input.yaml``, the output is at::

    <out_dir>/boltz_results_input/predictions/input/

Each sample produces:
- A structure file: ``<input_name>_model_<N>.cif`` (or ``.pdb``)
- A confidence JSON: ``confidence_<input_name>_model_<N>.json``
  with scalar metrics: ``confidence_score``, ``ptm``, ``iptm``,
  ``complex_plddt`` (scalar mean), ``complex_pde``
- Per-residue pLDDT: ``plddt_<input_name>_model_<N>.npz`` (optional)
- Affinity JSON: ``affinity_<input_name>_model_<N>.json`` (Boltz-2 only)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _find_predictions(raw_dir: Path) -> Path:
    """Locate the predictions output directory.

    Boltz outputs to ``<out_dir>/boltz_results_<name>/predictions/<name>/``.
    We walk the raw dir looking for a ``predictions/`` subdirectory.
    """
    # First check for boltz_results_* top-level directory
    boltz_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir() and d.name.startswith("boltz_results"))
    if boltz_dirs:
        for boltz_dir in boltz_dirs:
            predictions_dir = boltz_dir / "predictions"
            if predictions_dir.exists():
                subdirs = sorted(d for d in predictions_dir.iterdir() if d.is_dir())
                if subdirs:
                    return subdirs[0]

    # Fallback: look for predictions/ directly under raw_dir
    predictions_dir = raw_dir / "predictions"
    if predictions_dir.exists():
        subdirs = sorted(d for d in predictions_dir.iterdir() if d.is_dir())
        if subdirs:
            return subdirs[0]

    raise RuntimeError(
        f"No predictions directory found in {raw_dir}. "
        f"Contents: {[f.name for f in raw_dir.iterdir()]}"
    )


def _find_structure_files(pred_dir: Path) -> list[Path]:
    """Find all structure output files (.cif or .pdb) in the prediction dir.

    Boltz names them ``<input_name>_model_<N>.cif``. We exclude any files
    that contain ``confidence``, ``plddt``, ``pae``, ``pde``, or ``affinity``
    in the stem to avoid matching companion files.
    """
    skip_keywords = {"confidence", "plddt", "pae", "pde", "affinity"}
    structures = []
    for ext in ("*.cif", "*.pdb"):
        for f in sorted(pred_dir.glob(ext)):
            if not any(kw in f.stem for kw in skip_keywords):
                structures.append(f)

    if not structures:
        raise RuntimeError(
            f"No structure files (.cif/.pdb) found in {pred_dir}. "
            f"Contents: {[f.name for f in pred_dir.iterdir()]}"
        )
    return structures


def _read_confidence(structure_path: Path) -> dict | None:
    """Read the companion confidence JSON for a structure file.

    Boltz names confidence files as ``confidence_<input>_model_<N>.json``
    alongside ``<input>_model_<N>.cif``.
    """
    stem = structure_path.stem  # e.g., "input_model_0"
    parent = structure_path.parent

    # Primary pattern: confidence_<stem>.json
    confidence_path = parent / f"confidence_{stem}.json"
    if not confidence_path.exists():
        # Fallback: search for any confidence JSON containing the stem
        for candidate in parent.glob("confidence_*.json"):
            if stem in candidate.name:
                confidence_path = candidate
                break
        else:
            return None

    try:
        return json.loads(confidence_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _read_plddt_npz(structure_path: Path) -> list[float] | None:
    """Read per-residue pLDDT from the companion .npz file."""
    try:
        import numpy as np
    except ImportError:
        return None

    stem = structure_path.stem
    parent = structure_path.parent

    plddt_path = parent / f"plddt_{stem}.npz"
    if not plddt_path.exists():
        return None

    try:
        data = np.load(plddt_path)
        # The plddt npz typically has a single array
        for key in data.files:
            arr = data[key]
            if arr.ndim >= 1:
                return arr.flatten().tolist()
    except (OSError, ValueError):
        pass

    return None


def _read_affinity(structure_path: Path) -> dict | None:
    """Read the companion affinity JSON for a structure file (Boltz-2 only).

    Named as ``affinity_<stem>.json``.
    """
    stem = structure_path.stem
    parent = structure_path.parent

    affinity_path = parent / f"affinity_{stem}.json"
    if not affinity_path.exists():
        for candidate in parent.glob("affinity_*.json"):
            if stem in candidate.name:
                affinity_path = candidate
                break
        else:
            return None

    try:
        return json.loads(affinity_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def standardize(workspace: Path) -> None:
    """Transform raw Boltz outputs into standardized result_data.json."""
    raw_dir = workspace / "outputs" / "raw"
    std_dir = workspace / "outputs" / "standardized"
    std_dir.mkdir(parents=True, exist_ok=True)

    pred_dir = _find_predictions(raw_dir)
    structure_files = _find_structure_files(pred_dir)

    # Parse each structure with its confidence and affinity data
    models: list[dict] = []
    for struct_path in structure_files:
        confidence = _read_confidence(struct_path)
        plddt_per_residue = _read_plddt_npz(struct_path)
        affinity = _read_affinity(struct_path)

        # Extract confidence metrics
        confidence_score = None
        ptm = None
        iptm = None
        plddt_mean = None

        if confidence:
            confidence_score = confidence.get("confidence_score")
            ptm = confidence.get("ptm")
            iptm = confidence.get("iptm")
            # complex_plddt in Boltz is a scalar mean, not per-residue
            plddt_mean = confidence.get("complex_plddt")

        # If we have per-residue data but no mean, compute it
        if plddt_mean is None and plddt_per_residue:
            plddt_mean = sum(plddt_per_residue) / len(plddt_per_residue)

        # Extract affinity data
        affinity_probability = None
        affinity_value = None
        if affinity:
            affinity_probability = affinity.get("affinity_probability_binary")
            affinity_value = affinity.get("affinity_pred_value")

        models.append({
            "raw_path": struct_path,
            "confidence_score": confidence_score,
            "ptm": ptm,
            "iptm": iptm,
            "plddt_per_residue": plddt_per_residue,
            "plddt_mean": plddt_mean,
            "affinity_probability": affinity_probability,
            "affinity_value": affinity_value,
        })

    # Rank by confidence_score (highest first), with None values last
    models.sort(
        key=lambda m: (m["confidence_score"] is not None, m["confidence_score"] or 0),
        reverse=True,
    )

    # Copy structures to standardized dir and build result
    structures = []
    for rank, model in enumerate(models, start=1):
        suffix = model["raw_path"].suffix
        dest_name = f"model_{rank}{suffix}"
        dest_path = std_dir / dest_name
        shutil.copy2(model["raw_path"], dest_path)

        entry: dict = {
            "model_rank": rank,
            "structure_path": str(dest_path),
            "plddt_per_residue": model["plddt_per_residue"],
            "plddt_mean": model["plddt_mean"],
            "ptm": model["ptm"],
            "iptm": model["iptm"],
            "chain_mapping": None,
        }

        if model["affinity_probability"] is not None:
            entry["affinity_probability"] = model["affinity_probability"]
        if model["affinity_value"] is not None:
            entry["affinity_value"] = model["affinity_value"]

        structures.append(entry)

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
    parser = argparse.ArgumentParser(description="Standardize Boltz outputs.")
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    standardize(args.workspace)
