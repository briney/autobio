#!/usr/bin/env python3
"""Standardize Protenix outputs into autobio schema format.

Protenix writes predictions to:
    <output_dir>/<name>/<name>_<seed>_sample_<N>.cif
    <output_dir>/<name>/<name>_<seed>_summary_confidence_sample_<N>.json

Confidence JSON fields:
    - plddt: mean pLDDT
    - ptm: predicted TM-score
    - iptm: interface predicted TM-score
    - ranking_score: composite ranking score
    - has_clash: boolean steric clash flag
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _find_structure_files(raw_dir: Path) -> list[Path]:
    """Find all Protenix mmCIF output files, excluding confidence JSONs."""
    structures = sorted(raw_dir.rglob("*.cif"))
    # Filter out any non-structure files
    structures = [s for s in structures if "_summary_confidence_" not in s.name]
    if not structures:
        all_files = [str(f.relative_to(raw_dir)) for f in raw_dir.rglob("*") if f.is_file()]
        raise RuntimeError(
            f"No Protenix structure files (*.cif) found in {raw_dir}. "
            f"Contents: {all_files}"
        )
    return structures


def _read_confidence(structure_path: Path) -> dict[str, object]:
    """Read the companion confidence JSON for a structure file.

    Protenix naming convention:
        <name>_<seed>_sample_<N>.cif
        <name>_<seed>_summary_confidence_sample_<N>.json
    """
    stem = structure_path.stem  # e.g. prediction_101_sample_0

    # Transform: insert "summary_confidence_" before "sample_"
    if "_sample_" in stem:
        parts = stem.rsplit("_sample_", 1)
        conf_name = f"{parts[0]}_summary_confidence_sample_{parts[1]}.json"
    else:
        conf_name = f"{stem}_summary_confidence.json"

    conf_path = structure_path.parent / conf_name

    if not conf_path.exists():
        # Fallback: search for any confidence JSON with matching prefix
        prefix = stem.split("_sample_")[0] if "_sample_" in stem else stem
        for candidate in structure_path.parent.glob("*summary_confidence*.json"):
            if prefix in candidate.name:
                conf_path = candidate
                break
        else:
            return {}

    try:
        data = json.loads(conf_path.read_text())
        return {
            "plddt": data.get("plddt"),
            "ptm": data.get("ptm"),
            "iptm": data.get("iptm"),
            "ranking_score": data.get("ranking_score"),
            "has_clash": data.get("has_clash"),
        }
    except (OSError, ValueError, KeyError):
        return {}


def standardize(workspace: Path) -> None:
    """Transform raw Protenix outputs into standardized result_data.json."""
    raw_dir = workspace / "outputs" / "raw"
    std_dir = workspace / "outputs" / "standardized"
    std_dir.mkdir(parents=True, exist_ok=True)

    structure_files = _find_structure_files(raw_dir)

    models: list[dict[str, object]] = []
    for struct_path in structure_files:
        conf = _read_confidence(struct_path)
        models.append(
            {
                "raw_path": str(struct_path),
                "ranking_score": conf.get("ranking_score"),
                "plddt_mean": conf.get("plddt"),
                "ptm": conf.get("ptm"),
                "iptm": conf.get("iptm"),
                "has_clash": conf.get("has_clash"),
            }
        )

    # Rank by ranking_score (highest first), fallback to plddt_mean
    def _sort_key(m: dict[str, object]) -> tuple[bool, float]:
        score = m["ranking_score"]
        if score is not None:
            return (True, float(score))  # type: ignore[arg-type]
        plddt = m["plddt_mean"]
        if plddt is not None:
            return (True, float(plddt))  # type: ignore[arg-type]
        return (False, 0.0)

    models.sort(key=_sort_key, reverse=True)

    structures: list[dict[str, object]] = []
    for rank, model in enumerate(models, start=1):
        dest_name = f"model_{rank}.cif"
        dest_path = std_dir / dest_name
        shutil.copy2(model["raw_path"], dest_path)  # type: ignore[arg-type]

        structures.append(
            {
                "model_rank": rank,
                "structure_path": str(dest_path),
                "plddt_per_residue": None,  # Protenix provides mean only
                "plddt_mean": model["plddt_mean"],
                "ptm": model["ptm"],
                "iptm": model["iptm"],
                "chain_mapping": None,
            }
        )

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

    print(f"[protenix] Standardized {len(structures)} structure(s).")
    if best:
        print(
            f"[protenix] Best model: pLDDT={best.get('plddt_mean')}, "
            f"pTM={best.get('ptm')}, ipTM={best.get('iptm')}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardize Protenix outputs.")
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    standardize(args.workspace)
