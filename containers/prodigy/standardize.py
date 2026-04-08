#!/usr/bin/env python3
"""Standardize PRODIGY outputs into autobio ProteinBindingAffinityOutput format.

run_prodigy.py writes output.json with delta-G, Kd, contact counts, and NIS
percentages. This script maps that JSON into the standard protein binding
affinity schema format.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def standardize(workspace: Path) -> None:
    """Parse PRODIGY output JSON and write result_data.json."""
    raw_dir = workspace / "outputs" / "raw"
    std_dir = workspace / "outputs" / "standardized"
    config = json.loads((workspace / "config.json").read_text())

    output_file = raw_dir / "output.json"
    if not output_file.exists():
        raise RuntimeError(
            f"output.json not found in {raw_dir}. "
            f"Files present: {[f.name for f in raw_dir.iterdir() if f.is_file()]}"
        )

    data = json.loads(output_file.read_text())

    score_breakdown: dict[str, object] = {
        "intermolecular_contacts": data["intermolecular_contacts"],
        "charged_charged_contacts": data["charged_charged_contacts"],
        "charged_polar_contacts": data["charged_polar_contacts"],
        "charged_apolar_contacts": data["charged_apolar_contacts"],
        "polar_polar_contacts": data["polar_polar_contacts"],
        "polar_apolar_contacts": data["polar_apolar_contacts"],
        "apolar_apolar_contacts": data["apolar_apolar_contacts"],
        "pct_apolar_nis": data["pct_apolar_nis"],
        "pct_charged_nis": data["pct_charged_nis"],
        "chain_selection": config.get("selection"),
        "temperature_celsius": data.get("temperature", config.get("temperature", 25.0)),
        "distance_cutoff_angstrom": data.get(
            "distance_cutoff", config.get("distance_cutoff", 5.5)
        ),
        "n_chains": data.get("n_chains"),
        "n_residues": data.get("n_residues"),
        "structure": data.get("structure"),
    }

    # Include contact list if present
    if "contact_list" in data:
        score_breakdown["contact_list"] = data["contact_list"]

    prediction = {
        "delta_g_kcal_mol": data["delta_g"],
        "kd_molar": data["kd"],
        "units": "kcal/mol",
        "score_breakdown": score_breakdown,
    }

    result_data = {"predictions": [prediction]}
    (std_dir / "result_data.json").write_text(json.dumps(result_data, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    standardize(args.workspace)
