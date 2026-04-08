#!/usr/bin/env python3
"""Run PRODIGY binding affinity prediction via its Python API.

Reads config.json, parses the structure with BioPython, runs PRODIGY's
contact-based prediction, and writes structured JSON output. This avoids
fragile text-output parsing by using the PRODIGY API directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from prodigy_prot.modules.parsers import parse_structure
from prodigy_prot.modules.prodigy import Prodigy


def run(config_path: Path, output_dir: Path) -> None:
    """Execute PRODIGY prediction and write output.json."""
    config = json.loads(config_path.read_text())

    structure_path = config["structure_path"]
    selection = config.get("selection")  # None or flat string like "A B" or "A,B C"
    temperature = config.get("temperature", 25.0)
    distance_cutoff = config.get("distance_cutoff", 5.5)
    contact_list = config.get("contact_list", False)

    # Parse selection string into list format expected by Prodigy.
    # PRODIGY expects: ["A", "B"] for chains A vs B, or ["A,B", "C"] for
    # A+B as one partner against C. The config stores this as a flat string
    # "A B" or "A,B C" with space-separated groups.
    selection_list: list[str] | None = None
    if selection:
        selection_list = selection.strip().split()

    # Parse structure — returns (list[Model], n_chains, n_residues).
    # We process only the first model.
    models, n_chains, n_residues = parse_structure(structure_path)
    if not models:
        raise RuntimeError(f"No models found in structure: {structure_path}")

    model = models[0]

    # Run prediction
    name = Path(structure_path).stem
    prodigy = Prodigy(model, name=name, selection=selection_list, temp=temperature)
    prodigy.predict(distance_cutoff=distance_cutoff)

    # Build output
    result = prodigy.as_dict()
    output = {
        "delta_g": result["ba_val"],
        "kd": result["kd_val"],
        "intermolecular_contacts": result["ICs"],
        "charged_charged_contacts": result["CC"],
        "charged_polar_contacts": result["CP"],
        "charged_apolar_contacts": result["AC"],
        "polar_polar_contacts": result["PP"],
        "polar_apolar_contacts": result["AP"],
        "apolar_apolar_contacts": result["AA"],
        "pct_apolar_nis": result["nis_a"],
        "pct_charged_nis": result["nis_c"],
        "selection": result["selection"],
        "temperature": temperature,
        "distance_cutoff": distance_cutoff,
        "n_chains": n_chains,
        "n_residues": n_residues,
        "structure": name,
    }

    # Optional: write contact list
    if contact_list:
        contacts = []
        for res1, res2 in prodigy.ic_network:
            contacts.append({
                "res1_name": res1.resname,
                "res1_id": res1.id[1],
                "res1_chain": res1.parent.id,
                "res2_name": res2.resname,
                "res2_id": res2.id[1],
                "res2_chain": res2.parent.id,
            })
        output["contact_list"] = contacts

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "output.json").write_text(json.dumps(output, indent=2))
    print(f"[prodigy] Predicted delta-G: {output['delta_g']:.1f} kcal/mol")
    print(f"[prodigy] Predicted Kd: {output['kd']:.1e} M at {temperature:.1f} C")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run PRODIGY prediction")
    parser.add_argument("--config", type=Path, required=True, help="Path to config.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    args = parser.parse_args()
    run(args.config, args.output_dir)
