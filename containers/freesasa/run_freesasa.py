#!/usr/bin/env python3
"""Run FreeSASA SASA or BSA calculation.

Reads config.json, performs the calculation using the freesasa Python API,
and writes output.json to the raw output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import freesasa


def _build_parameters(config: dict) -> freesasa.Parameters:
    """Build FreeSASA Parameters from config."""
    params = freesasa.Parameters()

    algorithm = config.get("algorithm", "LeeRichards")
    if algorithm == "ShrakeRupley":
        params.setAlgorithm(freesasa.ShrakeRupley)
    else:
        params.setAlgorithm(freesasa.LeeRichards)

    probe_radius = config.get("probe_radius", 1.4)
    params.setProbeRadius(probe_radius)

    return params


def _extract_partner_structure(
    full_structure: freesasa.Structure,
    chain_ids: set[str],
) -> freesasa.Structure:
    """Build a new Structure containing only atoms from the specified chains."""
    partner = freesasa.Structure()
    for i in range(full_structure.nAtoms()):
        if full_structure.chainLabel(i) in chain_ids:
            x, y, z = full_structure.coord(i)
            partner.addAtom(
                full_structure.atomName(i),
                full_structure.residueName(i),
                full_structure.residueNumber(i),
                full_structure.chainLabel(i),
                x,
                y,
                z,
            )
    if partner.nAtoms() == 0:
        raise RuntimeError(
            f"No atoms found for chains {sorted(chain_ids)}. "
            f"Check that chain IDs match those in the PDB file."
        )
    return partner


def _per_chain_sasa(
    result: freesasa.Result,
    structure: freesasa.Structure,
) -> dict[str, float]:
    """Get SASA per chain using selectArea."""
    chains: set[str] = set()
    for i in range(structure.nAtoms()):
        chains.add(structure.chainLabel(i))

    if not chains:
        return {}

    commands = tuple(f"chain_{c}, chain {c}" for c in sorted(chains))
    sel = freesasa.selectArea(commands, structure, result)
    return {c: sel[f"chain_{c}"] for c in sorted(chains)}


def _per_residue_areas(
    result: freesasa.Result,
) -> list[dict[str, object]]:
    """Extract per-residue SASA as a flat list."""
    residues = []
    for chain_id, chain_residues in result.residueAreas().items():
        for resi, area in chain_residues.items():
            residues.append({
                "chain": chain_id,
                "residue_number": resi,
                "residue_type": area.residueType,
                "total": area.total,
                "polar": area.polar,
                "apolar": area.apolar,
                "main_chain": area.mainChain,
                "side_chain": area.sideChain,
            })
    return residues


def run_sasa(config: dict, output_dir: Path) -> None:
    """Calculate SASA of a structure."""
    params = _build_parameters(config)
    structure = freesasa.Structure(config["structure_path"])
    result = freesasa.calc(structure, params)

    classes = freesasa.classifyResults(result, structure)
    chain_sasa = _per_chain_sasa(result, structure)

    output: dict[str, object] = {
        "mode": "sasa",
        "total_sasa": result.totalArea(),
        "polar_sasa": classes.get("Polar", 0.0),
        "apolar_sasa": classes.get("Apolar", 0.0),
        "algorithm": config.get("algorithm", "LeeRichards"),
        "probe_radius": config.get("probe_radius", 1.4),
        "per_chain_sasa": chain_sasa,
    }

    if config.get("per_residue", False):
        output["per_residue_sasa"] = _per_residue_areas(result)

    (output_dir / "output.json").write_text(json.dumps(output, indent=2))


def run_bsa(config: dict, output_dir: Path) -> None:
    """Calculate BSA between two chain partner groups."""
    params = _build_parameters(config)

    # Parse chain groups
    p1_chains = {c.strip() for c in config["partner1"].split(",")}
    p2_chains = {c.strip() for c in config["partner2"].split(",")}

    # Load full complex
    complex_structure = freesasa.Structure(config["structure_path"])
    complex_result = freesasa.calc(complex_structure, params)
    complex_classes = freesasa.classifyResults(complex_result, complex_structure)
    complex_chain_sasa = _per_chain_sasa(complex_result, complex_structure)

    # Calculate SASA for partner 1 alone
    p1_structure = _extract_partner_structure(complex_structure, p1_chains)
    p1_result = freesasa.calc(p1_structure, params)
    p1_classes = freesasa.classifyResults(p1_result, p1_structure)

    # Calculate SASA for partner 2 alone
    p2_structure = _extract_partner_structure(complex_structure, p2_chains)
    p2_result = freesasa.calc(p2_structure, params)
    p2_classes = freesasa.classifyResults(p2_result, p2_structure)

    # BSA = SASA(p1 alone) + SASA(p2 alone) - SASA(complex)
    total_bsa = p1_result.totalArea() + p2_result.totalArea() - complex_result.totalArea()
    polar_bsa = (
        p1_classes.get("Polar", 0.0)
        + p2_classes.get("Polar", 0.0)
        - complex_classes.get("Polar", 0.0)
    )
    apolar_bsa = (
        p1_classes.get("Apolar", 0.0)
        + p2_classes.get("Apolar", 0.0)
        - complex_classes.get("Apolar", 0.0)
    )

    output: dict[str, object] = {
        "mode": "bsa",
        "total_bsa": total_bsa,
        "polar_bsa": polar_bsa,
        "apolar_bsa": apolar_bsa,
        "complex_sasa": complex_result.totalArea(),
        "partner1_sasa": p1_result.totalArea(),
        "partner2_sasa": p2_result.totalArea(),
        "partner1_chains": config["partner1"],
        "partner2_chains": config["partner2"],
        "algorithm": config.get("algorithm", "LeeRichards"),
        "probe_radius": config.get("probe_radius", 1.4),
        "per_chain_sasa": complex_chain_sasa,
    }

    if config.get("per_residue", False):
        # Per-residue BSA: difference between alone and in-complex
        complex_res = {
            (r["chain"], r["residue_number"]): r
            for r in _per_residue_areas(complex_result)
        }
        per_residue_bsa = []
        for partner_result, partner_chains in [
            (p1_result, p1_chains),
            (p2_result, p2_chains),
        ]:
            for r in _per_residue_areas(partner_result):
                key = (r["chain"], r["residue_number"])
                complex_r = complex_res.get(key)
                alone_total = r["total"]
                in_complex_total = complex_r["total"] if complex_r else 0.0
                per_residue_bsa.append({
                    "chain": r["chain"],
                    "residue_number": r["residue_number"],
                    "residue_type": r["residue_type"],
                    "bsa": alone_total - in_complex_total,
                    "sasa_alone": alone_total,
                    "sasa_complex": in_complex_total,
                })
        output["per_residue_bsa"] = per_residue_bsa

    (output_dir / "output.json").write_text(json.dumps(output, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FreeSASA calculation")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mode = config.get("mode", "sasa")
    if mode == "bsa":
        run_bsa(config, args.output_dir)
    else:
        run_sasa(config, args.output_dir)


if __name__ == "__main__":
    main()
