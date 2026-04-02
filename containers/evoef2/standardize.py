#!/usr/bin/env python3
"""Standardize EvoEF2 outputs into autobio ScoringOutput format."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Pattern for EvoEF2 energy term lines: "  term_name =   -12.345"
_ENERGY_LINE_RE = re.compile(r"^\s*([\w]+)\s*=\s*([-+]?\d+\.?\d*)\s*$")


def _parse_evoef2_energy(text: str) -> tuple[float, dict[str, float]]:
    """Parse EvoEF2 energy output text.

    Returns:
        Tuple of (total_score, score_breakdown dict).
    """
    breakdown: dict[str, float] = {}
    total_score = 0.0

    for line in text.splitlines():
        m = _ENERGY_LINE_RE.match(line)
        if m:
            term_name = m.group(1)
            value = float(m.group(2))
            if term_name == "Total":
                total_score = value
            else:
                breakdown[term_name] = value

    return total_score, breakdown


def standardize(workspace: Path) -> None:
    """Parse EvoEF2 raw outputs and write result_data.json."""
    raw_dir = workspace / "outputs" / "raw"
    std_dir = workspace / "outputs" / "standardized"
    logs_dir = workspace / "logs"
    config = json.loads((workspace / "config.json").read_text())

    command = config["command"]
    scores: list[dict] = []

    if command == "RepairStructure":
        # Parse energy from tool log (EvoEF2 prints energy terms to stdout)
        energy_text = ""
        tool_log = logs_dir / "tool.log"
        if tool_log.exists():
            energy_text = tool_log.read_text()
        total_score, breakdown = _parse_evoef2_energy(energy_text)

        # Find repaired PDB
        repair_pdbs = sorted(raw_dir.glob("*_Repair.pdb"))
        structure_path = None
        if repair_pdbs:
            structure_path = f"/workspace/outputs/raw/{repair_pdbs[0].name}"

        scores.append({
            "total_score": total_score,
            "score_breakdown": breakdown if breakdown else None,
            "units": "EvoEF2",
            "per_residue_scores": None,
            "structure_path": structure_path,
            "ddg": None,
            "mutations": None,
        })

    elif command == "ComputeBinding":
        # Parse binding energy from captured stdout file
        binding_file = raw_dir / "binding_output.txt"
        if not binding_file.exists():
            print(
                "ERROR: binding_output.txt not found in raw output directory.",
                file=sys.stderr,
            )
            raise RuntimeError(
                f"binding_output.txt not found in {raw_dir}. "
                f"Files present: {[f.name for f in raw_dir.iterdir()]}"
            )
        energy_text = binding_file.read_text()
        total_score, breakdown = _parse_evoef2_energy(energy_text)

        # Check for repaired structure (auto-repair writes it to raw/)
        repair = config.get("repair", True)
        structure_path = None
        if repair:
            repair_pdbs = sorted(raw_dir.glob("*_Repair.pdb"))
            if repair_pdbs:
                structure_path = f"/workspace/outputs/raw/{repair_pdbs[0].name}"

        scores.append({
            "total_score": total_score,
            "score_breakdown": breakdown if breakdown else None,
            "units": "EvoEF2",
            "per_residue_scores": None,
            "structure_path": structure_path,
            "ddg": None,
            "mutations": None,
        })

    elif command == "BuildMutant":
        mutations = config.get("mutations", [])

        # Parse energy from log (if available)
        energy_text = ""
        tool_log = logs_dir / "tool.log"
        if tool_log.exists():
            energy_text = tool_log.read_text()
        total_score, breakdown = _parse_evoef2_energy(energy_text)

        # Find all model PDBs
        model_pdbs = sorted(raw_dir.glob("*_Model_*.pdb"))
        if not model_pdbs:
            print(
                "ERROR: No Model PDB files found in raw output directory.",
                file=sys.stderr,
            )
            raise RuntimeError(
                f"No Model PDB files found in {raw_dir}. "
                f"Files present: {[f.name for f in raw_dir.iterdir()]}"
            )

        for pdb in model_pdbs:
            scores.append({
                "total_score": total_score,
                "score_breakdown": breakdown if breakdown else None,
                "units": "EvoEF2",
                "per_residue_scores": None,
                "structure_path": f"/workspace/outputs/raw/{pdb.name}",
                "ddg": None,
                "mutations": mutations,
            })

    else:
        raise RuntimeError(f"Unknown EvoEF2 command: {command!r}")

    result_data = {"scores": scores}
    (std_dir / "result_data.json").write_text(json.dumps(result_data, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    standardize(args.workspace)
