#!/usr/bin/env python3
"""Standardize OpenMM MD simulation outputs into autobio SimulationOutput format."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def standardize(workspace: Path) -> None:
    """Parse simulation outputs, copy files, and write result_data.json."""
    raw_dir = workspace / "outputs" / "raw"
    std_dir = workspace / "outputs" / "standardized"

    # Parse energy timeseries
    timeseries_path = raw_dir / "energy_timeseries.json"
    if not timeseries_path.exists():
        raise RuntimeError(
            f"No energy_timeseries.json found in {raw_dir}. "
            f"Files present: {[f.name for f in raw_dir.iterdir()]}"
        )
    energy_timeseries = json.loads(timeseries_path.read_text())

    # Parse simulation summary
    summary_path = raw_dir / "simulation_summary.json"
    if not summary_path.exists():
        raise RuntimeError(
            f"No simulation_summary.json found in {raw_dir}. "
            f"Files present: {[f.name for f in raw_dir.iterdir()]}"
        )
    summary = json.loads(summary_path.read_text())

    # Copy trajectory file
    trajectory_files = sorted(
        raw_dir.glob("trajectory.*"),
    )
    if not trajectory_files:
        raise RuntimeError(
            f"No trajectory file found in {raw_dir}. "
            f"Files present: {[f.name for f in raw_dir.iterdir()]}"
        )
    trajectory_src = trajectory_files[0]
    trajectory_dest = std_dir / trajectory_src.name
    shutil.copy2(trajectory_src, trajectory_dest)
    trajectory_path = f"/workspace/outputs/standardized/{trajectory_src.name}"

    # Copy final PDB
    final_structure_path = None
    final_pdb = raw_dir / "final.pdb"
    if final_pdb.exists():
        shutil.copy2(final_pdb, std_dir / "final.pdb")
        final_structure_path = "/workspace/outputs/standardized/final.pdb"

    # Build result_data.json (SimulationOutput format)
    result_data = {
        "trajectory_path": trajectory_path,
        "final_structure_path": final_structure_path,
        "energy_timeseries": energy_timeseries,
        "summary": summary,
    }
    (std_dir / "result_data.json").write_text(json.dumps(result_data, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    standardize(args.workspace)
