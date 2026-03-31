#!/usr/bin/env python3
"""Standardize StaB-ddG outputs into autobio ScoringOutput format.

StaB-ddG writes a CSV file with columns: Name, Mutation, pred_1, ..., pred_N.
Each pred column corresponds to a trial. This script reads the CSV, extracts
ddG predictions, and writes result_data.json in the standard scoring schema.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import csv


def standardize(workspace: Path) -> None:
    """Parse StaB-ddG CSV output and write result_data.json."""
    raw_dir = workspace / "outputs" / "raw"
    std_dir = workspace / "outputs" / "standardized"
    config = json.loads((workspace / "config.json").read_text())

    mutations_str = config.get("mutations", "")
    mutations_list = [m.strip() for m in mutations_str.split(",") if m.strip()]
    chains = config.get("chains", "")

    # Find the output CSV — StaB-ddG writes to output_dir/output.csv
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(
            f"No CSV files found in {raw_dir}. "
            f"Files present: {[f.name for f in raw_dir.iterdir() if f.is_file()]}"
        )

    with open(csv_files[0], newline="") as fh:
        reader = csv.DictReader(fh)
        columns = reader.fieldnames or []

        # Find prediction columns (pred_1, pred_2, ...)
        pred_cols = sorted(
            [c for c in columns if c.startswith("pred_")],
            key=lambda c: int(c.split("_")[1]),
        )
        if not pred_cols:
            raise RuntimeError(
                f"No prediction columns (pred_*) found in CSV. Columns: {columns}"
            )

        scores = []
        for row in reader:
            # Extract per-trial predictions
            trial_values = {col: float(row[col]) for col in pred_cols}

            # Primary ddG: mean across trials
            ddg_values = list(trial_values.values())
            ddg_mean = sum(ddg_values) / len(ddg_values)

            # Build score breakdown
            score_breakdown: dict = {
                "chains": chains,
            }
            if len(pred_cols) > 1:
                score_breakdown["trial_values"] = trial_values
                score_breakdown["n_trials"] = len(pred_cols)

            # Include mutation string from CSV if present.
            # StaB-ddG writes Mutation column as Python list repr (e.g., "['EA63Q']")
            # so we need to clean brackets and quotes before splitting.
            row_mutations = mutations_list
            csv_mutation = row.get("Mutation", "")
            if csv_mutation:
                csv_mutation = csv_mutation.strip()
                if csv_mutation and csv_mutation != "nan":
                    # Strip Python list repr brackets and quotes
                    cleaned = csv_mutation.strip("[]").replace("'", "").replace('"', "")
                    row_mutations = [
                        m.strip() for m in cleaned.split(",") if m.strip()
                    ]

            scores.append(
                {
                    "total_score": ddg_mean,
                    "score_breakdown": score_breakdown,
                    "units": "kcal/mol",
                    "per_residue_scores": None,
                    "structure_path": None,
                    "ddg": ddg_mean,
                    "mutations": row_mutations,
                }
            )

    result_data = {"scores": scores}
    (std_dir / "result_data.json").write_text(json.dumps(result_data, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    standardize(args.workspace)
