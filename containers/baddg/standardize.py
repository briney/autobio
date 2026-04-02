#!/usr/bin/env python3
"""Standardize BA-ddG outputs into autobio ScoringOutput format.

BA-ddG's inference.py writes a CSV with columns:
    mutation, ddg, fold_1, fold_2, ..., fold_N

This script reads the CSV and produces result_data.json in the standard
scoring schema, with per-fold values in the score breakdown.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def standardize(workspace: Path) -> None:
    """Parse BA-ddG CSV output and write result_data.json."""
    raw_dir = workspace / "outputs" / "raw"
    std_dir = workspace / "outputs" / "standardized"
    config = json.loads((workspace / "config.json").read_text())

    mutations_str = config.get("mutations", "")
    mutations_list = [m.strip() for m in mutations_str.split(",") if m.strip()]
    chains = config.get("chains", "")

    # Find the output CSV
    csv_files = sorted(raw_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError(
            f"No CSV files found in {raw_dir}. "
            f"Files present: {[f.name for f in raw_dir.iterdir() if f.is_file()]}"
        )

    with open(csv_files[0], newline="") as fh:
        reader = csv.DictReader(fh)
        columns = reader.fieldnames or []

        # Find fold columns (fold_1, fold_2, ...)
        fold_cols = sorted(
            [c for c in columns if c.startswith("fold_")],
            key=lambda c: int(c.split("_")[1]),
        )
        if not fold_cols:
            raise RuntimeError(
                f"No fold columns (fold_*) found in CSV. Columns: {columns}"
            )

        scores = []
        for row in reader:
            # Extract per-fold predictions
            fold_values = {col: float(row[col]) for col in fold_cols}

            # Primary ddG: mean across folds
            ddg_values = list(fold_values.values())
            ddg_mean = sum(ddg_values) / len(ddg_values)

            # Build score breakdown
            score_breakdown: dict = {
                "chains": chains,
            }
            if len(fold_cols) > 1:
                score_breakdown["fold_values"] = fold_values
                score_breakdown["n_folds"] = len(fold_cols)

            # Use mutations from CSV row if present, else from config
            row_mutations = mutations_list
            csv_mutation = row.get("mutation", "")
            if csv_mutation:
                csv_mutation = csv_mutation.strip()
                if csv_mutation and csv_mutation != "nan":
                    row_mutations = [
                        m.strip() for m in csv_mutation.split(",") if m.strip()
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
