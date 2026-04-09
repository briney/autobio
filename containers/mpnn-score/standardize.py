#!/usr/bin/env python3
"""Standardize ProteinMPNN/LigandMPNN scoring outputs to schema-compliant result_data.json."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def standardize(workspace: Path) -> None:
    """Convert score_results.json to ScoringOutput schema."""
    raw_dir = workspace / "outputs" / "raw"
    std_dir = workspace / "outputs" / "standardized"
    std_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads((raw_dir / "score_results.json").read_text())
    chain_scores = data["chain_scores"]
    overall_mean_nll = data["overall_mean_nll"]

    # Build per-chain breakdown
    breakdown: dict[str, float] = {}
    for cs in chain_scores:
        chain_id = cs["chain_id"]
        breakdown[f"{chain_id}_mean_nll"] = round(cs["mean_nll"], 6)
        breakdown[f"{chain_id}_perplexity"] = round(cs["perplexity"], 6)

    # Overall perplexity
    breakdown["perplexity"] = round(math.exp(overall_mean_nll), 6)

    # If single chain, also put plain keys
    if len(chain_scores) == 1:
        breakdown["mean_nll"] = round(chain_scores[0]["mean_nll"], 6)

    # Flatten per-residue NLL across all chains (sorted by chain ID)
    per_residue_scores: list[float] = []
    for cs in chain_scores:
        per_residue_scores.extend(cs["per_residue_nll"])

    scores = [{
        "total_score": round(overall_mean_nll, 6),
        "per_residue_scores": per_residue_scores if per_residue_scores else None,
        "score_breakdown": breakdown,
        "units": "avg_nll",
        "structure_path": None,
        "ddg": None,
        "mutations": None,
    }]

    result_data = {"scores": scores}
    (std_dir / "result_data.json").write_text(json.dumps(result_data, indent=2))
    print(f"[mpnn-score] Standardized scoring outputs ({len(chain_scores)} chain(s))")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardize MPNN scoring outputs.")
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    standardize(args.workspace)
