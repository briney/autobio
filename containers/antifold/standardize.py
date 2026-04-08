#!/usr/bin/env python3
"""Standardize AntiFold raw outputs to schema-compliant result_data.json.

Dispatches on the ``mode`` field in config.json:
- design → InverseFoldingOutput schema
- score  → ScoringOutput schema
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _compute_recovery(designed: str, native: str) -> float:
    """Compute sequence recovery (fraction of matching residues)."""
    if not designed or not native:
        return 0.0
    min_len = min(len(designed), len(native))
    matches = sum(1 for i in range(min_len) if designed[i] == native[i])
    return matches / max(len(designed), len(native))


def standardize_design(workspace: Path) -> None:
    """Convert design_results.json to InverseFoldingOutput schema."""
    raw_dir = workspace / "outputs" / "raw"
    std_dir = workspace / "outputs" / "standardized"
    std_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads((raw_dir / "design_results.json").read_text())
    samples = data["samples"]
    native_sequences = data["native_sequences"]
    scores = data.get("scores", [])
    recoveries = data.get("recoveries", [])

    # Build designed sequences ranked by score (lower NLL is better)
    indexed = list(enumerate(samples))
    if scores:
        # Sort by score ascending (lower = better)
        indexed.sort(key=lambda x: scores[x[0]] if x[0] < len(scores) else float("inf"))

    designed_sequences = []
    for rank, (orig_idx, sample) in enumerate(indexed, start=1):
        # Use AntiFold's reported recovery if available; compute otherwise
        if orig_idx < len(recoveries) and recoveries[orig_idx] is not None:
            recovery = round(recoveries[orig_idx], 4)
        else:
            chain_recoveries = []
            for chain_id, seq in sample.items():
                native = native_sequences.get(chain_id, "")
                if native:
                    chain_recoveries.append(_compute_recovery(seq, native))
            recovery = (
                round(sum(chain_recoveries) / len(chain_recoveries), 4)
                if chain_recoveries
                else None
            )

        score = round(scores[orig_idx], 6) if orig_idx < len(scores) else None

        designed_sequences.append({
            "rank": rank,
            "sequence": sample,
            "score": score,
            "recovery": recovery,
        })

    result_data = {
        "designed_sequences": designed_sequences,
        "native_sequence": native_sequences if native_sequences else None,
    }
    (std_dir / "result_data.json").write_text(json.dumps(result_data, indent=2))


def standardize_score(workspace: Path) -> None:
    """Convert score_results.json to ScoringOutput schema."""
    raw_dir = workspace / "outputs" / "raw"
    std_dir = workspace / "outputs" / "standardized"
    std_dir.mkdir(parents=True, exist_ok=True)

    chain_scores = json.loads((raw_dir / "score_results.json").read_text())

    if not chain_scores:
        result_data = {"scores": []}
        (std_dir / "result_data.json").write_text(json.dumps(result_data, indent=2))
        return

    # Aggregate per-chain scores
    total_ll = sum(s["mean_ll"] for s in chain_scores) / len(chain_scores)

    # Collect per-residue scores across all chains
    all_per_residue: list[float] = []
    for s in chain_scores:
        all_per_residue.extend(s.get("per_residue_ll", []))

    # Build score breakdown
    breakdown: dict[str, float] = {}
    for s in chain_scores:
        chain_id = s["chain_id"]
        breakdown[f"{chain_id}_mean_ll"] = s["mean_ll"]
        breakdown[f"{chain_id}_perplexity"] = s["perplexity"]

    # Overall perplexity
    import math

    breakdown["perplexity"] = round(math.exp(-total_ll), 4) if chain_scores else None

    scores = [{
        "total_score": round(total_ll, 6),
        "per_residue_scores": all_per_residue if all_per_residue else None,
        "score_breakdown": breakdown,
        "units": "avg_nll",
        "structure_path": None,
        "ddg": None,
        "mutations": None,
    }]

    result_data = {"scores": scores}
    (std_dir / "result_data.json").write_text(json.dumps(result_data, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()

    workspace = Path(args.workspace)
    config = json.loads((workspace / "config.json").read_text())

    mode = config["mode"]
    if mode == "design":
        standardize_design(workspace)
    elif mode == "score":
        standardize_score(workspace)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    print(f"[antifold] Standardized {mode} outputs")


if __name__ == "__main__":
    main()
