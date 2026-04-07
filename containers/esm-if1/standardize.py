#!/usr/bin/env python3
"""Standardize ESM-IF1 raw outputs to schema-compliant result_data.json.

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

    designed_sequences = []
    for rank, sample in enumerate(samples, start=1):
        # Compute average recovery across all chains
        recoveries = []
        for chain_id, seq in sample.items():
            native = native_sequences.get(chain_id, "")
            if native:
                recoveries.append(_compute_recovery(seq, native))

        avg_recovery = sum(recoveries) / len(recoveries) if recoveries else None

        designed_sequences.append({
            "rank": rank,
            "sequence": sample,
            "score": None,
            "recovery": round(avg_recovery, 4) if avg_recovery is not None else None,
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

    # Aggregate per-chain scores into a single ScoredStructure
    total_ll = sum(s["ll_fullseq"] for s in chain_scores) / len(chain_scores)
    breakdown = {}
    for s in chain_scores:
        chain_id = s["chain_id"]
        breakdown[f"{chain_id}_ll_fullseq"] = s["ll_fullseq"]
        breakdown[f"{chain_id}_ll_withcoord"] = s["ll_withcoord"]

    # If single chain, also put the plain keys
    if len(chain_scores) == 1:
        breakdown["ll_fullseq"] = chain_scores[0]["ll_fullseq"]
        breakdown["ll_withcoord"] = chain_scores[0]["ll_withcoord"]

    scores = [{
        "total_score": round(total_ll, 6),
        "per_residue_scores": None,
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

    print(f"[esm-if1] Standardized {mode} outputs")


if __name__ == "__main__":
    main()
