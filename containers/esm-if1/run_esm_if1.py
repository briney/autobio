#!/usr/bin/env python3
"""ESM-IF1 inverse folding and sequence scoring.

Reads config.json from the workspace and dispatches to either:
- design mode: sample sequences from backbone structure
- score mode: compute conditional log-likelihood of sequences
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def load_model() -> tuple:
    """Load ESM-IF1 model and alphabet, moving to GPU if available."""
    import esm

    model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("[esm-if1] Using GPU")
    else:
        print("[esm-if1] Using CPU")
    return model, alphabet


def get_chain_ids(structure_path: str) -> list[str]:
    """Extract unique chain IDs from a PDB file."""
    chains: list[str] = []
    seen: set[str] = set()
    with open(structure_path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                chain_id = line[21]
                if chain_id not in seen and chain_id.strip():
                    seen.add(chain_id)
                    chains.append(chain_id)
    return chains


def run_design(config: dict, workspace: Path) -> None:
    """Sample sequences from backbone structure."""
    from esm.inverse_folding.multichain_util import (
        extract_coords_from_complex,
        load_structure,
        sample_sequence_in_complex,
    )
    from esm.inverse_folding.util import load_coords

    model, alphabet = load_model()

    structure_path = config["structure_path"]
    num_sequences = config.get("num_sequences", 1)
    temperature = config.get("temperature", 0.1)
    chains_to_design = config.get("chains_to_design")
    fixed_positions = config.get("fixed_positions")
    seed = config.get("seed")

    if seed is not None:
        torch.manual_seed(seed)

    # Determine all chains in the structure
    all_chain_ids = get_chain_ids(structure_path)
    is_multichain = len(all_chain_ids) > 1

    if chains_to_design is None:
        chains_to_design = all_chain_ids

    # Extract native sequences for all chains being designed
    native_sequences: dict[str, str] = {}
    for chain_id in all_chain_ids:
        coords, native_seq = load_coords(structure_path, chain_id)
        native_sequences[chain_id] = native_seq

    # Sample sequences
    samples: list[dict[str, str]] = []
    for i in range(num_sequences):
        print(f"[esm-if1] Sampling sequence {i + 1}/{num_sequences}")
        designed: dict[str, str] = {}

        if is_multichain:
            # Multi-chain: use complex-aware sampling for each target chain
            # load_structure returns a biotite structure; extract_coords_from_complex
            # converts it to the dict format sample_sequence_in_complex expects
            structure = load_structure(structure_path)
            # extract_coords_from_complex returns (coords_dict, seqs_dict)
            complex_coords, _ = extract_coords_from_complex(structure)
            for chain_id in chains_to_design:
                seq = sample_sequence_in_complex(
                    model, complex_coords,
                    chain_id, temperature=temperature,
                )
                designed[chain_id] = seq
        else:
            # Single-chain: use simpler API
            # model.sample() handles device placement internally — pass raw coords
            chain_id = chains_to_design[0]
            coords, _ = load_coords(structure_path, chain_id)
            seq = model.sample(coords, temperature=temperature)
            designed[chain_id] = seq

        # Fill in non-designed chains with native sequence
        for chain_id in all_chain_ids:
            if chain_id not in designed:
                designed[chain_id] = native_sequences[chain_id]

        # Post-hoc enforcement of fixed positions
        if fixed_positions:
            for chain_id, positions in fixed_positions.items():
                if chain_id in designed and chain_id in native_sequences:
                    seq_list = list(designed[chain_id])
                    native = native_sequences[chain_id]
                    for pos in positions:
                        # positions are 1-based
                        idx = pos - 1
                        if 0 <= idx < len(seq_list) and idx < len(native):
                            seq_list[idx] = native[idx]
                    designed[chain_id] = "".join(seq_list)

        samples.append(designed)

    # Write raw output
    raw_dir = workspace / "outputs" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "samples": samples,
        "native_sequences": native_sequences,
    }
    (raw_dir / "design_results.json").write_text(json.dumps(result, indent=2))
    print(f"[esm-if1] Wrote {len(samples)} designed sequence(s)")


def run_score(config: dict, workspace: Path) -> None:
    """Score sequences against a structure using conditional log-likelihood."""
    from esm.inverse_folding.util import load_coords, score_sequence

    model, alphabet = load_model()

    structure_path = config["structure_path"]
    sequences = config["sequences"]  # dict[chain_id, aa_sequence]

    scores: list[dict] = []
    for chain_id, seq in sequences.items():
        print(f"[esm-if1] Scoring chain {chain_id} ({len(seq)} residues)")
        # score_sequence handles device placement internally — pass raw coords
        coords, native_seq = load_coords(structure_path, chain_id)
        ll_fullseq, ll_withcoord = score_sequence(
            model, alphabet, coords, seq,
        )
        scores.append({
            "chain_id": chain_id,
            "sequence": seq,
            "native_sequence": native_seq,
            "ll_fullseq": float(ll_fullseq),
            "ll_withcoord": float(ll_withcoord),
        })

    # Write raw output
    raw_dir = workspace / "outputs" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    (raw_dir / "score_results.json").write_text(json.dumps(scores, indent=2))
    print(f"[esm-if1] Scored {len(scores)} chain(s)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, help="Path to workspace directory")
    args = parser.parse_args()

    workspace = Path(args.workspace)
    config = json.loads((workspace / "config.json").read_text())

    mode = config["mode"]
    if mode == "design":
        run_design(config, workspace)
    elif mode == "score":
        run_score(config, workspace)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")


if __name__ == "__main__":
    main()
