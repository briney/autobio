#!/usr/bin/env python3
"""Score protein sequences against backbone structures using ProteinMPNN/LigandMPNN.

Reads config.json from the workspace and computes conditional log-likelihoods
for the given (or native) sequences using the ProteinMPNN/LigandMPNN model.

Uses the single-amino-acid scoring method: p(AA_i | backbone, all other AAs),
which gives the conditional probability of each residue given its structural
context and all other residues in the sequence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

# Add LigandMPNN to Python path
sys.path.insert(0, "/app/LigandMPNN")

from data_utils import (  # noqa: E402
    featurize,
    parse_PDB,
    restype_int_to_str,
    restype_str_to_int,
)
from model_utils import ProteinMPNN  # noqa: E402


def load_model(
    model_type: str,
    checkpoint_path: str,
    device: torch.device,
) -> ProteinMPNN:
    """Load a ProteinMPNN or LigandMPNN model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract hyperparams from checkpoint (following score.py pattern)
    if model_type == "ligand_mpnn":
        atom_context_num = checkpoint["atom_context_num"]
        ligand_mpnn_use_side_chain_context = 0
    else:
        atom_context_num = 1
        ligand_mpnn_use_side_chain_context = 0
    k_neighbors = checkpoint["num_edges"]

    model = ProteinMPNN(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        k_neighbors=k_neighbors,
        device=device,
        atom_context_num=atom_context_num,
        model_type=model_type,
        ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"[mpnn-score] Loaded {model_type} model from {checkpoint_path}")
    print(f"[mpnn-score] k_neighbors={k_neighbors}, atom_context_num={atom_context_num}")
    return model


def thread_sequences(
    protein_dict: dict,
    sequences: dict[str, str],
) -> None:
    """Replace the sequence tensor S with user-provided sequences.

    Modifies protein_dict["S"] in place.

    Args:
        protein_dict: Parsed structure dict from parse_PDB.
        sequences: Mapping of chain_id to amino acid sequence.

    Raises:
        ValueError: If a chain ID is not in the structure or sequence length
            does not match the chain's residue count.
    """
    chain_letters = protein_dict["chain_letters"]
    S = protein_dict["S"].clone()

    # Group residue indices by chain
    chain_positions: dict[str, list[int]] = {}
    for i, ch in enumerate(chain_letters):
        chain_positions.setdefault(ch, []).append(i)

    for chain_id, seq in sequences.items():
        if chain_id not in chain_positions:
            raise ValueError(
                f"Chain '{chain_id}' not found in structure. "
                f"Available chains: {sorted(chain_positions)}"
            )
        positions = chain_positions[chain_id]
        if len(seq) != len(positions):
            raise ValueError(
                f"Sequence length mismatch for chain '{chain_id}': "
                f"provided {len(seq)} residues but structure has {len(positions)}"
            )
        for pos, aa in zip(positions, seq):
            idx = restype_str_to_int.get(aa)
            if idx is None:
                raise ValueError(
                    f"Unknown amino acid '{aa}' in chain '{chain_id}' sequence"
                )
            S[pos] = idx

    protein_dict["S"] = S


def run_score(config: dict, workspace: Path) -> None:
    """Score sequences against a structure using conditional log-likelihood."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mpnn-score] Using device: {device}")

    model_type = config["model_type"]
    checkpoint_path = config["checkpoint_path"]
    structure_path = config["structure_path"]
    sequences = config.get("sequences")

    # Load model
    model = load_model(model_type, checkpoint_path, device)

    # Parse structure
    is_ligand_mpnn = model_type == "ligand_mpnn"
    protein_dict, _, _, icodes, _ = parse_PDB(
        structure_path, device=device, parse_all_atoms=is_ligand_mpnn,
    )

    chain_letters = protein_dict["chain_letters"]
    num_residues = protein_dict["mask"].shape[0]

    # Extract native sequence before potential threading
    native_S = protein_dict["S"].clone()
    chain_positions: dict[str, list[int]] = {}
    for i, ch in enumerate(chain_letters):
        chain_positions.setdefault(ch, []).append(i)

    native_seqs: dict[str, str] = {}
    for ch, positions in chain_positions.items():
        native_seqs[ch] = "".join(
            restype_int_to_str[native_S[pos].item()] for pos in positions
        )

    # Thread user sequences if provided
    if sequences is not None:
        thread_sequences(protein_dict, sequences)

    # Set chain_mask: all residues scored (1 = designable/scored)
    protein_dict["chain_mask"] = torch.ones(num_residues, dtype=torch.int32, device=device)

    # Featurize for model (following score.py pattern)
    feature_dict = featurize(
        protein_dict,
        cutoff_for_score=8.0,
        use_atom_context=1 if is_ligand_mpnn else 0,
        number_of_ligand_atoms=16 if is_ligand_mpnn else 1,
        model_type=model_type,
    )
    feature_dict["batch_size"] = 1

    B, L, _, _ = feature_dict["X"].shape
    feature_dict["symmetry_residues"] = [[]]

    # Score using single-AA scoring: p(AA_i | backbone, all other AAs)
    with torch.no_grad():
        feature_dict["randn"] = torch.randn(
            [feature_dict["batch_size"], feature_dict["mask"].shape[1]],
            device=device,
        )
        score_dict = model.single_aa_score(feature_dict, use_sequence=1)

    log_probs = score_dict["log_probs"]  # (B, L, 21)
    mask = feature_dict["mask"][0].cpu().numpy()  # (L,)
    S = feature_dict["S"][0]  # (L,) — batched S from featurize

    # Per-residue NLL: -log_prob of the actual amino acid at each position
    per_residue_nll: list[tuple[int, str, float]] = []
    for i in range(L):
        if mask[i] > 0.5:
            nll = -log_probs[0, i, S[i]].item()
            per_residue_nll.append((i, chain_letters[i], nll))

    # Group by chain
    chain_scores: list[dict] = []
    for chain_id in sorted(chain_positions):
        chain_nll = [nll for _, ch, nll in per_residue_nll if ch == chain_id]

        if not chain_nll:
            continue

        scored_seq = (
            sequences[chain_id]
            if sequences and chain_id in sequences
            else native_seqs[chain_id]
        )

        chain_scores.append({
            "chain_id": chain_id,
            "sequence": scored_seq,
            "native_sequence": native_seqs[chain_id],
            "mean_nll": float(np.mean(chain_nll)),
            "perplexity": float(math.exp(np.mean(chain_nll))),
            "per_residue_nll": [round(v, 6) for v in chain_nll],
        })

    # Overall score
    all_nll = [nll for _, _, nll in per_residue_nll]
    overall_mean_nll = float(np.mean(all_nll)) if all_nll else 0.0

    # Write raw output
    raw_dir = workspace / "outputs" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "chain_scores": chain_scores,
        "overall_mean_nll": round(overall_mean_nll, 6),
        "model_type": model_type,
    }
    (raw_dir / "score_results.json").write_text(json.dumps(result, indent=2))

    print(f"[mpnn-score] Scored {len(chain_scores)} chain(s), overall NLL: {overall_mean_nll:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, help="Path to workspace directory")
    args = parser.parse_args()

    workspace = Path(args.workspace)
    config = json.loads((workspace / "config.json").read_text())

    run_score(config, workspace)


if __name__ == "__main__":
    main()
