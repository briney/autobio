#!/usr/bin/env python3
"""Custom inference wrapper for BA-ddG binding ddG prediction.

The upstream BA-ddG code (https://github.com/aim-uofa/BA-DDG) is hardcoded
for SKEMPI v2 benchmarking. This script provides a generic single-PDB
inference interface that:

1. Parses an arbitrary PDB with BioPython
2. Applies user-specified mutations
3. Constructs the complex + unbound batch for the thermodynamic cycle
4. Runs DDGPredictor across CV folds and averages predictions
5. Writes a CSV with per-fold and mean ddG values

The thermodynamic cycle computes:
    ddG = beta * [logP(mut|complex) - logP(wt|complex)]
          - beta * [logP(mut|unbound) - logP(wt|unbound)]
where beta is the learned Boltzmann scalar.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import index_to_one, one_to_index
from easydict import EasyDict

# BA-ddG imports (available via PYTHONPATH=/app/baddg/training)
from ddg_predictor import DDGPredictor
from trainer import recursive_to

# parse_biopython_structure is in common_utils/protein/parsers.py
from common_utils.protein.parsers import parse_biopython_structure


# ---------------------------------------------------------------------------
# Model configuration (fixed for BA-ddG architecture)
# ---------------------------------------------------------------------------

_MODEL_CFG = EasyDict(
    ca_only=False,
    hidden_dim=128,
    num_layers=3,
    backbone_noise=0.0,
    num_edges=48,
    loss_weight_boltzmann=1.0,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_mutation(mut_str: str) -> dict:
    """Parse a mutation string like 'EA63Q' -> {wt, chain, resseq, mt}."""
    return {
        "wt": mut_str[0],
        "mt": mut_str[-1],
        "chain": mut_str[1],
        "resseq": int(mut_str[2:-1]),
    }


def reset_residue_idx(res_nb: torch.Tensor) -> torch.Tensor:
    """Add inter-chain offsets to sequential residue numbering.

    Matches the implementation in ``training/dataset.py``.
    """
    reset_points = (res_nb == 1).nonzero(as_tuple=True)[0][1:]
    offsets = torch.zeros_like(res_nb)
    if len(reset_points) > 0:
        offsets[reset_points] = 100 + res_nb[reset_points - 1]
    offsets = torch.cumsum(offsets, dim=0)
    return res_nb + offsets


def pad_tensor(
    tensor: torch.Tensor, length: int, value: int | float = 0
) -> torch.Tensor:
    """Pad the first dimension of *tensor* to *length*."""
    n_pad = length - tensor.shape[0]
    if n_pad <= 0:
        return tensor
    pad_shape = [n_pad] + list(tensor.shape[1:])
    pad = torch.full(pad_shape, fill_value=value, dtype=tensor.dtype)
    return torch.cat([tensor, pad], dim=0)


# ---------------------------------------------------------------------------
# Batch construction
# ---------------------------------------------------------------------------


def prepare_batch(
    data: dict,
    seq_map: dict,
    mutations: list[dict],
    device: torch.device,
) -> dict:
    """Build the batch dict expected by ``DDGPredictor.forward()``.

    The batch contains:
    - One **complex** entry (all chains, full structure)
    - One **single-chain** entry per chain that contains a mutation

    This mirrors ``MPNNPaddingCollate.__call__()`` from the upstream code.
    """
    # -- Apply mutations to the amino acid sequence -------------------------
    aa_mut = data["aa"].clone()
    for mut in mutations:
        key = (mut["chain"], mut["resseq"])
        if key not in seq_map:
            raise ValueError(
                f"Residue {mut['chain']}{mut['resseq']} not found in structure. "
                f"Available chains/residues: check PDB file."
            )
        expected_aa = index_to_one(data["aa"][seq_map[key]].item())
        if expected_aa != mut["wt"]:
            print(
                f"[baddg] WARNING: Expected {mut['wt']} at "
                f"{mut['chain']}{mut['resseq']}, found {expected_aa}. "
                f"Proceeding with the structure's residue.",
                file=sys.stderr,
            )
        aa_mut[seq_map[key]] = one_to_index(mut["mt"])

    mut_flag = data["aa"] != aa_mut

    # -- Extract backbone atoms (N, CA, C, O) from heavy atom coords -------
    X = data["pos_heavyatom"][:, :4, :]  # (L, 4, 3)
    L = X.shape[0]
    max_length = math.ceil(L / 8) * 8

    # -- Pad all tensors to max_length --------------------------------------
    chain_enc = data["chain_nb"] + 1
    res_idx = reset_residue_idx(data["res_nb"])

    X_pad = pad_tensor(X, max_length)
    aa_pad = pad_tensor(data["aa"], max_length)
    aa_mut_pad = pad_tensor(aa_mut, max_length)
    mask_pad = pad_tensor(torch.ones(L, dtype=data["aa"].dtype), max_length)
    chain_M_pad = pad_tensor(mut_flag.long(), max_length)
    chain_enc_pad = pad_tensor(chain_enc, max_length)
    res_idx_pad = pad_tensor(res_idx, max_length, value=-100)

    # -- Complex entry (all chains visible) ---------------------------------
    X_batch = [X_pad]
    aa_batch = [aa_pad]
    aa_mut_batch = [aa_mut_pad]
    mask_batch = [mask_pad]
    chain_M_batch = [chain_M_pad]
    chain_enc_batch = [chain_enc_pad]
    res_idx_batch = [res_idx_pad]
    ddG_batch = [torch.tensor(0.0)]

    # -- Per-mutated-chain entries (unbound state) --------------------------
    mut_indices = torch.nonzero(mut_flag, as_tuple=True)[0]
    # chain_enc values at mutated positions (use un-padded chain_enc)
    mut_chain_ids = torch.unique(chain_enc[mut_indices])

    for chain_idx in mut_chain_ids.tolist():
        chain_mask = chain_enc_pad == chain_idx  # bool mask over padded length
        X_batch.append(X_pad * chain_mask[:, None, None].float())
        aa_batch.append(aa_pad * chain_mask.long())
        aa_mut_batch.append(aa_mut_pad * chain_mask.long())
        mask_batch.append(chain_mask.long())
        chain_M_batch.append(chain_M_pad * chain_mask.long())
        chain_enc_batch.append(chain_enc_pad.clone())
        res_idx_batch.append(res_idx_pad.clone())
        ddG_batch.append(torch.tensor(0.0))

    # -- Stack into batch tensors -------------------------------------------
    batch = {
        "X": torch.stack(X_batch),
        "aa": torch.stack(aa_batch),
        "aa_mut": torch.stack(aa_mut_batch),
        "mask": torch.stack(mask_batch),
        "chain_M": torch.stack(chain_M_batch),
        "chain_encoding_all": torch.stack(chain_enc_batch),
        "residue_idx": torch.stack(res_idx_batch),
        "ddG": torch.stack(ddG_batch),
        "num_mut_chains": [mut_chain_ids.shape[0]],
    }

    return recursive_to(batch, device)


# ---------------------------------------------------------------------------
# Main inference
# ---------------------------------------------------------------------------


def run_inference(args: argparse.Namespace) -> None:
    """Run BA-ddG inference and write output CSV."""
    # -- Device -------------------------------------------------------------
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[baddg] Device: {device}")

    # -- Seed ---------------------------------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # -- Parse PDB ----------------------------------------------------------
    print(f"[baddg] Parsing PDB: {args.pdb_path}")
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("input", args.pdb_path)
    model = structure[0]

    # Split chains into partners.  The upstream SKEMPI convention is:
    #   complex = "PDB_partner1chains_partner2chains"
    #   antigen_chain_id = list(partner1)
    #   antibody_chain_id = list(partner2)
    partner1, partner2 = args.chains.split("_")
    data, seq_map = parse_biopython_structure(
        model,
        antigen_chain_id=list(partner1),
        antibody_chain_id=list(partner2),
    )

    # -- Parse and verify mutations -----------------------------------------
    mutation_strs = [m.strip() for m in args.mutations.split(",") if m.strip()]
    mutations = [parse_mutation(m) for m in mutation_strs]
    print(f"[baddg] Mutations: {mutation_strs}")
    print(f"[baddg] Chains: {args.chains}")

    # -- Prepare batch ------------------------------------------------------
    batch = prepare_batch(data, seq_map, mutations, device)

    # -- Load checkpoints and run per-fold ----------------------------------
    print(f"[baddg] Loading MPNN backbone: {args.mpnn_checkpoint}")
    mpnn_ckpt = torch.load(args.mpnn_checkpoint, map_location="cpu", weights_only=False)

    print(f"[baddg] Loading BA-ddG weights: {args.ddg_checkpoint}")
    ddg_ckpt = torch.load(args.ddg_checkpoint, map_location="cpu", weights_only=False)

    n_available = len(ddg_ckpt["models"])
    n_folds = min(args.n_folds, n_available)
    if n_folds < args.n_folds:
        print(
            f"[baddg] WARNING: Requested {args.n_folds} folds but checkpoint "
            f"has {n_available}. Using {n_folds}.",
            file=sys.stderr,
        )

    fold_predictions: list[float] = []

    for fold_idx in range(n_folds):
        print(f"[baddg] Running fold {fold_idx + 1}/{n_folds}...")
        ddg_model = DDGPredictor(_MODEL_CFG)

        # Load MPNN backbone first, then fine-tuned weights override
        ddg_model.mpnn.load_state_dict(
            mpnn_ckpt["model_state_dict"], strict=False
        )
        ddg_model.load_state_dict(ddg_ckpt["models"][fold_idx], strict=False)

        ddg_model.to(device)
        ddg_model.eval()

        with torch.no_grad():
            _, output_dict, _ = ddg_model(batch)
            ddg_pred = output_dict["ddG_pred"].item()

        fold_predictions.append(ddg_pred)
        ddg_model.to("cpu")
        torch.cuda.empty_cache()

    # -- Compute mean ddG ---------------------------------------------------
    ddg_mean = sum(fold_predictions) / len(fold_predictions)
    print(f"[baddg] ddG prediction: {ddg_mean:.4f} kcal/mol")

    # -- Write output CSV ---------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "output.csv"

    fieldnames = ["mutation", "ddg"] + [f"fold_{i + 1}" for i in range(n_folds)]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        row: dict[str, str] = {
            "mutation": args.mutations,
            "ddg": f"{ddg_mean:.6f}",
        }
        for i, pred in enumerate(fold_predictions):
            row[f"fold_{i + 1}"] = f"{pred:.6f}"
        writer.writerow(row)

    print(f"[baddg] Results written to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BA-ddG binding ddG prediction (single-PDB inference)"
    )
    parser.add_argument("--pdb_path", required=True, help="Path to input PDB file")
    parser.add_argument(
        "--mutations",
        required=True,
        help="Comma-separated mutations: WT_AA CHAIN RESNUM MUT_AA (e.g., EA63Q,KA66A)",
    )
    parser.add_argument(
        "--chains",
        required=True,
        help="Partner chains in 'binder1_binder2' format (e.g., AB_C)",
    )
    parser.add_argument("--mpnn_checkpoint", required=True, help="ProteinMPNN backbone weights")
    parser.add_argument("--ddg_checkpoint", required=True, help="Fine-tuned BA-ddG weights")
    parser.add_argument("--output_dir", required=True, help="Directory for output CSV")
    parser.add_argument("--device", default="auto", help="Device: auto, cuda, or cpu")
    parser.add_argument("--n_folds", type=int, default=3, help="Number of CV folds (max 3)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
