#!/usr/bin/env python3
"""Run LigandMPNN sidechain packing on a mutated PDB structure.

Directly calls the Packer model from LigandMPNN's sc_utils module,
bypassing the sequence design step entirely.  This produces full-atom
PDB structures with repacked sidechains and per-residue confidence scores.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add LigandMPNN to Python path
sys.path.insert(0, "/app/LigandMPNN")

from data_utils import (  # noqa: E402
    featurize,
    parse_PDB,
    write_full_PDB,
)
from sc_utils import Packer, pack_side_chains  # noqa: E402


def main() -> None:
    """Load model, pack sidechains, write output PDBs and scores."""
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/workspace/config.json")
    config = json.loads(config_path.read_text())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pack_sidechains] Using device: {device}")

    # --- Parse mutated PDB ---------------------------------------------------
    mutated_pdb = Path("/workspace/inputs/mutated.pdb")
    protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
        str(mutated_pdb), device=device, parse_all_atoms=True,
    )

    # --- Set chain_mask (all zeros = no residues designed, packing only) ------
    num_residues = protein_dict["mask"].shape[0]
    protein_dict["chain_mask"] = torch.zeros(num_residues, dtype=torch.int32, device=device)

    # Ensure S is int64 (LongTensor) for one_hot in sidechain packing
    if protein_dict["S"].dtype != torch.int64:
        protein_dict["S"] = protein_dict["S"].long()

    # --- Featurize -----------------------------------------------------------
    use_ligand_context = config.get("pack_with_ligand_context", True)
    batch = featurize(
        protein_dict,
        cutoff_for_score=8.0,
        use_atom_context=use_ligand_context,
        number_of_ligand_atoms=16,
        model_type="ligand_mpnn",
    )

    # --- Load sidechain packing model ----------------------------------------
    checkpoint_sc = config["checkpoint_sc"]
    model_sc = Packer(
        node_features=128,
        edge_features=128,
        hidden_dim=128,
        num_encoder_layers=3,
        num_decoder_layers=3,
        atom_context_num=16,
        lower_bound=0.0,
        upper_bound=20.0,
        top_k=32,
        dropout=0.0,
        augment_eps=0.0,
        atom37_order=False,
        device=device,
        num_mix=3,
    )
    ckpt = torch.load(checkpoint_sc, map_location=device, weights_only=False)
    model_sc.load_state_dict(ckpt["model_state_dict"])
    model_sc.to(device)
    model_sc.eval()

    # --- Packing parameters --------------------------------------------------
    num_packs = config.get("num_packs", 4)
    num_denoising_steps = config.get("num_denoising_steps", 3)
    num_samples = config.get("num_samples", 16)
    repack_everything = config.get("repack_everything", True)

    if "seed" in config:
        torch.manual_seed(config["seed"])

    # --- Run sidechain packing -----------------------------------------------
    out_dir = Path("/workspace/outputs/raw")
    all_scores: list[dict] = []

    # Extract arrays needed for write_full_PDB (protein_dict is 1D, not batched)
    S = protein_dict["S"].cpu().numpy()
    R_idx = protein_dict["R_idx"].cpu().numpy()
    chain_letters = protein_dict["chain_letters"]
    mask = protein_dict["mask"].cpu().numpy()

    print(f"[pack_sidechains] Generating {num_packs} packed structure(s)...")
    with torch.no_grad():
        for i in range(num_packs):
            # pack_side_chains modifies feature_dict in place, so copy the batch
            batch_copy = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            sc_result = pack_side_chains(
                batch_copy,
                model_sc,
                num_denoising_steps=num_denoising_steps,
                num_samples=num_samples,
                repack_everything=repack_everything,
            )

            # Write packed PDB using LigandMPNN's native writer
            pdb_path = out_dir / f"packed_{i:04d}.pdb"
            X_packed = sc_result["X"][0].cpu().numpy()  # (L, 14, 3)
            X_m_packed = sc_result["X_m"][0].cpu().numpy()  # (L, 14)
            b_factors = sc_result["b_factors"][0].cpu().numpy()  # (L, 14)

            write_full_PDB(
                save_path=str(pdb_path),
                X=X_packed,
                X_m=X_m_packed,
                b_factors=b_factors,
                R_idx=R_idx,
                chain_letters=chain_letters,
                S=S,
                other_atoms=other_atoms,
                icodes=icodes,
            )
            print(f"[pack_sidechains] Wrote {pdb_path.name}")

            # Collect scores
            log_prob = sc_result.get("log_prob")

            per_residue = None
            total_score = 0.0
            if log_prob is not None:
                # log_prob shape: (B, L, 4) — per chi angle
                lp = log_prob[0].cpu().numpy()  # (L, 4)
                # Mean log-prob across chi angles per residue
                per_res_lp = lp.mean(axis=1)  # (L,)
                per_residue = [
                    float(per_res_lp[j])
                    for j in range(len(per_res_lp))
                    if mask[j] > 0.5
                ]
                total_score = float(np.mean(per_residue)) if per_residue else 0.0

            all_scores.append({
                "pack_id": i,
                "total_score": total_score,
                "per_residue_scores": per_residue,
                "structure_file": pdb_path.name,
            })

    # Write scores
    scores_path = out_dir / "packing_scores.json"
    scores_path.write_text(json.dumps(all_scores, indent=2))
    print(f"[pack_sidechains] Wrote {scores_path.name}")


if __name__ == "__main__":
    main()
