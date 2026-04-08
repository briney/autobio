#!/usr/bin/env python3
"""Custom inference wrapper for ANTIPASTI binding affinity prediction.

The upstream ANTIPASTI Preprocessing class is tightly coupled to the training
workflow (expects all training PDBs, runs assertions against saved metadata).
This script bypasses it and directly implements the prediction pipeline:

1. Extract Fv region from the input PDB using chain IDs
2. Compute DCCM via Normal Mode Analysis (R + bio3d)
3. Align the DCCM map to the training grid using pre-computed metadata
4. Load the CNN model and predict log10(Kd)
5. Write output.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


def setup_antipasti_env(antipasti_dir: str) -> None:
    """Add ANTIPASTI to Python path."""
    sys.path.insert(0, antipasti_dir)


def extract_chains(
    pdb_path: Path,
    heavy_chain: str,
    light_chain: str,
    antigen_chains: list[str],
    output_path: Path,
    max_h_residues: int = 113,
    max_l_residues: int = 107,
) -> tuple[int, int]:
    """Extract antibody variable region and antigen from PDB.

    Extracts the first max_h_residues unique residue positions from the heavy
    chain and first max_l_residues from the light chain (approximating the Fv
    region for both Chothia-numbered and sequentially-numbered PDBs). All
    antigen residues are included. Chains are relabeled to H/L/A.

    Returns the number of heavy and light chain residues extracted.
    """
    # First pass: collect sorted unique residue numbers per chain
    h_all_res: set[int] = set()
    l_all_res: set[int] = set()
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain = line[21]
            try:
                resseq = int(line[22:26].strip())
            except ValueError:
                continue
            if chain == heavy_chain:
                h_all_res.add(resseq)
            elif chain == light_chain:
                l_all_res.add(resseq)

    # Keep only the first N residue positions (Fv approximation)
    h_keep = set(sorted(h_all_res)[:max_h_residues])
    l_keep = set(sorted(l_all_res)[:max_l_residues])

    # Second pass: extract atoms for selected residues
    h_residues: set[int] = set()
    l_residues: set[int] = set()
    lines_out: list[str] = []

    with open(pdb_path) as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                if line.startswith("END"):
                    lines_out.append(line)
                continue

            chain = line[21]
            try:
                resseq = int(line[22:26].strip())
            except ValueError:
                continue

            if chain == heavy_chain and resseq in h_keep:
                new_line = line[:21] + "H" + line[22:]
                lines_out.append(new_line)
                h_residues.add(resseq)
            elif chain == light_chain and resseq in l_keep:
                new_line = line[:21] + "L" + line[22:]
                lines_out.append(new_line)
                l_residues.add(resseq)
            elif chain in antigen_chains:
                new_line = line[:21] + "A" + line[22:]
                lines_out.append(new_line)

    with open(output_path, "w") as f:
        f.writelines(lines_out)

    return len(h_residues), len(l_residues)


def compute_dccm(
    pdb_path: Path,
    output_npy: Path,
    scripts_path: Path,
    modes: str | int,
) -> None:
    """Compute DCCM via R bio3d Normal Mode Analysis."""
    r_script = scripts_path / "pdb_to_dccm.r"

    # The R script expects: Rscript pdb_to_dccm.r <pdb> <output_npy> <modes>
    cmd = [
        "Rscript",
        str(r_script),
        str(pdb_path),
        str(output_npy),
        str(modes),
    ]
    print(f"[antipasti] Running NMA: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"[antipasti] R stdout: {result.stdout}", file=sys.stderr)
        print(f"[antipasti] R stderr: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"DCCM computation failed (exit {result.returncode})")
    print("[antipasti] DCCM computation complete.")


def load_training_metadata(data_dir: Path) -> dict:
    """Load pre-computed training metadata for map alignment.

    The training set chain lengths determine the dimensions of the DCCM
    alignment grid. These are saved as .npy files in data/chain_lengths/.
    """
    chain_lengths_dir = data_dir / "chain_lengths"

    heavy_file = chain_lengths_dir / "heavy_lengths.npy"
    light_file = chain_lengths_dir / "light_lengths.npy"

    if not heavy_file.exists() or not light_file.exists():
        raise FileNotFoundError(
            f"Chain length files not found in {chain_lengths_dir}. "
            f"Expected heavy_lengths.npy and light_lengths.npy. "
            f"Found: {list(chain_lengths_dir.iterdir())}"
        )

    heavy_lengths = np.load(heavy_file)
    light_lengths = np.load(light_file)
    max_h = int(heavy_lengths.max())
    max_l = int(light_lengths.max())
    print(f"[antipasti] Training max chains: H={max_h}, L={max_l}")

    return {"max_h": max_h, "max_l": max_l}


def align_dccm_map(
    dccm: np.ndarray,
    h_len: int,
    l_len: int,
    max_h: int,
    max_l: int,
    target_size: int = 281,
) -> np.ndarray:
    """Align a DCCM map to the fixed model input size.

    Places H/L blocks at their padded positions (H padded to max_h, L to
    max_l) and fills antigen residues into the remaining space. The final
    map is cropped or zero-padded to exactly target_size x target_size.
    """
    total_fv = h_len + l_len
    ag_size = dccm.shape[0] - total_fv
    # Cap antigen to fit within target_size
    ag_available = max(0, target_size - max_h - max_l)
    ag_use = min(ag_size, ag_available)

    aligned = np.zeros((target_size, target_size), dtype=dccm.dtype)

    # Copy H-H block
    hh = min(h_len, max_h, target_size)
    aligned[:hh, :hh] = dccm[:hh, :hh]

    # Copy H-L and L-H blocks
    ll = min(l_len, max_l)
    if max_h + ll <= target_size:
        aligned[:hh, max_h:max_h + ll] = dccm[:hh, h_len:h_len + ll]
        aligned[max_h:max_h + ll, :hh] = dccm[h_len:h_len + ll, :hh]
        # Copy L-L block
        aligned[max_h:max_h + ll, max_h:max_h + ll] = dccm[h_len:h_len + ll, h_len:h_len + ll]

    # Copy antigen blocks
    if ag_use > 0:
        ag_start = max_h + max_l
        if ag_start + ag_use <= target_size:
            # H-Ag
            aligned[:hh, ag_start:ag_start + ag_use] = dccm[:hh, total_fv:total_fv + ag_use]
            # Ag-H
            aligned[ag_start:ag_start + ag_use, :hh] = dccm[total_fv:total_fv + ag_use, :hh]
            # L-Ag
            aligned[max_h:max_h + ll, ag_start:ag_start + ag_use] = dccm[h_len:h_len + ll, total_fv:total_fv + ag_use]
            # Ag-L
            aligned[ag_start:ag_start + ag_use, max_h:max_h + ll] = dccm[total_fv:total_fv + ag_use, h_len:h_len + ll]
            # Ag-Ag
            aligned[ag_start:ag_start + ag_use, ag_start:ag_start + ag_use] = dccm[total_fv:total_fv + ag_use, total_fv:total_fv + ag_use]

    return aligned


def run_inference(args: argparse.Namespace) -> None:
    """Run ANTIPASTI inference and write output JSON."""
    antipasti_dir = Path(args.antipasti_dir)
    setup_antipasti_env(args.antipasti_dir)

    from antipasti.utils.torch_utils import load_checkpoint

    pdb_path = Path(args.pdb_path)
    pdb_id = pdb_path.stem.lower()
    antigen_chains = json.loads(args.antigen_chains)
    modes = args.modes if args.modes == "all" else int(args.modes)

    print(f"[antipasti] PDB ID: {pdb_id}")
    print(f"[antipasti] Heavy chain: {args.heavy_chain}")
    print(f"[antipasti] Light chain: {args.light_chain}")
    print(f"[antipasti] Antigen chains: {antigen_chains}")

    # -- Set up working directory -----------------------------------------------
    work_dir = Path(tempfile.mkdtemp(prefix="antipasti_"))

    # -- Step 1: Extract chains ---------------------------------------------------
    fv_pdb = work_dir / f"{pdb_id}_fv.pdb"
    h_len, l_len = extract_chains(
        pdb_path=pdb_path,
        heavy_chain=args.heavy_chain,
        light_chain=args.light_chain,
        antigen_chains=antigen_chains,
        output_path=fv_pdb,
    )
    print(f"[antipasti] Extracted Fv: H={h_len} residues, L={l_len} residues")

    if h_len == 0 or l_len == 0:
        raise RuntimeError(
            f"Failed to extract chains: H={h_len}, L={l_len} residues. "
            f"Check that chain IDs '{args.heavy_chain}' (heavy) and "
            f"'{args.light_chain}' (light) exist in the PDB file."
        )

    # -- Step 2: Compute DCCM via R bio3d NMA -----------------------------------
    dccm_npy = work_dir / f"{pdb_id}_dccm.npy"
    compute_dccm(
        pdb_path=fv_pdb,
        output_npy=dccm_npy,
        scripts_path=antipasti_dir / "scripts",
        modes=modes,
    )
    dccm = np.load(dccm_npy)
    print(f"[antipasti] DCCM shape: {dccm.shape}")

    # -- Step 3: Load training metadata and align map ---------------------------
    data_dir = antipasti_dir / "data"
    metadata = load_training_metadata(data_dir)

    # The trained model has a fixed input_shape (281 for the default checkpoint).
    # Load the checkpoint first to determine the expected input shape.
    print(f"[antipasti] Loading model: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    # Infer input_shape from fc1 weight: fc1 maps (n_filters * pooled_size²) -> 1
    # With default pool_size=1 and filter_size=4: pooled = input_shape - 3
    # fc1.weight shape = (1, n_filters * (input_shape - filter_size + 1)²)
    fc1_size = checkpoint["model_state_dict"]["fc1.weight"].shape[1]
    n_filters = checkpoint.get("n_filters", 4)
    filter_size = checkpoint.get("filter_size", 4)
    pooled_per_filter = fc1_size // n_filters
    pooled_dim = int(pooled_per_filter ** 0.5)
    input_shape = pooled_dim + filter_size - 1
    print(f"[antipasti] Model expects input_shape={input_shape}")

    aligned = align_dccm_map(
        dccm=dccm,
        h_len=h_len,
        l_len=l_len,
        max_h=metadata["max_h"],
        max_l=metadata["max_l"],
        target_size=input_shape,
    )
    print(f"[antipasti] Aligned map shape: {aligned.shape}")

    # Convert to model input: (1, 1, H, W) tensor
    test_x = torch.tensor(aligned, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # -- Step 4: Load model and predict -----------------------------------------
    from antipasti.model.model import ANTIPASTI as AntipastiModel

    model = AntipastiModel(
        n_filters=n_filters,
        filter_size=filter_size,
        pooling_size=checkpoint.get("pooling_size", 1),
        input_shape=input_shape,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        prediction, _filters = model(test_x)

    log10_kd = prediction.item()
    kd_molar = 10 ** log10_kd
    print(f"[antipasti] Predicted log10(Kd): {log10_kd:.4f}")
    print(f"[antipasti] Predicted Kd: {kd_molar:.2e} M")

    # -- Step 5: Write output JSON -----------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_name = Path(args.checkpoint_path).stem
    output = {
        "pdb_id": pdb_id,
        "log10_kd": log10_kd,
        "kd_molar": kd_molar,
        "heavy_chain": args.heavy_chain,
        "light_chain": args.light_chain,
        "antigen_chains": antigen_chains,
        "modes": args.modes,
        "checkpoint": checkpoint_name,
    }

    output_path = output_dir / "output.json"
    output_path.write_text(json.dumps(output, indent=2))
    print(f"[antipasti] Results written to: {output_path}")

    # -- Cleanup ----------------------------------------------------------------
    shutil.rmtree(work_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ANTIPASTI binding affinity prediction (single-PDB inference)"
    )
    parser.add_argument("--pdb_path", required=True, help="Path to input PDB file")
    parser.add_argument("--heavy_chain", required=True, help="Heavy chain ID")
    parser.add_argument("--light_chain", required=True, help="Light chain ID")
    parser.add_argument(
        "--antigen_chains",
        required=True,
        help="JSON array of antigen chain IDs",
    )
    parser.add_argument("--checkpoint_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--antipasti_dir", required=True, help="Path to ANTIPASTI repo root")
    parser.add_argument("--output_dir", required=True, help="Directory for output JSON")
    parser.add_argument("--modes", default="all", help="Number of normal modes ('all' or integer)")
    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
