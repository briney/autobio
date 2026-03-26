#!/usr/bin/env python3
"""Predict protein structure from sequence using ESMFold via HuggingFace transformers.

Reads config.json from the workspace, loads the ESMFold model, predicts the
structure, and saves the PDB output and confidence metrics to outputs/raw/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import EsmForProteinFolding, EsmTokenizer


def parse_fasta(path: Path) -> dict[str, str]:
    """Parse a FASTA file into {header: sequence} pairs."""
    sequences: dict[str, str] = {}
    current_header: str | None = None
    chunks: list[str] = []

    with open(path) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    sequences[current_header] = "".join(chunks)
                current_header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)

    if current_header is not None:
        sequences[current_header] = "".join(chunks)

    return sequences


def convert_outputs_to_pdb(model, outputs) -> str:
    """Convert ESMFold model outputs to PDB string using the model's built-in method."""
    pdb_list = model.output_to_pdb(outputs)
    return pdb_list[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="ESMFold structure prediction")
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()

    workspace = args.workspace
    config = json.loads((workspace / "config.json").read_text())

    model_name = config["model_name"]
    input_fasta = Path(config["input_fasta"])
    output_dir = Path(config["output_dir"])

    # Parse input — ESMFold expects exactly one sequence
    sequences = parse_fasta(input_fasta)
    if len(sequences) != 1:
        raise ValueError(f"ESMFold expects exactly 1 sequence, got {len(sequences)}")
    seq_id, sequence = next(iter(sequences.items()))

    print(f"[esmfold] Model: {model_name}")
    print(f"[esmfold] Sequence: {seq_id!r} ({len(sequence)} residues)")

    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[esmfold] Device: {device}")
    print(f"[esmfold] Loading model {model_name}...")
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForProteinFolding.from_pretrained(model_name).to(device)
    model.eval()
    print("[esmfold] Model loaded.")

    # Run inference
    print("[esmfold] Running structure prediction...")
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Extract pTM
    ptm = outputs["ptm"].cpu().item()

    # Convert to PDB using the model's built-in method — this correctly
    # computes pLDDT and writes it into B-factors
    pdb_string = convert_outputs_to_pdb(model, outputs)

    # Extract per-residue pLDDT from the PDB B-factors (CA atoms)
    plddt_per_residue = []
    for line in pdb_string.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA":
            bfactor = float(line[60:66].strip())
            plddt_per_residue.append(bfactor)
    plddt_mean = float(np.mean(plddt_per_residue)) if plddt_per_residue else 0.0

    print(f"[esmfold] pLDDT mean: {plddt_mean:.1f}")
    print(f"[esmfold] pTM: {ptm:.3f}")

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    pdb_path = output_dir / "prediction.pdb"
    pdb_path.write_text(pdb_string)

    # Write metadata for standardize.py
    metadata = {
        "seq_id": seq_id,
        "pdb_path": str(pdb_path),
        "plddt_per_residue": plddt_per_residue,
        "plddt_mean": plddt_mean,
        "ptm": ptm,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[esmfold] Done. Structure saved to {pdb_path.name}")


if __name__ == "__main__":
    main()
