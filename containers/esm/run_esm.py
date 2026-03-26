#!/usr/bin/env python3
"""Extract protein sequence embeddings using ESM models via HuggingFace transformers.

Reads config.json from the workspace, loads the specified ESM model, extracts
embeddings from the requested layer with the specified pooling strategy, and
saves each embedding as a NumPy .npy file in outputs/raw/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import EsmModel, EsmTokenizer


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


def extract_embeddings(
    model: EsmModel,
    tokenizer: EsmTokenizer,
    sequences: dict[str, str],
    layer: int | None,
    pooling: str,
    output_dir: Path,
    device: torch.device,
) -> list[dict]:
    """Extract embeddings for each sequence and save as .npy files.

    Returns a list of result dicts for each sequence.
    """
    results = []

    for seq_id, sequence in sequences.items():
        print(f"[esm] Embedding sequence {seq_id!r} ({len(sequence)} residues)...")

        # Tokenize
        inputs = tokenizer(sequence, return_tensors="pt", padding=False, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward pass — get all hidden states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # Select layer
        hidden_states = outputs.hidden_states  # tuple of (1, L+2, D) tensors
        if layer is not None:
            # layer 0 = embedding layer, layer N = Nth transformer layer
            selected = hidden_states[layer]
        else:
            # Default: final layer
            selected = hidden_states[-1]

        # Remove BOS and EOS tokens: positions [1:-1]
        # selected shape: (1, L+2, D) -> (L, D)
        token_embeddings = selected[0, 1:-1, :]  # (L, D)

        # Apply pooling
        if pooling == "per_residue":
            embedding = token_embeddings.cpu().numpy()  # (L, D)
        elif pooling == "mean":
            embedding = token_embeddings.mean(dim=0).cpu().numpy()  # (D,)
        elif pooling == "cls":
            # CLS token is at position 0 (BOS token in ESM)
            embedding = selected[0, 0, :].cpu().numpy()  # (D,)
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")

        # Save as .npy
        out_path = output_dir / f"{seq_id}.npy"
        np.save(out_path, embedding)

        actual_layer = layer if layer is not None else len(hidden_states) - 1

        results.append({
            "sequence_id": seq_id,
            "embedding_path": str(out_path),
            "dimension": int(token_embeddings.shape[-1]),
            "layer": actual_layer,
            "pooling": pooling,
        })

        print(f"[esm]   -> shape {embedding.shape}, saved to {out_path.name}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="ESM embedding extraction")
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()

    workspace = args.workspace
    config = json.loads((workspace / "config.json").read_text())

    model_name = config["model_name"]
    input_fasta = Path(config["input_fasta"])
    output_dir = Path(config["output_dir"])
    layer = config.get("layer")
    pooling = config.get("pooling", "mean")

    # Parse input sequences
    sequences = parse_fasta(input_fasta)
    print(f"[esm] Model: {model_name}")
    print(f"[esm] Sequences: {len(sequences)}")
    print(f"[esm] Layer: {layer if layer is not None else 'final'}")
    print(f"[esm] Pooling: {pooling}")

    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[esm] Device: {device}")
    print(f"[esm] Loading model {model_name}...")
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmModel.from_pretrained(model_name).to(device)
    model.eval()
    print("[esm] Model loaded.")

    # Extract embeddings
    output_dir.mkdir(parents=True, exist_ok=True)
    results = extract_embeddings(model, tokenizer, sequences, layer, pooling, output_dir, device)

    # Write a metadata file for standardize.py to use
    metadata = {
        "model_name": model_name,
        "embedding_dimension": results[0]["dimension"] if results else 0,
        "results": results,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[esm] Done. {len(results)} embeddings extracted.")


if __name__ == "__main__":
    main()
