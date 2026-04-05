#!/usr/bin/env python3
"""Extract antibody sequence embeddings or compute pseudo log-likelihood using AbLang2.

Reads config.json from the workspace, loads the AbLang2 model, and either
extracts embeddings or computes PLL scores depending on the ``mode`` field.

AbLang2 uses a custom ESM-2-derived architecture with RoPE and SwiGLU.
Tokenization is character-level (26 tokens). Paired sequences are formatted
as ``<HEAVY>|<LIGHT>``, unpaired as ``<CHAIN>``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ablang2.load_model import load_model


# ---------------------------------------------------------------------------
# Special token IDs (from ablang2 vocab)
# ---------------------------------------------------------------------------

START_TOKEN = 0   # <
PAD_TOKEN = 21    # -
END_TOKEN = 22    # >
MASK_TOKEN = 23   # *
UNK_TOKEN = 24    # X
SEP_TOKEN = 25    # |

SPECIAL_TOKEN_IDS = frozenset({START_TOKEN, PAD_TOKEN, END_TOKEN, MASK_TOKEN, UNK_TOKEN, SEP_TOKEN})


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def format_sequence(heavy: str | None, light: str | None) -> str:
    """Format an antibody sequence for AbLang2 tokenization.

    AbLang2 expects chains wrapped in ``<>`` with ``|`` separator:
      - Paired: ``<HEAVY>|<LIGHT>``
      - Heavy only: ``<HEAVY>``
      - Light only: ``<LIGHT>``
    """
    if heavy and light:
        return f"<{heavy}>|<{light}>"
    if heavy:
        return f"<{heavy}>"
    return f"<{light}>"


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


def extract_embeddings(
    model,
    tokenizer,
    sequences: list[dict],
    config: dict,
    output_dir: Path,
    device: torch.device,
) -> list[dict]:
    """Extract embeddings for each sequence and save as .npy files."""
    layer = config.get("layer")
    pooling = config.get("pooling", "mean")
    num_layers = 12  # AbLang2 has 12 transformer blocks

    results = []

    for seq in sequences:
        seq_id = seq["id"]
        heavy = seq.get("heavy_chain")
        light = seq.get("light_chain")
        print(f"[ablang2] Embedding sequence {seq_id!r}...")

        formatted = format_sequence(heavy, light)
        tokens = tokenizer([formatted], pad=True, w_extra_tkns=False, device=device)

        with torch.no_grad():
            if layer is not None:
                # Get specific layer hidden states
                layer_dict = model(tokens, return_rep_layers=[layer])
                hidden = layer_dict[layer]  # (1, L, D)
            else:
                # Get last hidden states from AbRep (after final layer norm)
                hidden = model.AbRep(tokens).last_hidden_states  # (1, L, D)

        # Mask special tokens
        token_ids = tokens[0]  # (L,)
        mask = torch.tensor(
            [tid.item() not in SPECIAL_TOKEN_IDS for tid in token_ids],
            dtype=torch.bool,
            device=device,
        )
        token_embeddings = hidden[0][mask]  # (L_content, D)

        # Apply pooling
        if pooling == "per_residue":
            embedding = token_embeddings.cpu().numpy()
        elif pooling == "mean":
            embedding = token_embeddings.mean(dim=0).cpu().numpy()
        elif pooling == "cls":
            # Use first token (start token <) embedding
            embedding = hidden[0, 0, :].cpu().numpy()
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")

        out_path = output_dir / f"{seq_id}.npy"
        np.save(out_path, embedding)

        actual_layer = layer if layer is not None else num_layers
        results.append({
            "sequence_id": seq_id,
            "embedding_path": str(out_path),
            "dimension": int(token_embeddings.shape[-1]),
            "layer": actual_layer,
            "pooling": pooling,
        })
        print(f"[ablang2]   -> shape {embedding.shape}, saved to {out_path.name}")

    return results


# ---------------------------------------------------------------------------
# Pseudo log-likelihood
# ---------------------------------------------------------------------------


def compute_pll(
    model,
    tokenizer,
    sequences: list[dict],
    config: dict,
    device: torch.device,
) -> list[dict]:
    """Compute pseudo log-likelihood by iterative masking."""
    per_position = config.get("per_position", False)
    results = []

    for seq in sequences:
        seq_id = seq["id"]
        heavy = seq.get("heavy_chain")
        light = seq.get("light_chain")
        print(f"[ablang2] Computing PLL for sequence {seq_id!r}...")

        formatted = format_sequence(heavy, light)
        tokens = tokenizer([formatted], pad=True, w_extra_tkns=False, device=device)
        original_ids = tokens.clone()  # (1, L)

        # Identify non-special token positions to score
        scoreable_positions = [
            i for i in range(original_ids.shape[1])
            if original_ids[0, i].item() not in SPECIAL_TOKEN_IDS
        ]

        log_probs = []
        for pos in scoreable_positions:
            # Mask this position
            masked_ids = original_ids.clone()
            masked_ids[0, pos] = MASK_TOKEN

            with torch.no_grad():
                logits = model(masked_ids)  # (1, L, vocab_size)

            # Extract log-probability of the true token at the masked position
            pos_logits = logits[0, pos]  # (vocab_size,)
            log_softmax = torch.log_softmax(pos_logits, dim=-1)
            true_token_id = original_ids[0, pos].item()
            log_prob = log_softmax[true_token_id].item()
            log_probs.append(log_prob)

        total_pll = sum(log_probs)
        result = {
            "sequence_id": seq_id,
            "pll": total_pll,
            "sequence_length": len(scoreable_positions),
        }
        if per_position:
            result["per_position_pll"] = log_probs

        results.append(result)
        print(f"[ablang2]   -> PLL={total_pll:.4f} ({len(scoreable_positions)} positions)")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="AbLang2 embedding/PLL")
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()

    workspace = args.workspace
    config = json.loads((workspace / "config.json").read_text())

    model_name = config["model_name"]
    mode = config["mode"]

    print(f"[ablang2] Model: {model_name}")
    print(f"[ablang2] Mode: {mode}")

    # Load sequences
    input_file = Path(config["input_file"])
    sequences = json.loads(input_file.read_text())
    print(f"[ablang2] Sequences: {len(sequences)}")

    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ablang2] Device: {device}")
    print(f"[ablang2] Loading model {model_name}...")
    model, tokenizer, hparams = load_model(model_name)
    model.eval()
    model.to(device)
    print("[ablang2] Model loaded.")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "embedding":
        results = extract_embeddings(model, tokenizer, sequences, config, output_dir, device)
        metadata = {
            "model_name": model_name,
            "embedding_dimension": results[0]["dimension"] if results else 0,
            "results": results,
        }
    elif mode == "pll":
        results = compute_pll(model, tokenizer, sequences, config, device)
        metadata = {
            "model_name": model_name,
            "results": results,
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"[ablang2] Done. {len(results)} sequences processed.")


if __name__ == "__main__":
    main()
