#!/usr/bin/env python3
"""Extract antibody sequence embeddings or compute pseudo log-likelihood using AntiBERTa2.

Reads config.json from the workspace, loads the AntiBERTa2 model
(RoFormerForMaskedLM), and either extracts embeddings or computes PLL scores.

AntiBERTa2 uses a RoFormer architecture with character-level tokenization.
Amino acids are space-separated, with chain-type prefix tokens:
  - Heavy chain: ``\u1e22`` (H with dot below)
  - Light chain: ``\u1e36`` (L with dot below)
Paired sequences use ``[SEP]`` between chains.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import RoFormerForMaskedLM, RoFormerTokenizer


# ---------------------------------------------------------------------------
# Special tokens
# ---------------------------------------------------------------------------

# Chain-type prefix token IDs in the AntiBERTa2 vocabulary
HEAVY_PREFIX_ID = 26   # \u1e22 (Ḣ)
LIGHT_PREFIX_ID = 27   # \u1e36 (Ḷ)


def get_special_token_ids(tokenizer: RoFormerTokenizer) -> set[int]:
    """Return the set of token IDs considered 'special' for AntiBERTa2.

    Includes standard special tokens plus the chain-type prefix tokens.
    """
    ids = {HEAVY_PREFIX_ID, LIGHT_PREFIX_ID}
    for attr in ("cls_token_id", "sep_token_id", "pad_token_id", "mask_token_id", "unk_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            ids.add(tid)
    return ids


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def tokenize_sequence(
    tokenizer: RoFormerTokenizer,
    heavy: str | None,
    light: str | None,
) -> dict[str, torch.Tensor]:
    """Build token IDs for an antibody sequence with AntiBERTa2 formatting.

    AntiBERTa2 expects space-separated amino acids with chain prefix tokens:
      - Paired:  ``[CLS] Ḣ H E A V Y [SEP] Ḷ L I G H T [SEP]``
      - Heavy only: ``[CLS] Ḣ H E A V Y [SEP]``
      - Light only: ``[CLS] Ḷ L I G H T [SEP]``

    Returns:
        Dict with ``input_ids``, ``attention_mask``, and ``token_type_ids``.
    """
    if heavy and light:
        # Paired: two segments separated by [SEP]
        heavy_text = "\u1e22 " + " ".join(heavy)
        light_text = "\u1e36 " + " ".join(light)
        encoded = tokenizer(
            heavy_text,
            light_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
    elif heavy:
        heavy_text = "\u1e22 " + " ".join(heavy)
        encoded = tokenizer(heavy_text, return_tensors="pt", padding=False, truncation=True)
    else:
        light_text = "\u1e36 " + " ".join(light)
        encoded = tokenizer(light_text, return_tensors="pt", padding=False, truncation=True)

    return {k: v for k, v in encoded.items()}


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------


def extract_embeddings(
    model: RoFormerForMaskedLM,
    tokenizer: RoFormerTokenizer,
    sequences: list[dict],
    config: dict,
    output_dir: Path,
    device: torch.device,
) -> list[dict]:
    """Extract embeddings for each sequence and save as .npy files."""
    layer = config.get("layer")
    pooling = config.get("pooling", "mean")

    base_model = model.roformer
    special_ids = get_special_token_ids(tokenizer)
    results = []

    for seq in sequences:
        seq_id = seq["id"]
        heavy = seq.get("heavy_chain")
        light = seq.get("light_chain")
        print(f"[antiberta2] Embedding sequence {seq_id!r}...")

        inputs = tokenize_sequence(tokenizer, heavy, light)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = base_model(**inputs, output_hidden_states=True)

        # Select layer
        hidden_states = outputs.hidden_states
        selected = hidden_states[layer] if layer is not None else hidden_states[-1]

        # Mask special tokens (CLS, SEP, prefix tokens)
        input_ids = inputs["input_ids"][0]
        mask = torch.tensor(
            [tid.item() not in special_ids for tid in input_ids],
            dtype=torch.bool,
            device=device,
        )
        token_embeddings = selected[0][mask]  # (L_content, D)

        # Apply pooling
        if pooling == "per_residue":
            embedding = token_embeddings.cpu().numpy()
        elif pooling == "mean":
            embedding = token_embeddings.mean(dim=0).cpu().numpy()
        elif pooling == "cls":
            # CLS token is at position 0
            embedding = selected[0, 0, :].cpu().numpy()
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}")

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
        print(f"[antiberta2]   -> shape {embedding.shape}, saved to {out_path.name}")

    return results


# ---------------------------------------------------------------------------
# Pseudo log-likelihood
# ---------------------------------------------------------------------------


def compute_pll(
    model: RoFormerForMaskedLM,
    tokenizer: RoFormerTokenizer,
    sequences: list[dict],
    config: dict,
    device: torch.device,
) -> list[dict]:
    """Compute pseudo log-likelihood by iterative masking."""
    per_position = config.get("per_position", False)
    mask_id = tokenizer.mask_token_id
    special_ids = get_special_token_ids(tokenizer)

    results = []

    for seq in sequences:
        seq_id = seq["id"]
        heavy = seq.get("heavy_chain")
        light = seq.get("light_chain")
        print(f"[antiberta2] Computing PLL for sequence {seq_id!r}...")

        inputs = tokenize_sequence(tokenizer, heavy, light)
        original_ids = inputs["input_ids"].to(device)  # (1, L)
        attention_mask = inputs["attention_mask"].to(device)
        token_type_ids = inputs.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        # Identify non-special token positions to score
        scoreable_positions = [
            i for i in range(original_ids.shape[1])
            if original_ids[0, i].item() not in special_ids
        ]

        log_probs = []
        for pos in scoreable_positions:
            # Mask this position
            masked_ids = original_ids.clone()
            masked_ids[0, pos] = mask_id

            fwd_kwargs = {
                "input_ids": masked_ids,
                "attention_mask": attention_mask,
            }
            if token_type_ids is not None:
                fwd_kwargs["token_type_ids"] = token_type_ids

            with torch.no_grad():
                outputs = model(**fwd_kwargs)

            # Extract log-probability of the true token at the masked position
            logits = outputs.logits[0, pos]  # (vocab_size,)
            log_softmax = torch.log_softmax(logits, dim=-1)
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
        print(f"[antiberta2]   -> PLL={total_pll:.4f} ({len(scoreable_positions)} positions)")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="AntiBERTa2 embedding/PLL")
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()

    workspace = args.workspace
    config = json.loads((workspace / "config.json").read_text())

    model_name = config["model_name"]
    mode = config["mode"]

    print(f"[antiberta2] Model: {model_name}")
    print(f"[antiberta2] Mode: {mode}")

    # Load sequences
    input_file = Path(config["input_file"])
    sequences = json.loads(input_file.read_text())
    print(f"[antiberta2] Sequences: {len(sequences)}")

    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[antiberta2] Device: {device}")
    print(f"[antiberta2] Loading model {model_name}...")
    tokenizer = RoFormerTokenizer.from_pretrained(model_name)
    model = RoFormerForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()
    print("[antiberta2] Model loaded.")

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
    print(f"[antiberta2] Done. {len(results)} sequences processed.")


if __name__ == "__main__":
    main()
