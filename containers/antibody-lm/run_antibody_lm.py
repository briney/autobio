#!/usr/bin/env python3
"""Extract antibody sequence embeddings or compute pseudo log-likelihood.

Reads config.json from the workspace, loads the specified antibody language
model (CurrAb, ft-ESM, BALM-paired, or BALM-unpaired), and either extracts
embeddings or computes PLL scores depending on the ``mode`` field.

Handles two model families:
  - **ESM** (CurrAb, ft-ESM): uses EsmTokenizer, chain separator is <cls>
  - **RoBERTa** (BALM): uses RobertaTokenizer, chain separator is </s>

Token construction is done manually for paired sequences to ensure correct
placement of separator tokens.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------


def tokenize_sequence(
    tokenizer,
    heavy_chain: str | None,
    light_chain: str | None,
    chain_separator: str,
    model_family: str,
) -> dict[str, torch.Tensor]:
    """Build token IDs for an antibody sequence with the correct chain separator.

    Constructs token IDs manually rather than relying on the tokenizer
    to parse inline special tokens.

    Returns:
        Dict with ``input_ids`` and ``attention_mask`` tensors of shape (1, L).
    """
    if chain_separator == "none":
        # Single chain — use standard tokenization
        chain = heavy_chain or light_chain
        encoded = tokenizer(chain, return_tensors="pt", padding=False, truncation=True)
        return {k: v for k, v in encoded.items()}

    # Tokenize each chain separately (without special tokens)
    heavy_ids = _encode_chain(tokenizer, heavy_chain) if heavy_chain else []
    light_ids = _encode_chain(tokenizer, light_chain) if light_chain else []

    # Build full token sequence with separator
    bos_id = tokenizer.cls_token_id
    eos_id = tokenizer.eos_token_id if model_family == "esm" else tokenizer.sep_token_id

    if chain_separator == "single_cls":
        sep_ids = [tokenizer.cls_token_id]
    elif chain_separator == "double_cls":
        sep_ids = [tokenizer.cls_token_id, tokenizer.cls_token_id]
    elif chain_separator == "sep":
        sep_ids = [tokenizer.sep_token_id]
    else:
        raise ValueError(f"Unknown chain_separator: {chain_separator}")

    # Construct: [BOS] + heavy_tokens + [separator(s)] + light_tokens + [EOS]
    input_ids = [bos_id] + heavy_ids + sep_ids + light_ids + [eos_id]
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids_tensor)

    return {"input_ids": input_ids_tensor, "attention_mask": attention_mask}


def _encode_chain(tokenizer, chain: str) -> list[int]:
    """Tokenize a single amino acid chain, returning only the content token IDs.

    Strips any BOS/EOS/CLS/SEP tokens that the tokenizer may auto-add.
    """
    encoded = tokenizer.encode(chain, add_special_tokens=False)
    return encoded


def get_special_token_ids(tokenizer, model_family: str) -> set[int]:
    """Return the set of token IDs considered 'special' for this model."""
    ids = set()
    for attr in ("cls_token_id", "eos_token_id", "sep_token_id", "pad_token_id", "mask_token_id"):
        tid = getattr(tokenizer, attr, None)
        if tid is not None:
            ids.add(tid)
    return ids


def get_base_model(model, model_family: str):
    """Extract the base encoder model (without the LM head)."""
    if model_family == "esm":
        return model.esm
    return model.roberta


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
    model_family = config["model_family"]
    chain_separator = config["chain_separator"]
    layer = config.get("layer")
    pooling = config.get("pooling", "mean")

    base_model = get_base_model(model, model_family)
    special_ids = get_special_token_ids(tokenizer, model_family)
    results = []

    for seq in sequences:
        seq_id = seq["id"]
        heavy = seq.get("heavy_chain")
        light = seq.get("light_chain")
        print(f"[antibody-lm] Embedding sequence {seq_id!r}...")

        inputs = tokenize_sequence(tokenizer, heavy, light, chain_separator, model_family)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = base_model(**inputs, output_hidden_states=True)

        # Select layer
        hidden_states = outputs.hidden_states
        selected = hidden_states[layer] if layer is not None else hidden_states[-1]

        # Remove special tokens
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
            # CLS token is at position 0 (BOS)
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
        print(f"[antibody-lm]   -> shape {embedding.shape}, saved to {out_path.name}")

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
    model_family = config["model_family"]
    chain_separator = config["chain_separator"]
    per_position = config.get("per_position", False)
    mask_id = tokenizer.mask_token_id

    special_ids = get_special_token_ids(tokenizer, model_family)
    results = []

    for seq in sequences:
        seq_id = seq["id"]
        heavy = seq.get("heavy_chain")
        light = seq.get("light_chain")
        print(f"[antibody-lm] Computing PLL for sequence {seq_id!r}...")

        inputs = tokenize_sequence(tokenizer, heavy, light, chain_separator, model_family)
        original_ids = inputs["input_ids"].to(device)  # (1, L)
        attention_mask = inputs["attention_mask"].to(device)

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

            with torch.no_grad():
                outputs = model(input_ids=masked_ids, attention_mask=attention_mask)

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
        print(f"[antibody-lm]   -> PLL={total_pll:.4f} ({len(scoreable_positions)} positions)")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Antibody LM embedding/PLL")
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()

    workspace = args.workspace
    config = json.loads((workspace / "config.json").read_text())

    model_name = config["model_name"]
    mode = config["mode"]

    print(f"[antibody-lm] Model: {model_name}")
    print(f"[antibody-lm] Mode: {mode}")

    # Load sequences
    input_file = Path(config["input_file"])
    sequences = json.loads(input_file.read_text())
    print(f"[antibody-lm] Sequences: {len(sequences)}")

    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[antibody-lm] Device: {device}")
    print(f"[antibody-lm] Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    model.eval()
    print("[antibody-lm] Model loaded.")

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
    print(f"[antibody-lm] Done. {len(results)} sequences processed.")


if __name__ == "__main__":
    main()
