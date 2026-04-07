#!/usr/bin/env python3
"""AntiFold antibody inverse folding and sequence scoring.

Reads config.json from the workspace and dispatches to either:
- design mode: sample antibody sequences from backbone structure
- score mode: compute conditional log-likelihood of sequences

Includes ANARCI-based IMGT renumbering of input PDBs before running
AntiFold inference.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ---------------------------------------------------------------------------
# ANARCI renumbering
# ---------------------------------------------------------------------------


def _extract_sequences_from_pdb(pdb_path: str) -> dict[str, str]:
    """Extract amino acid sequences per chain from a PDB file."""
    three_to_one = {
        "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
        "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
        "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
        "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
    }
    chains: dict[str, list[tuple[int, str, str]]] = {}
    seen: set[tuple[str, int, str]] = set()

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            chain_id = line[21]
            resname = line[17:20].strip()
            resseq = int(line[22:26].strip())
            icode = line[26].strip()

            key = (chain_id, resseq, icode)
            if key in seen:
                continue
            seen.add(key)

            aa = three_to_one.get(resname)
            if aa:
                chains.setdefault(chain_id, []).append((resseq, icode, aa))

    result: dict[str, str] = {}
    for chain_id, residues in chains.items():
        residues.sort(key=lambda x: (x[0], x[1]))
        result[chain_id] = "".join(r[2] for r in residues)
    return result


def renumber_pdb_imgt(
    pdb_path: str,
    heavy_chain: str | None,
    light_chain: str | None,
    workspace: Path,
) -> str:
    """Renumber a PDB to IMGT scheme using ANARCI.

    Writes the renumbered PDB to the workspace and returns its path.
    Falls back to the original PDB if ANARCI fails.
    """
    try:
        from anarci import anarci
    except ImportError:
        print("[antifold] ANARCI not available, using original PDB numbering")
        return pdb_path

    sequences = _extract_sequences_from_pdb(pdb_path)
    ab_chains = []
    if heavy_chain and heavy_chain in sequences:
        ab_chains.append((heavy_chain, sequences[heavy_chain]))
    if light_chain and light_chain in sequences:
        ab_chains.append((light_chain, sequences[light_chain]))

    if not ab_chains:
        print("[antifold] No antibody chains found, using original PDB")
        return pdb_path

    # Run ANARCI to get IMGT numbering
    anarci_input = [(chain_id, seq) for chain_id, seq in ab_chains]
    try:
        numbering_results, _, _ = anarci(anarci_input, scheme="imgt", output=False)
    except Exception as e:
        print(f"[antifold] ANARCI renumbering failed: {e}, using original PDB")
        return pdb_path

    # Build mapping: (chain_id, old_resseq, icode) -> (new_resseq, new_icode)
    renumber_map: dict[tuple[str, int, str], tuple[int, str]] = {}
    for chain_idx, (chain_id, seq) in enumerate(ab_chains):
        if numbering_results[chain_idx] is None:
            print(f"[antifold] ANARCI could not number chain {chain_id}")
            continue

        # numbering_results[chain_idx] is a list of domain numberings
        # Each domain is a list of ((position, insertion_code), amino_acid) tuples
        domain = numbering_results[chain_idx][0]  # Take first domain
        numbering = domain[0]  # List of ((pos, icode), aa) tuples

        # Build old-to-new mapping by aligning ANARCI output to original sequence
        orig_residues = []
        with open(pdb_path) as f:
            seen_keys: set[tuple[str, int, str]] = set()
            for line in f:
                if not line.startswith("ATOM") or line[21] != chain_id:
                    continue
                resseq = int(line[22:26].strip())
                icode = line[26].strip()
                key = (chain_id, resseq, icode)
                if key not in seen_keys:
                    seen_keys.add(key)
                    orig_residues.append((resseq, icode))

        # ANARCI output includes gaps (-); filter to non-gap positions
        non_gap = [(pos, ic, aa) for (pos, ic), aa in numbering if aa != "-"]

        if len(non_gap) != len(orig_residues):
            print(
                f"[antifold] Length mismatch for chain {chain_id}: "
                f"ANARCI={len(non_gap)}, PDB={len(orig_residues)}. "
                f"Using original numbering for this chain."
            )
            continue

        for (old_resseq, old_icode), (new_pos, new_icode, _) in zip(
            orig_residues, non_gap
        ):
            new_icode_str = new_icode.strip() if new_icode.strip() != " " else ""
            renumber_map[(chain_id, old_resseq, old_icode)] = (
                new_pos,
                new_icode_str,
            )

    if not renumber_map:
        print("[antifold] No residues to renumber, using original PDB")
        return pdb_path

    # Write renumbered PDB
    output_path = str(workspace / "inputs" / "renumbered.pdb")
    with open(pdb_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")):
                chain_id = line[21]
                resseq = int(line[22:26].strip())
                icode = line[26].strip()

                key = (chain_id, resseq, icode)
                if key in renumber_map:
                    new_pos, new_icode = renumber_map[key]
                    line = (
                        line[:22]
                        + f"{new_pos:>4}"
                        + (new_icode if new_icode else " ")
                        + line[27:]
                    )
            fout.write(line)

    print(f"[antifold] Renumbered PDB to IMGT scheme: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model():
    """Load AntiFold model, moving to GPU if available."""
    from antifold.main import load_model as _load_model

    model = _load_model()
    if torch.cuda.is_available():
        model = model.cuda()
        print("[antifold] Using GPU")
    else:
        print("[antifold] Using CPU")
    return model


# ---------------------------------------------------------------------------
# Design mode
# ---------------------------------------------------------------------------


def run_design(config: dict, workspace: Path) -> None:
    """Sample antibody sequences from backbone structure."""
    from antifold.main import (
        df_logits_to_logprobs,
        get_pdbs_logits,
        sample_from_df_logits_H,
        sample_from_df_logits_HL,
    )

    structure_path = config["structure_path"]
    num_sequences = config.get("num_sequences", 1)
    temperature = config.get("temperature", 0.2)
    heavy_chain = config.get("heavy_chain")
    light_chain = config.get("light_chain")
    antigen_chain = config.get("antigen_chain")
    regions = config.get("regions")
    seed = config.get("seed", 42)

    # Renumber PDB to IMGT
    structure_path = renumber_pdb_imgt(
        structure_path, heavy_chain, light_chain, workspace
    )

    # Load model
    model = load_model()

    # Set up PDB directory and CSV for AntiFold
    pdb_dir = str(Path(structure_path).parent)
    pdb_name = Path(structure_path).stem

    # Build pdbs_csv DataFrame
    csv_data: dict[str, list] = {"pdb": [pdb_name], "Hchain": [heavy_chain or ""]}
    csv_data["Lchain"] = [light_chain or ""]

    # Add antigen chain(s) if provided
    if antigen_chain:
        antigen_chains = antigen_chain.split()
        for i, ac in enumerate(antigen_chains):
            csv_data[f"chain{i + 3}"] = [ac]

    pdbs_csv = pd.DataFrame(csv_data)

    # Get logits
    print(f"[antifold] Computing logits for {pdb_name}")
    df_logits_list = get_pdbs_logits(
        model=model,
        pdbs_csv_or_dataframe=pdbs_csv,
        pdb_dir=pdb_dir,
        seed=seed,
    )

    if not df_logits_list:
        raise RuntimeError("AntiFold returned no logits for the input structure")

    df_logits = df_logits_list[0]

    # Determine regions to mutate
    if regions is not None:
        regions_to_mutate = regions
    else:
        regions_to_mutate = ["CDR1", "CDR2", "CDR3"]

    # Sample sequences
    print(f"[antifold] Sampling {num_sequences} sequence(s) at T={temperature}")
    is_nanobody = bool(heavy_chain) and not bool(light_chain)

    if is_nanobody:
        fasta_dict = sample_from_df_logits_H(
            df_logits,
            sample_n=num_sequences,
            sampling_temp=temperature,
            regions_to_mutate=regions_to_mutate,
            nanobody_mode=True,
            seed=seed,
        )
    else:
        fasta_dict = sample_from_df_logits_HL(
            df_logits,
            sample_n=num_sequences,
            sampling_temp=temperature,
            regions_to_mutate=regions_to_mutate,
            seed=seed,
        )

    # Parse results
    native_sequences: dict[str, str] = {}
    samples: list[dict[str, str]] = []
    scores: list[float] = []
    recoveries: list[float] = []

    for key, record in fasta_dict.items():
        seq_str = str(record.seq)
        desc = record.description

        if "__" not in key:
            # This is the original/native sequence
            if is_nanobody:
                native_sequences[heavy_chain] = seq_str
            elif "/" in seq_str:
                h_seq, l_seq = seq_str.split("/", 1)
                if heavy_chain:
                    native_sequences[heavy_chain] = h_seq
                if light_chain:
                    native_sequences[light_chain] = l_seq
            continue

        # Parse designed sequence
        designed: dict[str, str] = {}
        if is_nanobody:
            designed[heavy_chain] = seq_str
        elif "/" in seq_str:
            h_seq, l_seq = seq_str.split("/", 1)
            if heavy_chain:
                designed[heavy_chain] = h_seq
            if light_chain:
                designed[light_chain] = l_seq
        else:
            designed[heavy_chain or light_chain] = seq_str

        samples.append(designed)

        # Parse score and recovery from FASTA description
        score = _parse_fasta_field(desc, "global_score")
        recovery = _parse_fasta_field(desc, "seq_recovery")
        scores.append(score if score is not None else 0.0)
        recoveries.append(recovery if recovery is not None else 0.0)

    # Write raw output
    raw_dir = workspace / "outputs" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "samples": samples,
        "native_sequences": native_sequences,
        "scores": scores,
        "recoveries": recoveries,
    }
    (raw_dir / "design_results.json").write_text(json.dumps(result, indent=2))
    print(f"[antifold] Wrote {len(samples)} designed sequence(s)")


def _parse_fasta_field(description: str, field: str) -> float | None:
    """Parse a numeric field from AntiFold's FASTA description line."""
    for part in description.split(","):
        part = part.strip()
        if part.startswith(f"{field}="):
            try:
                return float(part.split("=", 1)[1])
            except ValueError:
                return None
    return None


# ---------------------------------------------------------------------------
# Score mode
# ---------------------------------------------------------------------------


def run_score(config: dict, workspace: Path) -> None:
    """Score sequences against a structure using conditional log-likelihoods."""
    from antifold.main import df_logits_to_logprobs, get_pdbs_logits

    structure_path = config["structure_path"]
    heavy_chain = config.get("heavy_chain")
    light_chain = config.get("light_chain")
    antigen_chain = config.get("antigen_chain")
    regions = config.get("regions")
    sequences = config.get("sequences")  # None or dict[chain_id, aa_sequence]
    seed = config.get("seed", 42)

    # Renumber PDB to IMGT
    structure_path = renumber_pdb_imgt(
        structure_path, heavy_chain, light_chain, workspace
    )

    # Load model
    model = load_model()

    # Set up PDB directory and CSV
    pdb_dir = str(Path(structure_path).parent)
    pdb_name = Path(structure_path).stem

    csv_data: dict[str, list] = {"pdb": [pdb_name], "Hchain": [heavy_chain or ""]}
    csv_data["Lchain"] = [light_chain or ""]

    if antigen_chain:
        antigen_chains = antigen_chain.split()
        for i, ac in enumerate(antigen_chains):
            csv_data[f"chain{i + 3}"] = [ac]

    pdbs_csv = pd.DataFrame(csv_data)

    # Get logits
    print(f"[antifold] Computing logits for scoring {pdb_name}")
    df_logits_list = get_pdbs_logits(
        model=model,
        pdbs_csv_or_dataframe=pdbs_csv,
        pdb_dir=pdb_dir,
        seed=seed,
    )

    if not df_logits_list:
        raise RuntimeError("AntiFold returned no logits for the input structure")

    df_logits = df_logits_list[0]
    df_logprobs = df_logits_to_logprobs(df_logits)

    # Amino acid columns in the logits DataFrame
    aa_cols = list("ACDEFGHIKLMNPQRSTVWY")

    # Determine which sequences to score
    if sequences is None:
        # Score native sequence — use pdb_res column
        sequences_to_score = _extract_native_from_logits(df_logprobs, heavy_chain, light_chain)
    else:
        sequences_to_score = sequences

    # Score each chain
    chain_scores: list[dict] = []
    for chain_id, seq in sequences_to_score.items():
        chain_rows = df_logprobs[df_logprobs["pdb_chain"] == chain_id]

        if chain_rows.empty:
            print(f"[antifold] No logits for chain {chain_id}, skipping")
            continue

        # Optionally filter by regions
        if regions:
            chain_rows = chain_rows[chain_rows["assumed_region"].isin(regions)]

        per_residue_ll: list[float] = []
        for idx, (_, row) in enumerate(chain_rows.iterrows()):
            if idx < len(seq):
                aa = seq[idx]
                if aa in aa_cols:
                    per_residue_ll.append(float(row[aa]))
                else:
                    per_residue_ll.append(float("nan"))

        valid_ll = [ll for ll in per_residue_ll if not math.isnan(ll)]
        mean_ll = sum(valid_ll) / len(valid_ll) if valid_ll else 0.0
        perplexity = math.exp(-mean_ll) if valid_ll else float("inf")

        chain_scores.append({
            "chain_id": chain_id,
            "sequence": seq,
            "per_residue_ll": per_residue_ll,
            "mean_ll": round(mean_ll, 6),
            "perplexity": round(perplexity, 4),
        })

    # Write raw output
    raw_dir = workspace / "outputs" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    (raw_dir / "score_results.json").write_text(json.dumps(chain_scores, indent=2))
    print(f"[antifold] Scored {len(chain_scores)} chain(s)")


def _extract_native_from_logits(
    df_logprobs: pd.DataFrame,
    heavy_chain: str | None,
    light_chain: str | None,
) -> dict[str, str]:
    """Extract native sequences from the logits DataFrame."""
    sequences: dict[str, str] = {}
    for chain_id in [heavy_chain, light_chain]:
        if chain_id is None:
            continue
        chain_rows = df_logprobs[df_logprobs["pdb_chain"] == chain_id]
        if not chain_rows.empty:
            sequences[chain_id] = "".join(chain_rows["pdb_res"].values)
    return sequences


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


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
