#!/usr/bin/env python3
"""Standardize ProteinMPNN/LigandMPNN outputs into autobio schema format.

Reads FASTA output from ``outputs/raw/``, parses the input structure for
chain boundaries, and produces ``outputs/standardized/result_data.json``
conforming to the ``InverseFoldingOutput`` schema.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# --- Three-letter to one-letter amino acid mapping ---------------------------
_AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O",
}


def _parse_chain_lengths(structure_path: Path) -> dict[str, int]:
    """Read a PDB/mmCIF file and return residue count per chain.

    Only counts CA atoms to get one residue per position.
    """
    chains: dict[str, list[int]] = {}

    if structure_path.suffix.lower() in (".cif", ".mmcif"):
        # Minimal mmCIF parsing — look for _atom_site rows with CA
        _parse_cif_chains(structure_path, chains)
    else:
        _parse_pdb_chains(structure_path, chains)

    return {chain: len(residues) for chain, residues in chains.items()}


def _parse_pdb_chains(path: Path, chains: dict[str, list[int]]) -> None:
    """Extract chain lengths from PDB ATOM records (CA only)."""
    for line in path.read_text().splitlines():
        if not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        chain_id = line[21]
        res_seq = int(line[22:26].strip())
        if chain_id not in chains:
            chains[chain_id] = []
        if res_seq not in chains[chain_id]:
            chains[chain_id].append(res_seq)


def _parse_cif_chains(path: Path, chains: dict[str, list[int]]) -> None:
    """Extract chain lengths from mmCIF _atom_site records (CA only)."""
    in_atom_site = False
    columns: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith("_atom_site."):
            in_atom_site = True
            columns.append(line.split(".")[1].strip())
            continue
        if in_atom_site and (line.startswith("_") or line.startswith("#") or line.startswith("loop_")):
            in_atom_site = False
            columns = []
            continue
        if not in_atom_site or not columns:
            continue

        fields = line.split()
        if len(fields) < len(columns):
            continue

        col_map = dict(zip(columns, fields))
        atom_name = col_map.get("label_atom_id", "").strip('"')
        if atom_name != "CA":
            continue
        group = col_map.get("group_PDB", "")
        if group not in ("ATOM", "HETATM"):
            continue

        chain_id = col_map.get("label_asym_id", col_map.get("auth_asym_id", "?"))
        res_seq = int(col_map.get("label_seq_id", col_map.get("auth_seq_id", "0")))
        if chain_id not in chains:
            chains[chain_id] = []
        if res_seq not in chains[chain_id]:
            chains[chain_id].append(res_seq)


def _extract_native_sequence(structure_path: Path, chain_lengths: dict[str, int]) -> dict[str, str]:
    """Extract native one-letter sequences from the structure file."""
    chain_residues: dict[str, dict[int, str]] = {}

    if structure_path.suffix.lower() in (".cif", ".mmcif"):
        _extract_native_cif(structure_path, chain_residues)
    else:
        _extract_native_pdb(structure_path, chain_residues)

    result = {}
    for chain_id in chain_lengths:
        if chain_id in chain_residues:
            ordered = sorted(chain_residues[chain_id].items())
            result[chain_id] = "".join(
                _AA3_TO_1.get(res, "X") for _, res in ordered
            )
    return result


def _extract_native_pdb(path: Path, chain_residues: dict[str, dict[int, str]]) -> None:
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        atom_name = line[12:16].strip()
        if atom_name != "CA":
            continue
        chain_id = line[21]
        res_seq = int(line[22:26].strip())
        res_name = line[17:20].strip()
        if chain_id not in chain_residues:
            chain_residues[chain_id] = {}
        chain_residues[chain_id][res_seq] = res_name


def _extract_native_cif(path: Path, chain_residues: dict[str, dict[int, str]]) -> None:
    in_atom_site = False
    columns: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith("_atom_site."):
            in_atom_site = True
            columns.append(line.split(".")[1].strip())
            continue
        if in_atom_site and (line.startswith("_") or line.startswith("#") or line.startswith("loop_")):
            in_atom_site = False
            columns = []
            continue
        if not in_atom_site or not columns:
            continue

        fields = line.split()
        if len(fields) < len(columns):
            continue

        col_map = dict(zip(columns, fields))
        if col_map.get("label_atom_id", "").strip('"') != "CA":
            continue
        if col_map.get("group_PDB", "") != "ATOM":
            continue

        chain_id = col_map.get("label_asym_id", col_map.get("auth_asym_id", "?"))
        res_seq = int(col_map.get("label_seq_id", col_map.get("auth_seq_id", "0")))
        res_name = col_map.get("label_comp_id", "UNK")
        if chain_id not in chain_residues:
            chain_residues[chain_id] = {}
        chain_residues[chain_id][res_seq] = res_name


def _split_by_chains(sequence: str, chain_lengths: dict[str, int]) -> dict[str, str]:
    """Split a concatenated sequence string into per-chain sequences."""
    result = {}
    offset = 0
    for chain_id, length in chain_lengths.items():
        result[chain_id] = sequence[offset:offset + length]
        offset += length
    return result


def _parse_fasta(fasta_path: Path) -> list[dict]:
    """Parse MPNN FASTA output into a list of design records.

    Expected header format:
        >{name}_b{batch}_d{design}, sequence_recovery={float}

    Optionally also:
        ..., ligand_interface_sequence_recovery={float}
    """
    records = []
    with fasta_path.open() as f:
        header = ""
        sequence = ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header and sequence:
                    records.append(_parse_fasta_record(header, sequence))
                header = line
                sequence = ""
            elif line:
                sequence += line
        if header and sequence:
            records.append(_parse_fasta_record(header, sequence))
    return records


def _parse_fasta_record(header: str, sequence: str) -> dict:
    """Parse a single FASTA header + sequence into a record dict."""
    record = {"sequence": sequence, "recovery": None}

    # Parse sequence_recovery from header
    match = re.search(r"sequence_recovery=([\d.]+)", header)
    if match:
        record["recovery"] = float(match.group(1))

    return record


def standardize(workspace: Path) -> None:
    """Transform raw MPNN outputs into standardized result_data.json."""
    raw_dir = workspace / "outputs" / "raw"
    std_dir = workspace / "outputs" / "standardized"
    config = json.loads((workspace / "config.json").read_text())

    # Get input structure path and chain info. Sort chains alphabetically
    # because the foundry MPNN CLI concatenates output in sorted chain order.
    structure_path = Path(config["structure_path"])
    chain_lengths = dict(sorted(_parse_chain_lengths(structure_path).items()))

    # Parse all FASTA files from raw output
    fasta_files = sorted(raw_dir.glob("*.fa")) + sorted(raw_dir.glob("*.fasta"))
    all_records: list[dict] = []
    for fasta_path in fasta_files:
        all_records.extend(_parse_fasta(fasta_path))

    if not all_records:
        raise RuntimeError(
            f"No FASTA records found in {raw_dir}. "
            f"Files present: {[f.name for f in raw_dir.iterdir()]}"
        )

    # Extract native sequence from input structure
    native_sequence = _extract_native_sequence(structure_path, chain_lengths)

    # Build designed_sequences list
    designed_sequences = []
    for rank, record in enumerate(all_records, start=1):
        chain_seqs = _split_by_chains(record["sequence"], chain_lengths)
        designed_sequences.append({
            "rank": rank,
            "sequence": chain_seqs,
            "score": None,
            "recovery": record["recovery"],
        })

    # Write result_data.json
    result_data = {
        "designed_sequences": designed_sequences,
        "native_sequence": native_sequence if native_sequence else None,
    }
    (std_dir / "result_data.json").write_text(json.dumps(result_data, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standardize MPNN outputs.")
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()
    standardize(args.workspace)
