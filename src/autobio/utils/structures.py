"""Pure-Python helpers for reading PDB and mmCIF structure files."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Three-letter to one-letter amino-acid mapping
# ---------------------------------------------------------------------------

THREE_TO_ONE: dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

# ---------------------------------------------------------------------------
# PDB parsing
# ---------------------------------------------------------------------------


def read_pdb_sequences(path: Path) -> dict[str, str]:
    """Extract per-chain amino-acid sequences from a PDB file.

    Only ``ATOM`` records with standard residue names are considered.
    Residues are deduped by ``(chain_id, residue_number, insertion_code)``.

    Args:
        path: Path to a PDB file.

    Returns:
        Mapping of chain ID to one-letter sequence string.
    """
    chains: dict[str, list[tuple[int, str, str]]] = {}

    with open(path) as fh:
        for line in fh:
            if not line.startswith("ATOM  "):
                continue
            res_name = line[17:20].strip()
            if res_name not in THREE_TO_ONE:
                continue
            chain_id = line[21]
            res_seq = int(line[22:26].strip())
            icode = line[26].strip()
            key = (res_seq, icode, THREE_TO_ONE[res_name])
            chains.setdefault(chain_id, [])
            if key not in chains[chain_id]:
                chains[chain_id].append(key)

    return {chain: "".join(aa for _, _, aa in residues) for chain, residues in chains.items()}


# ---------------------------------------------------------------------------
# mmCIF parsing
# ---------------------------------------------------------------------------


def read_mmcif_sequences(path: Path) -> dict[str, str]:
    """Extract per-chain amino-acid sequences from an mmCIF file.

    Parses ``_atom_site`` loop records.  Only ``ATOM`` records with
    standard residue names are included.

    Args:
        path: Path to an mmCIF file.

    Returns:
        Mapping of chain ID (``auth_asym_id``) to one-letter sequence.
    """
    columns: list[str] = []
    in_atom_site = False
    reading_header = False
    chains: dict[str, list[tuple[int, str, str]]] = {}

    with open(path) as fh:
        for raw_line in fh:
            line = raw_line.strip()

            if line == "loop_":
                reading_header = True
                columns = []
                in_atom_site = False
                continue

            if reading_header:
                if line.startswith("_atom_site."):
                    in_atom_site = True
                    columns.append(line.split(".")[1])
                    continue
                if in_atom_site and not line.startswith("_"):
                    reading_header = False
                    # fall through to process this data line
                else:
                    if not line.startswith("_atom_site."):
                        reading_header = False
                        in_atom_site = False
                    continue

            if not in_atom_site:
                if line.startswith("loop_"):
                    reading_header = True
                    columns = []
                    in_atom_site = False
                continue

            if line.startswith("#"):
                in_atom_site = False
                continue

            tokens = line.split()
            if len(tokens) < len(columns):
                continue

            col_map = dict(zip(columns, tokens, strict=False))
            group = col_map.get("group_PDB", "")
            if group != "ATOM":
                continue

            res_name = col_map.get("label_comp_id", "")
            if res_name not in THREE_TO_ONE:
                continue

            chain_id = col_map.get("auth_asym_id", col_map.get("label_asym_id", "?"))
            res_seq = int(col_map.get("auth_seq_id", col_map.get("label_seq_id", "0")))
            icode = col_map.get("pdbx_PDB_ins_code", "").rstrip("?.")

            key = (res_seq, icode, THREE_TO_ONE[res_name])
            chains.setdefault(chain_id, [])
            if key not in chains[chain_id]:
                chains[chain_id].append(key)

    return {chain: "".join(aa for _, _, aa in residues) for chain, residues in chains.items()}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def detect_structure_format(path: Path) -> str:
    """Detect whether *path* is a PDB or mmCIF file.

    Detection uses the file extension, falling back to content inspection.

    Returns:
        ``"pdb"`` or ``"mmcif"``.

    Raises:
        ValueError: If the format cannot be determined.
    """
    suffix = path.suffix.lower()
    if suffix in (".pdb", ".ent"):
        return "pdb"
    if suffix in (".cif", ".mmcif"):
        return "mmcif"

    with open(path) as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("data_"):
                return "mmcif"
            if stripped.startswith(("ATOM", "HETATM", "HEADER", "REMARK")):
                return "pdb"
            break

    raise ValueError(f"Cannot determine structure format of {path}")


def count_residues(path: Path) -> int:
    """Count total residues across all chains in a structure file.

    Args:
        path: PDB or mmCIF file.
    """
    fmt = detect_structure_format(path)
    sequences = read_pdb_sequences(path) if fmt == "pdb" else read_mmcif_sequences(path)
    return sum(len(seq) for seq in sequences.values())
