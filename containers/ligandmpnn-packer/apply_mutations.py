#!/usr/bin/env python3
"""Apply amino acid mutations to a PDB file.

Reads mutation specifications (e.g., 'EA63Q' = chain E, Ala-63 -> Gln),
modifies residue types, and strips sidechain atoms from mutated residues
so the packing model can predict new sidechain coordinates.

This script performs pure PDB string manipulation with no ML dependencies.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Standard amino acid 1-letter -> 3-letter mapping
_AA_1TO3: dict[str, str] = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}

_AA_3TO1: dict[str, str] = {v: k for k, v in _AA_1TO3.items()}

# Backbone atom names that are preserved during mutation
_BACKBONE_ATOMS = {"N", "CA", "C", "O"}

# Mutation format: WT_AA (1-letter) + chain (letter) + resnum (digits) + new_AA (1-letter)
_MUTATION_RE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])([A-Za-z])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")


def parse_mutation(mutation_str: str) -> tuple[str, str, int, str]:
    """Parse a mutation string into (wt_aa_1, chain, resnum, new_aa_1).

    Args:
        mutation_str: e.g., 'EA63Q'

    Returns:
        Tuple of (wt_aa_1letter, chain_id, residue_number, new_aa_1letter).
    """
    m = _MUTATION_RE.match(mutation_str)
    if not m:
        raise ValueError(f"Invalid mutation format: {mutation_str!r}")
    wt_aa, chain, resnum, new_aa = m.group(1), m.group(2), int(m.group(3)), m.group(4)
    return wt_aa, chain, resnum, new_aa


def apply_mutations(pdb_text: str, mutations: list[str]) -> str:
    """Apply mutations to PDB text, stripping old sidechains at mutation sites.

    Args:
        pdb_text: Raw PDB file content.
        mutations: List of mutation strings (e.g., ['EA63Q', 'KB42A']).

    Returns:
        Modified PDB text with mutations applied.

    Raises:
        ValueError: If a mutation references a residue not found in the PDB
            or if the wild-type amino acid does not match.
    """
    # Parse all mutations into a lookup: (chain, resnum) -> (wt_aa_1, new_aa_3)
    mutation_map: dict[tuple[str, int], tuple[str, str]] = {}
    for mut_str in mutations:
        wt_aa, chain, resnum, new_aa = parse_mutation(mut_str)
        mutation_map[(chain.upper(), resnum)] = (wt_aa, _AA_1TO3[new_aa])

    # Track which mutations were found in the PDB
    found: set[tuple[str, int]] = set()

    output_lines: list[str] = []
    for line in pdb_text.splitlines(keepends=True):
        record = line[:6].strip()

        if record not in ("ATOM", "HETATM"):
            output_lines.append(line)
            continue

        # Parse PDB ATOM fields (fixed-width format)
        chain_id = line[21].upper()
        try:
            resnum = int(line[22:26].strip())
        except ValueError:
            output_lines.append(line)
            continue
        resname = line[17:20].strip()
        atom_name = line[12:16].strip()

        key = (chain_id, resnum)
        if key not in mutation_map:
            output_lines.append(line)
            continue

        wt_aa_expected, new_resname = mutation_map[key]

        # Validate wild-type amino acid on first encounter
        if key not in found:
            actual_wt = _AA_3TO1.get(resname)
            if actual_wt is None:
                raise ValueError(
                    f"Residue {chain_id}{resnum} has non-standard residue name "
                    f"'{resname}', cannot verify wild-type identity for mutation."
                )
            if actual_wt != wt_aa_expected:
                raise ValueError(
                    f"Wild-type mismatch at {chain_id}{resnum}: expected "
                    f"{wt_aa_expected} ({_AA_1TO3[wt_aa_expected]}), "
                    f"found {actual_wt} ({resname})."
                )
            found.add(key)

        # Strip sidechain atoms (keep only backbone)
        if atom_name not in _BACKBONE_ATOMS:
            continue

        # Replace residue name in the line (columns 17-20, right-justified in 3 chars)
        new_line = line[:17] + f"{new_resname:>3}" + line[20:]
        output_lines.append(new_line)

    # Verify all mutations were found
    missing = set(mutation_map.keys()) - found
    if missing:
        missing_str = ", ".join(f"{c}{r}" for c, r in sorted(missing))
        raise ValueError(f"Mutations reference residues not found in PDB: {missing_str}")

    return "".join(output_lines)


def main() -> None:
    """Read config.json, apply mutations, write mutated PDB."""
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/workspace/config.json")
    config = json.loads(config_path.read_text())

    structure_path = Path(config["structure_path"])
    mutations = config["mutations"]

    pdb_text = structure_path.read_text()
    mutated_text = apply_mutations(pdb_text, mutations)

    out_path = Path("/workspace/inputs/mutated.pdb")
    out_path.write_text(mutated_text)
    print(f"[apply_mutations] Applied {len(mutations)} mutation(s), wrote {out_path}")


if __name__ == "__main__":
    main()
