"""Tests for the apply_mutations.py container script."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the container script
# ---------------------------------------------------------------------------

_CONTAINER_DIR = str(
    Path(__file__).resolve().parent.parent.parent / "containers" / "ligandmpnn-packer"
)


def _import_apply_mutations():
    if _CONTAINER_DIR not in sys.path:
        sys.path.insert(0, _CONTAINER_DIR)
    mod = importlib.import_module("apply_mutations")
    importlib.reload(mod)
    return mod


# ---------------------------------------------------------------------------
# Test PDB content
# ---------------------------------------------------------------------------

_PDB_WITH_SIDECHAINS = (
    "HEADER    TEST\n"
    "ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 10.00           N\n"
    "ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00 10.00           C\n"
    "ATOM      3  C   ALA A   1       3.000   4.000   5.000  1.00 10.00           C\n"
    "ATOM      4  O   ALA A   1       3.500   4.500   5.500  1.00 10.00           O\n"
    "ATOM      5  CB  ALA A   1       2.500   3.500   3.000  1.00 10.00           C\n"
    "ATOM      6  N   GLU E  63       4.000   5.000   6.000  1.00 12.00           N\n"
    "ATOM      7  CA  GLU E  63       5.000   6.000   7.000  1.00 12.00           C\n"
    "ATOM      8  C   GLU E  63       6.000   7.000   8.000  1.00 12.00           C\n"
    "ATOM      9  O   GLU E  63       6.500   7.500   8.500  1.00 12.00           O\n"
    "ATOM     10  CB  GLU E  63       5.500   6.500   6.000  1.00 12.00           C\n"
    "ATOM     11  CG  GLU E  63       5.800   6.800   5.000  1.00 12.00           C\n"
    "ATOM     12  CD  GLU E  63       6.100   7.100   4.000  1.00 12.00           C\n"
    "ATOM     13  OE1 GLU E  63       6.400   7.400   3.500  1.00 12.00           O\n"
    "ATOM     14  OE2 GLU E  63       6.000   7.000   3.000  1.00 12.00           O\n"
    "END\n"
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParseMutation:
    """Tests for the parse_mutation helper."""

    def test_basic(self) -> None:
        mod = _import_apply_mutations()
        wt, chain, resnum, new = mod.parse_mutation("EA63Q")
        assert wt == "E"
        assert chain == "A"
        assert resnum == 63
        assert new == "Q"

    def test_single_digit_resnum(self) -> None:
        mod = _import_apply_mutations()
        wt, chain, resnum, new = mod.parse_mutation("AA1G")
        assert wt == "A"
        assert chain == "A"
        assert resnum == 1
        assert new == "G"

    def test_large_resnum(self) -> None:
        mod = _import_apply_mutations()
        wt, chain, resnum, new = mod.parse_mutation("KB999A")
        assert chain == "B"
        assert resnum == 999

    def test_invalid_format(self) -> None:
        mod = _import_apply_mutations()
        with pytest.raises(ValueError, match="Invalid mutation format"):
            mod.parse_mutation("invalid")


class TestApplyMutations:
    """Tests for the apply_mutations function."""

    def test_single_mutation_changes_resname(self) -> None:
        """Mutation changes residue name from GLU to GLN."""
        mod = _import_apply_mutations()
        result = mod.apply_mutations(_PDB_WITH_SIDECHAINS, ["EE63Q"])

        # GLU should be replaced by GLN
        assert "GLN" in result
        assert "GLU" not in result

    def test_sidechain_atoms_stripped(self) -> None:
        """Sidechain atoms are removed from mutated residues."""
        mod = _import_apply_mutations()
        result = mod.apply_mutations(_PDB_WITH_SIDECHAINS, ["EE63Q"])

        # GLU sidechain atoms (CB, CG, CD, OE1, OE2) should be stripped
        lines = result.splitlines()
        mutated_atoms = [line for line in lines if line.startswith("ATOM") and "E  63" in line]
        atom_names = [line[12:16].strip() for line in mutated_atoms]
        assert set(atom_names) == {"N", "CA", "C", "O"}

    def test_unmutated_residues_preserved(self) -> None:
        """Non-mutated residues are not modified."""
        mod = _import_apply_mutations()
        result = mod.apply_mutations(_PDB_WITH_SIDECHAINS, ["EE63Q"])

        # ALA at A/1 should be unchanged, including its CB
        lines = result.splitlines()
        ala_atoms = [line for line in lines if line.startswith("ATOM") and "ALA A   1" in line]
        assert len(ala_atoms) == 5  # N, CA, C, O, CB

    def test_wt_mismatch_raises(self) -> None:
        """Wrong wild-type amino acid raises ValueError."""
        mod = _import_apply_mutations()
        # E63 is GLU (E), not ALA (A)
        with pytest.raises(ValueError, match="Wild-type mismatch"):
            mod.apply_mutations(_PDB_WITH_SIDECHAINS, ["AE63Q"])

    def test_missing_residue_raises(self) -> None:
        """Mutation referencing nonexistent residue raises ValueError."""
        mod = _import_apply_mutations()
        with pytest.raises(ValueError, match="not found in PDB"):
            mod.apply_mutations(_PDB_WITH_SIDECHAINS, ["AA999G"])

    def test_header_and_end_preserved(self) -> None:
        """Non-ATOM records are preserved."""
        mod = _import_apply_mutations()
        result = mod.apply_mutations(_PDB_WITH_SIDECHAINS, ["EE63Q"])
        assert "HEADER    TEST" in result
        assert "END" in result

    def test_multiple_mutations(self) -> None:
        """Multiple mutations can be applied simultaneously."""
        # Build a PDB with two mutable residues
        pdb = (
            "ATOM      1  N   ALA A   1       1.0   2.0   3.0  1.00 10.0           N\n"
            "ATOM      2  CA  ALA A   1       2.0   3.0   4.0  1.00 10.0           C\n"
            "ATOM      3  C   ALA A   1       3.0   4.0   5.0  1.00 10.0           C\n"
            "ATOM      4  O   ALA A   1       3.5   4.5   5.5  1.00 10.0           O\n"
            "ATOM      5  CB  ALA A   1       2.5   3.5   3.0  1.00 10.0           C\n"
            "ATOM      6  N   LYS B  10       4.0   5.0   6.0  1.00 12.0           N\n"
            "ATOM      7  CA  LYS B  10       5.0   6.0   7.0  1.00 12.0           C\n"
            "ATOM      8  C   LYS B  10       6.0   7.0   8.0  1.00 12.0           C\n"
            "ATOM      9  O   LYS B  10       6.5   7.5   8.5  1.00 12.0           O\n"
            "ATOM     10  CB  LYS B  10       5.5   6.5   6.0  1.00 12.0           C\n"
            "ATOM     11  CG  LYS B  10       5.8   6.8   5.0  1.00 12.0           C\n"
            "END\n"
        )
        mod = _import_apply_mutations()
        result = mod.apply_mutations(pdb, ["AA1G", "KB10A"])

        assert "GLY" in result
        assert "ALA" in result  # B10 mutated to ALA
        assert "LYS" not in result

    def test_mutation_to_glycine_strips_cb(self) -> None:
        """Mutation to glycine strips CB (glycine has no sidechain)."""
        mod = _import_apply_mutations()
        pdb = (
            "ATOM      1  N   ALA A   1       1.0   2.0   3.0  1.00 10.0           N\n"
            "ATOM      2  CA  ALA A   1       2.0   3.0   4.0  1.00 10.0           C\n"
            "ATOM      3  C   ALA A   1       3.0   4.0   5.0  1.00 10.0           C\n"
            "ATOM      4  O   ALA A   1       3.5   4.5   5.5  1.00 10.0           O\n"
            "ATOM      5  CB  ALA A   1       2.5   3.5   3.0  1.00 10.0           C\n"
            "END\n"
        )
        result = mod.apply_mutations(pdb, ["AA1G"])

        lines = [line for line in result.splitlines() if line.startswith("ATOM")]
        atom_names = [line[12:16].strip() for line in lines]
        assert "CB" not in atom_names
        assert set(atom_names) == {"N", "CA", "C", "O"}
