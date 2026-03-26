"""Tests for autobio.utils.structures."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - used in fixture type hints

import pytest

from autobio.utils.structures import (
    THREE_TO_ONE,
    count_residues,
    detect_structure_format,
    read_mmcif_sequences,
    read_pdb_sequences,
)

# Minimal PDB fragment (chain A, 3 residues: MET, LYS, TRP)
MINI_PDB = """\
ATOM      1  N   MET A   1       1.000   2.000   3.000  1.00  0.00           N
ATOM      2  CA  MET A   1       2.000   3.000   4.000  1.00  0.00           C
ATOM      3  N   LYS A   2       3.000   4.000   5.000  1.00  0.00           N
ATOM      4  CA  LYS A   2       4.000   5.000   6.000  1.00  0.00           C
ATOM      5  N   TRP A   3       5.000   6.000   7.000  1.00  0.00           N
ATOM      6  CA  TRP A   3       6.000   7.000   8.000  1.00  0.00           C
END
"""

# Minimal mmCIF fragment
MINI_CIF = """\
data_test
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_seq_id
_atom_site.auth_asym_id
_atom_site.auth_seq_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_PDB_ins_code
ATOM 1 N N MET A 1 A 1 1.000 2.000 3.000 1.00 0.00 .
ATOM 2 C CA MET A 1 A 1 2.000 3.000 4.000 1.00 0.00 .
ATOM 3 N N ALA A 2 A 2 3.000 4.000 5.000 1.00 0.00 .
ATOM 4 C CA ALA A 2 A 2 4.000 5.000 6.000 1.00 0.00 .
#
"""


class TestReadPdbSequences:
    def test_extracts_chain_sequence(self, tmp_path: Path) -> None:
        pdb = tmp_path / "test.pdb"
        pdb.write_text(MINI_PDB)
        seqs = read_pdb_sequences(pdb)
        assert seqs == {"A": "MKW"}

    def test_multi_chain(self, tmp_path: Path) -> None:
        pdb_text = MINI_PDB + (
            "ATOM      7  N   GLY B   1       7.000   8.000   9.000  1.00  0.00           N\n"
            "ATOM      8  CA  GLY B   1       8.000   9.000  10.000  1.00  0.00           C\n"
        )
        pdb = tmp_path / "multi.pdb"
        pdb.write_text(pdb_text)
        seqs = read_pdb_sequences(pdb)
        assert "A" in seqs
        assert "B" in seqs
        assert seqs["B"] == "G"


class TestReadMmcifSequences:
    def test_extracts_chain_sequence(self, tmp_path: Path) -> None:
        cif = tmp_path / "test.cif"
        cif.write_text(MINI_CIF)
        seqs = read_mmcif_sequences(cif)
        assert seqs == {"A": "MA"}


class TestDetectFormat:
    def test_pdb_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "model.pdb"
        p.write_text("ATOM  ...")
        assert detect_structure_format(p) == "pdb"

    def test_cif_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "model.cif"
        p.write_text("data_test")
        assert detect_structure_format(p) == "mmcif"

    def test_ent_extension(self, tmp_path: Path) -> None:
        p = tmp_path / "model.ent"
        p.write_text("")
        assert detect_structure_format(p) == "pdb"

    def test_content_detection_pdb(self, tmp_path: Path) -> None:
        p = tmp_path / "model.txt"
        p.write_text("HEADER  test\n")
        assert detect_structure_format(p) == "pdb"

    def test_content_detection_mmcif(self, tmp_path: Path) -> None:
        p = tmp_path / "model.txt"
        p.write_text("data_test\n")
        assert detect_structure_format(p) == "mmcif"

    def test_unknown_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "model.xyz"
        p.write_text("some random content\n")
        with pytest.raises(ValueError, match="Cannot determine"):
            detect_structure_format(p)


class TestCountResidues:
    def test_count_pdb(self, tmp_path: Path) -> None:
        pdb = tmp_path / "test.pdb"
        pdb.write_text(MINI_PDB)
        assert count_residues(pdb) == 3

    def test_count_mmcif(self, tmp_path: Path) -> None:
        cif = tmp_path / "test.cif"
        cif.write_text(MINI_CIF)
        assert count_residues(cif) == 2


class TestThreeToOne:
    def test_all_20_standard_amino_acids(self) -> None:
        assert len(THREE_TO_ONE) == 20

    def test_known_mappings(self) -> None:
        assert THREE_TO_ONE["ALA"] == "A"
        assert THREE_TO_ONE["TRP"] == "W"
        assert THREE_TO_ONE["MET"] == "M"
