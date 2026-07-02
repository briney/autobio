"""Tests for autobio.utils.sequences."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - used in fixture type hints

from autobio.utils.sequences import (
    AMINO_ACIDS,
    ANTIBODY_AMINO_ACIDS,
    DNA_BASES,
    RNA_BASES,
    parse_fasta,
    parse_fasta_string,
    validate_antibody_sequence,
    validate_nucleotide_sequence,
    validate_protein_sequence,
    write_fasta,
)

# ---------------------------------------------------------------------------
# FASTA round-trip
# ---------------------------------------------------------------------------


class TestFastaIO:
    def test_parse_single_sequence(self, tmp_path: Path) -> None:
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">seq1\nMKWVTFISL\n")
        result = parse_fasta(fasta)
        assert result == {"seq1": "MKWVTFISL"}

    def test_parse_multiline_sequence(self, tmp_path: Path) -> None:
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">seq1\nMKWV\nTFISL\n")
        result = parse_fasta(fasta)
        assert result == {"seq1": "MKWVTFISL"}

    def test_parse_multiple_sequences(self, tmp_path: Path) -> None:
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">s1\nAAAA\n>s2\nCCCC\n")
        result = parse_fasta(fasta)
        assert list(result.keys()) == ["s1", "s2"]
        assert result["s1"] == "AAAA"
        assert result["s2"] == "CCCC"

    def test_parse_ignores_blank_lines(self, tmp_path: Path) -> None:
        fasta = tmp_path / "test.fasta"
        fasta.write_text(">s1\nAA\n\nCC\n")
        result = parse_fasta(fasta)
        assert result["s1"] == "AACC"

    def test_write_and_reparse(self, tmp_path: Path) -> None:
        seqs = {"protein_A": "MKWVTFISL", "protein_B": "ACDEFGHIKLMNPQRSTVWY"}
        out = tmp_path / "out.fasta"
        write_fasta(seqs, out)
        reparsed = parse_fasta(out)
        assert reparsed == seqs

    def test_write_wraps_long_lines(self, tmp_path: Path) -> None:
        long_seq = "A" * 200
        out = tmp_path / "out.fasta"
        write_fasta({"long": long_seq}, out)
        lines = out.read_text().strip().split("\n")
        # header + 3 sequence lines (80 + 80 + 40)
        assert len(lines) == 4
        assert len(lines[1]) == 80


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestProteinValidation:
    def test_valid_protein(self) -> None:
        assert validate_protein_sequence("MKWVTFISL") is True

    def test_lowercase_valid(self) -> None:
        assert validate_protein_sequence("mkwv") is True

    def test_invalid_character(self) -> None:
        assert validate_protein_sequence("MKWX") is False

    def test_empty_string(self) -> None:
        assert validate_protein_sequence("") is False

    def test_numbers_invalid(self) -> None:
        assert validate_protein_sequence("MK123") is False


class TestNucleotideValidation:
    def test_valid_dna(self) -> None:
        assert validate_nucleotide_sequence("ACGTACGT", "DNA") is True

    def test_valid_rna(self) -> None:
        assert validate_nucleotide_sequence("ACGUACGU", "RNA") is True

    def test_dna_rejects_uracil(self) -> None:
        assert validate_nucleotide_sequence("ACGU", "DNA") is False

    def test_rna_rejects_thymine(self) -> None:
        assert validate_nucleotide_sequence("ACGT", "RNA") is False

    def test_empty_string(self) -> None:
        assert validate_nucleotide_sequence("", "DNA") is False

    def test_lowercase(self) -> None:
        assert validate_nucleotide_sequence("acgt", "DNA") is True


class TestAntibodyValidation:
    def test_standard_aa_accepted(self) -> None:
        assert validate_antibody_sequence("EVQLVESGGGLVQPGG") is True

    def test_ambiguous_residues_accepted(self) -> None:
        assert validate_antibody_sequence("EVQLXBZ") is True

    def test_lowercase_accepted(self) -> None:
        assert validate_antibody_sequence("evqlv") is True

    def test_invalid_character_rejected(self) -> None:
        assert validate_antibody_sequence("EVQL@V") is False

    def test_numbers_rejected(self) -> None:
        assert validate_antibody_sequence("EVQL123") is False

    def test_empty_string_rejected(self) -> None:
        assert validate_antibody_sequence("") is False

    def test_all_ambiguous_codes(self) -> None:
        """All ambiguous residue codes (B, O, U, X, Z) are accepted."""
        assert validate_antibody_sequence("BOUXZ") is True


class TestAlphabets:
    def test_amino_acids_count(self) -> None:
        assert len(AMINO_ACIDS) == 20

    def test_antibody_amino_acids_superset(self) -> None:
        """Antibody alphabet is a superset of the standard alphabet."""
        assert AMINO_ACIDS < ANTIBODY_AMINO_ACIDS

    def test_antibody_amino_acids_includes_ambiguous(self) -> None:
        for code in "BOUXZ":
            assert code in ANTIBODY_AMINO_ACIDS

    def test_dna_bases_count(self) -> None:
        assert len(DNA_BASES) == 4

    def test_rna_bases_count(self) -> None:
        assert len(RNA_BASES) == 4


# ---------------------------------------------------------------------------
# Generic FASTA string parsing
# ---------------------------------------------------------------------------


def test_parse_fasta_string_basic():
    text = ">a\nMKT\nVLL\n>b\nGGG\n"
    assert parse_fasta_string(text) == {"a": "MKTVLL", "b": "GGG"}


def test_parse_fasta_string_rejects_duplicate_ids():
    import pytest

    with pytest.raises(ValueError, match="[Dd]uplicate.*'a'"):
        parse_fasta_string(">a\nMKT\n>a\nGGG\n")


def test_parse_fasta_string_rejects_sequence_before_header():
    import pytest

    with pytest.raises(ValueError, match="before any header"):
        parse_fasta_string("MKT\n>a\nGGG\n")


def test_parse_fasta_string_ignores_blank_lines():
    text = "\n>a\nMKT\n\n\n>b\nGGG\n\n"
    assert parse_fasta_string(text) == {"a": "MKT", "b": "GGG"}
