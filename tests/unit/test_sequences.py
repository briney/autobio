"""Tests for autobio.utils.sequences."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - used in fixture type hints

from autobio.utils.sequences import (
    AMINO_ACIDS,
    DNA_BASES,
    RNA_BASES,
    parse_fasta,
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


class TestAlphabets:
    def test_amino_acids_count(self) -> None:
        assert len(AMINO_ACIDS) == 20

    def test_dna_bases_count(self) -> None:
        assert len(DNA_BASES) == 4

    def test_rna_bases_count(self) -> None:
        assert len(RNA_BASES) == 4
