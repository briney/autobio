"""FASTA parsing, writing, and sequence validation (pure Python)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Alphabet constants
# ---------------------------------------------------------------------------

AMINO_ACIDS: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")
ANTIBODY_AMINO_ACIDS: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWYBOUXZ")
DNA_BASES: frozenset[str] = frozenset("ACGT")
RNA_BASES: frozenset[str] = frozenset("ACGU")

# ---------------------------------------------------------------------------
# FASTA I/O
# ---------------------------------------------------------------------------

_FASTA_LINE_WIDTH = 80


def parse_fasta_string(text: str) -> dict[str, str]:
    """Parse FASTA text into an insertion-ordered ``{id: sequence}`` mapping.

    Args:
        text: Raw FASTA content. ``>``-prefixed lines are headers (the leading
            ``>`` is stripped); subsequent non-blank lines are concatenated.

    Returns:
        Mapping of header id to sequence, in first-seen order.

    Raises:
        ValueError: On a duplicate id, or a sequence line before any header.
    """
    sequences: dict[str, str] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            current = line[1:].strip()
            if current in sequences:
                raise ValueError(f"Duplicate sequence id {current!r} in FASTA input.")
            sequences[current] = ""
        else:
            if current is None:
                raise ValueError("FASTA sequence data appears before any header line.")
            sequences[current] += line
    return sequences


def parse_fasta(path: Path) -> dict[str, str]:
    """Parse a FASTA file into an insertion-ordered ``{id: sequence}`` mapping.

    Args:
        path: Path to a FASTA file.

    Returns:
        Mapping of header id to sequence, in first-seen order.

    Raises:
        ValueError: On a duplicate id, or a sequence line before any header.
    """
    return parse_fasta_string(path.read_text())


def write_fasta(sequences: dict[str, str], path: Path) -> None:
    """Write sequences to a FASTA file.

    Args:
        sequences: Mapping of header to sequence string.
        path: Destination file path.
    """
    with open(path, "w") as fh:
        for header, seq in sequences.items():
            fh.write(f">{header}\n")
            for i in range(0, len(seq), _FASTA_LINE_WIDTH):
                fh.write(seq[i : i + _FASTA_LINE_WIDTH] + "\n")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_protein_sequence(seq: str) -> bool:
    """Return ``True`` if *seq* contains only standard amino acid characters."""
    return len(seq) > 0 and set(seq.upper()) <= AMINO_ACIDS


def validate_antibody_sequence(seq: str) -> bool:
    """Return ``True`` if *seq* contains only amino acid characters valid for antibody LMs.

    Accepts the 20 standard amino acids plus ambiguous residue codes
    (B, O, U, X, Z) recognised by CurrAb and BALM tokenizers.
    """
    return len(seq) > 0 and set(seq.upper()) <= ANTIBODY_AMINO_ACIDS


def validate_nucleotide_sequence(
    seq: str,
    molecule: str = "DNA",
) -> bool:
    """Return ``True`` if *seq* contains only valid nucleotide characters.

    Args:
        seq: Nucleotide sequence string.
        molecule: ``"DNA"`` or ``"RNA"``.
    """
    alphabet = DNA_BASES if molecule.upper() == "DNA" else RNA_BASES
    return len(seq) > 0 and set(seq.upper()) <= alphabet
