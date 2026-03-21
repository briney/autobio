"""FASTA parsing, writing, and sequence validation (pure Python)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Alphabet constants
# ---------------------------------------------------------------------------

AMINO_ACIDS: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")
DNA_BASES: frozenset[str] = frozenset("ACGT")
RNA_BASES: frozenset[str] = frozenset("ACGU")

# ---------------------------------------------------------------------------
# FASTA I/O
# ---------------------------------------------------------------------------

_FASTA_LINE_WIDTH = 80


def parse_fasta(path: Path) -> dict[str, str]:
    """Parse a FASTA file into ``{header: sequence}`` pairs.

    Multi-line sequences are concatenated.  The leading ``>`` is stripped
    from header lines.

    Args:
        path: Path to the FASTA file.

    Returns:
        Ordered mapping of header strings to sequence strings.
    """
    sequences: dict[str, str] = {}
    current_header: str | None = None
    chunks: list[str] = []

    with open(path) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    sequences[current_header] = "".join(chunks)
                current_header = line[1:].strip()
                chunks = []
            else:
                chunks.append(line)

    if current_header is not None:
        sequences[current_header] = "".join(chunks)

    return sequences


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
