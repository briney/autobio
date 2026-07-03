"""FASTA parsing, writing, and sequence validation (pure Python)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from autobio.schemas.antibody_types import AntibodySequence

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


# ---------------------------------------------------------------------------
# Antibody FASTA pairing
# ---------------------------------------------------------------------------

_HEAVY_TOKENS = {"heavy", "h", "vh"}
_LIGHT_TOKENS = {"light", "l", "vl"}


def normalize_chain_token(token: str) -> str:
    """Map a case-insensitive chain token to ``"heavy"`` or ``"light"``.

    Accepts aliases ``heavy``/``h``/``vh`` and ``light``/``l``/``vl``.

    Args:
        token: Raw chain token parsed from a FASTA header (case-insensitive,
            surrounding whitespace tolerated).

    Returns:
        Either ``"heavy"`` or ``"light"``.

    Raises:
        ValueError: If the token is not a recognized chain identifier.
    """
    key = token.strip().lower()
    if key in _HEAVY_TOKENS:
        return "heavy"
    if key in _LIGHT_TOKENS:
        return "light"
    raise ValueError(
        f"Unknown chain token {token!r}. Expected one of {sorted(_HEAVY_TOKENS | _LIGHT_TOKENS)}."
    )


def parse_antibody_fasta_string(text: str) -> list[AntibodySequence]:
    """Parse antibody FASTA text into paired :class:`AntibodySequence` objects.

    Headers encode a pair id and a chain: ``>{pair_id}|{chain}``. Records sharing
    a ``pair_id`` are paired into one antibody; a lone record becomes an unpaired
    antibody (that chain only).

    Args:
        text: Raw FASTA content with ``>{pair_id}|{chain}`` headers.

    Returns:
        List of :class:`AntibodySequence` objects, one per distinct ``pair_id``,
        in first-seen order.

    Raises:
        ValueError: For a header without a ``|`` chain tag, an unknown chain
            token, a duplicate ``(pair_id, chain)``, or a non-protein sequence.
            Every raised error names the offending record's header.
    """
    raw = parse_fasta_string(text)
    chains: dict[str, dict[str, str]] = {}
    order: list[str] = []
    for header, seq in raw.items():
        if "|" not in header:
            raise ValueError(
                f"Antibody FASTA header {header!r} is missing a chain tag "
                f"(expected '{{pair_id}}|{{chain}}')."
            )
        pair_id, _, chain_token = header.rpartition("|")
        pair_id = pair_id.strip()
        try:
            chain = normalize_chain_token(chain_token)
        except ValueError as exc:
            raise ValueError(f"Record {header!r}: {exc}") from exc
        if not validate_antibody_sequence(seq):
            raise ValueError(f"Record {header!r}: sequence contains non-protein characters.")
        if pair_id not in chains:
            chains[pair_id] = {}
            order.append(pair_id)
        if chain in chains[pair_id]:
            raise ValueError(f"Duplicate record for pair {pair_id!r} chain {chain!r}.")
        chains[pair_id][chain] = seq

    return [
        AntibodySequence(
            id=pair_id,
            heavy_chain=chains[pair_id].get("heavy"),
            light_chain=chains[pair_id].get("light"),
        )
        for pair_id in order
    ]
