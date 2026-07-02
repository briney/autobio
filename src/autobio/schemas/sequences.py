"""SequenceSet input types — accept structured JSON, FASTA text, or a FASTA file.

Each SequenceSet's field type is the canonical structured form (so existing JSON
callers and agents are unaffected); a ``BeforeValidator`` additionally accepts
FASTA text or a ``.fasta``/``.fa`` file path and normalizes it centrally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import BeforeValidator

from autobio.utils.sequences import parse_fasta, parse_fasta_string

_FASTA_SUFFIXES = (".fasta", ".fa")


def _looks_like_fasta_text(value: str) -> bool:
    stripped = value.lstrip()
    return stripped.startswith(">") or "\n" in value


def _looks_like_fasta_path(value: str) -> bool:
    return value.strip().lower().endswith(_FASTA_SUFFIXES)


def normalize_generic_sequences(value: object) -> dict[str, str]:
    """Normalize a generic sequence input to a ``{id: sequence}`` mapping.

    Accepts a native ``dict[str, str]``, FASTA text, or a ``.fasta``/``.fa`` path.

    Raises:
        ValueError: For an unsupported input type, or a path that does not exist.
    """
    if isinstance(value, dict):
        return {str(k): str(v) for k, v in value.items()}
    if isinstance(value, str):
        if _looks_like_fasta_text(value):
            return parse_fasta_string(value)
        if _looks_like_fasta_path(value):
            path = Path(value)
            if not path.is_file():
                raise ValueError(f"FASTA file not found: {value!r}.")
            return parse_fasta(path)
        raise ValueError(
            "String sequence input must be FASTA text (starting with '>' or "
            "multi-line) or a path to a .fasta/.fa file."
        )
    raise ValueError(
        f"Unsupported sequence input type {type(value).__name__!r}; expected a "
        "dict, FASTA text, or a FASTA file path."
    )


GenericSequenceSet = Annotated[dict[str, str], BeforeValidator(normalize_generic_sequences)]
"""Field type accepting ``dict[str, str]``, FASTA text, or a FASTA file path."""
