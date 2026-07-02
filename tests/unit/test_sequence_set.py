"""Tests for SequenceSet accepting input types."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel, ValidationError

from autobio.schemas.sequences import GenericSequenceSet, normalize_generic_sequences

if TYPE_CHECKING:
    from pathlib import Path


class _Generic(BaseModel):
    sequences: GenericSequenceSet


def test_generic_accepts_native_dict() -> None:
    assert _Generic(sequences={"a": "MKT"}).sequences == {"a": "MKT"}


def test_generic_accepts_fasta_text() -> None:
    assert _Generic(sequences=">a\nMKT\n>b\nGGG\n").sequences == {"a": "MKT", "b": "GGG"}


def test_generic_accepts_fasta_file(tmp_path: Path) -> None:
    f = tmp_path / "seqs.fasta"
    f.write_text(">a\nMKT\n")
    assert _Generic(sequences=str(f)).sequences == {"a": "MKT"}


def test_generic_rejects_missing_file() -> None:
    with pytest.raises((ValidationError, ValueError)):
        _Generic(sequences="/no/such/path.fasta")


def test_generic_rejects_bad_type() -> None:
    with pytest.raises((ValidationError, ValueError)):
        _Generic(sequences=12345)


def test_normalize_is_idempotent_on_dict() -> None:
    assert normalize_generic_sequences({"a": "MKT"}) == {"a": "MKT"}
