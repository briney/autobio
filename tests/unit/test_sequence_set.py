"""Tests for SequenceSet accepting input types."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel, ValidationError

from autobio.schemas.antibody import AntibodySequence
from autobio.schemas.sequences import (
    AntibodySequenceSet,
    GenericSequenceSet,
    normalize_generic_sequences,
)

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


class _Ab(BaseModel):
    sequences: AntibodySequenceSet


def test_antibody_accepts_native_list_of_models() -> None:
    ab = AntibodySequence(id="ab1", heavy_chain="QVQLVQSG")
    assert _Ab(sequences=[ab]).sequences == [ab]


def test_antibody_accepts_list_of_dicts() -> None:
    got = _Ab(sequences=[{"id": "ab1", "heavy_chain": "QVQLVQSG"}]).sequences
    assert got == [AntibodySequence(id="ab1", heavy_chain="QVQLVQSG")]


def test_antibody_accepts_fasta_text_with_pairing() -> None:
    text = ">ab1|heavy\nQVQLVQSG\n>ab1|light\nDIQMTQSP\n"
    assert _Ab(sequences=text).sequences == [
        AntibodySequence(id="ab1", heavy_chain="QVQLVQSG", light_chain="DIQMTQSP")
    ]


def test_antibody_accepts_fasta_file(tmp_path: Path) -> None:
    f = tmp_path / "ab.fa"
    f.write_text(">ab1|heavy\nQVQLVQSG\n")
    assert _Ab(sequences=str(f)).sequences == [AntibodySequence(id="ab1", heavy_chain="QVQLVQSG")]


def test_antibody_rejects_bad_type() -> None:
    with pytest.raises((ValidationError, ValueError)):
        _Ab(sequences=42)
