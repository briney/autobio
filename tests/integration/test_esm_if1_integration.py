"""Integration tests for ESM-IF1 inverse folding and scoring runners.

These tests require Docker and a GPU. They download PDB structures from RCSB,
build and run the autobio-esm-if1 container, and verify end-to-end output.

Run with:
    pytest tests/integration/test_esm_if1_integration.py -v -m docker
"""

from __future__ import annotations

import urllib.request
from typing import TYPE_CHECKING

import pytest

from autobio.core.config import AutobioConfig
from autobio.schemas.inverse_folding import InverseFoldingInput, InverseFoldingOutput
from autobio.schemas.scoring import ScoringInput, ScoringOutput
from autobio.tools import get_runner

if TYPE_CHECKING:
    from pathlib import Path

# All tests in this module require Docker and GPU
pytestmark = [pytest.mark.docker, pytest.mark.gpu]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def rcsb_pdb(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download 1CRN (crambin, 46 residues, single chain) from RCSB."""
    cache = tmp_path_factory.mktemp("pdb_cache")
    pdb_path = cache / "1crn.pdb"
    urllib.request.urlretrieve(  # noqa: S310
        "https://files.rcsb.org/download/1CRN.pdb",
        pdb_path,
    )
    return pdb_path


@pytest.fixture(scope="session")
def rcsb_fab_pdb(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download 7FAB (Fab fragment, 2 chains: H ~209 res, L ~204 res) from RCSB."""
    cache = tmp_path_factory.mktemp("pdb_cache")
    pdb_path = cache / "7fab.pdb"
    urllib.request.urlretrieve(  # noqa: S310
        "https://files.rcsb.org/download/7FAB.pdb",
        pdb_path,
    )
    return pdb_path


@pytest.fixture(scope="session")
def autobio_config() -> AutobioConfig:
    return AutobioConfig.resolve()


# ---------------------------------------------------------------------------
# ESM-IF1 Design — single chain (1CRN)
# ---------------------------------------------------------------------------


class TestESMIF1DesignSingleChain:
    """Full pipeline tests for ESM-IF1 inverse folding on single-chain protein."""

    def test_full_pipeline(
        self, rcsb_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Run ESM-IF1 end-to-end and verify output structure."""
        input_data = InverseFoldingInput(
            structure_path=rcsb_pdb,
            num_sequences=2,
            temperature=0.1,
        )
        runner = get_runner("esm_if1", autobio_config)
        output = runner.run(input_data, gpu="auto", mode="design", output_dir=tmp_path / "ws")

        assert isinstance(output, InverseFoldingOutput)
        assert len(output.designed_sequences) == 2
        assert output.metadata.tool_name == "esm_if1"
        assert output.metadata.mode == "design"
        assert output.metadata.wall_time_seconds > 0

        for seq in output.designed_sequences:
            assert "A" in seq.sequence
            # Crambin is 46 residues; ESM-IF1 should produce same-length sequences
            assert len(seq.sequence["A"]) >= 40
            assert seq.recovery is not None
            assert 0.0 <= seq.recovery <= 1.0

    def test_native_sequence_extracted(
        self, rcsb_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Verify native sequence is extracted from the input structure."""
        input_data = InverseFoldingInput(structure_path=rcsb_pdb)
        runner = get_runner("esm_if1", autobio_config)
        output = runner.run(input_data, gpu="auto", mode="design", output_dir=tmp_path / "ws")

        assert output.native_sequence is not None
        assert "A" in output.native_sequence
        # Crambin native sequence starts with T
        assert output.native_sequence["A"].startswith("T")

    def test_multiple_sequences(
        self, rcsb_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Request multiple designs and verify count."""
        input_data = InverseFoldingInput(
            structure_path=rcsb_pdb,
            num_sequences=5,
        )
        runner = get_runner("esm_if1", autobio_config)
        output = runner.run(input_data, gpu="auto", mode="design", output_dir=tmp_path / "ws")

        assert len(output.designed_sequences) == 5
        ranks = [s.rank for s in output.designed_sequences]
        assert ranks == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# ESM-IF1 Design — multi-chain Fab (7FAB)
# ---------------------------------------------------------------------------

_7FAB_CHAINS = ("H", "L")
_7FAB_MIN_RESIDUES = {"H": 190, "L": 190}


class TestESMIF1DesignMultiChain:
    """ESM-IF1 on a multi-chain Fab antibody fragment (7FAB)."""

    def test_full_pipeline_multi_chain(
        self, rcsb_fab_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Run ESM-IF1 on a 2-chain Fab and verify per-chain output."""
        input_data = InverseFoldingInput(
            structure_path=rcsb_fab_pdb,
            num_sequences=2,
            temperature=0.1,
        )
        runner = get_runner("esm_if1", autobio_config)
        output = runner.run(input_data, gpu="auto", mode="design", output_dir=tmp_path / "ws")

        assert isinstance(output, InverseFoldingOutput)
        assert len(output.designed_sequences) == 2
        assert output.metadata.tool_name == "esm_if1"

        for seq in output.designed_sequences:
            for chain_id in _7FAB_CHAINS:
                assert chain_id in seq.sequence, f"Missing chain {chain_id}"
                assert len(seq.sequence[chain_id]) >= _7FAB_MIN_RESIDUES[chain_id], (
                    f"Chain {chain_id}: expected >= {_7FAB_MIN_RESIDUES[chain_id]}, "
                    f"got {len(seq.sequence[chain_id])}"
                )
            assert seq.recovery is not None
            assert 0.0 <= seq.recovery <= 1.0

    def test_chains_to_design(
        self, rcsb_fab_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Design only the heavy chain H, keeping light chain L at native sequence."""
        input_data = InverseFoldingInput(
            structure_path=rcsb_fab_pdb,
            chains_to_design=["H"],
            num_sequences=1,
            temperature=0.1,
        )
        runner = get_runner("esm_if1", autobio_config)
        output = runner.run(input_data, gpu="auto", mode="design", output_dir=tmp_path / "ws")

        assert len(output.designed_sequences) == 1
        seq = output.designed_sequences[0]

        # Both chains should still be present in the output
        for chain_id in _7FAB_CHAINS:
            assert chain_id in seq.sequence, f"Missing chain {chain_id}"

        # Light chain should match the native sequence exactly
        assert output.native_sequence is not None
        assert seq.sequence["L"] == output.native_sequence["L"], (
            "Light chain L should be native (not redesigned)"
        )


# ---------------------------------------------------------------------------
# ESM-IF1 Design — fixed positions
# ---------------------------------------------------------------------------


class TestESMIF1FixedPositions:
    """Verify post-hoc enforcement of fixed positions."""

    def test_fixed_positions_enforced(
        self, rcsb_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Fixed positions in the designed sequence should match the native residue."""
        # Fix positions 1, 5, 10 in chain A
        fixed = {"A": [1, 5, 10]}
        input_data = InverseFoldingInput(
            structure_path=rcsb_pdb,
            num_sequences=1,
            temperature=1.0,  # high temperature to ensure variation
            fixed_positions=fixed,
        )
        runner = get_runner("esm_if1", autobio_config)
        output = runner.run(input_data, gpu="auto", mode="design", output_dir=tmp_path / "ws")

        assert output.native_sequence is not None
        native = output.native_sequence["A"]
        designed = output.designed_sequences[0].sequence["A"]

        for pos in fixed["A"]:
            idx = pos - 1  # 1-based to 0-based
            assert designed[idx] == native[idx], (
                f"Position {pos}: designed={designed[idx]!r}, native={native[idx]!r}"
            )


# ---------------------------------------------------------------------------
# ESM-IF1 Score
# ---------------------------------------------------------------------------


class TestESMIF1Score:
    """Full pipeline tests for ESM-IF1 sequence scoring."""

    def test_score_native_sequence(
        self, rcsb_pdb: Path, autobio_config: AutobioConfig, tmp_path: Path
    ) -> None:
        """Score the native sequence of crambin against its structure."""
        # First, get the native sequence via a design run
        design_input = InverseFoldingInput(structure_path=rcsb_pdb)
        design_runner = get_runner("esm_if1", autobio_config)
        design_output = design_runner.run(
            design_input, gpu="auto", mode="design", output_dir=tmp_path / "design_ws"
        )
        assert design_output.native_sequence is not None

        # Now score the native sequence
        score_input = ScoringInput(
            structure_path=rcsb_pdb,
            sequences=design_output.native_sequence,
        )
        score_runner = get_runner("esm_if1", autobio_config)
        output = score_runner.run(
            score_input, gpu="auto", mode="score", output_dir=tmp_path / "score_ws"
        )

        assert isinstance(output, ScoringOutput)
        assert len(output.scores) == 1
        score = output.scores[0]
        assert score.total_score < 0  # NLL should be negative
        assert score.units == "avg_nll"
        assert score.score_breakdown is not None
        assert output.metadata.tool_name == "esm_if1"
        assert output.metadata.mode == "score"
