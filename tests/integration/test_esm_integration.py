"""Integration tests for ESM-1b and ESM-2 embedding runners.

These tests require Docker and a GPU. They run the autobio-esm container
and verify end-to-end embedding extraction output.

Run with:
    pytest tests/integration/test_esm_integration.py -v -m docker
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

np = pytest.importorskip("numpy")

from autobio.core.config import AutobioConfig  # noqa: E402
from autobio.schemas.embedding import (  # noqa: E402
    EmbeddingOutput,
    ESM2Input,
    ESMEmbedInput,
)
from autobio.tools import get_runner  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

# All tests in this module require Docker, GPU, and are slow
pytestmark = [pytest.mark.docker, pytest.mark.gpu, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Crambin (PDB: 1CRN) — 46 residues, single chain, well-folded small protein
_CRAMBIN_SEQ = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"

# Insulin B chain — 30 residues
_INSULIN_B = "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"


@pytest.fixture(scope="session")
def autobio_config() -> AutobioConfig:
    return AutobioConfig.resolve()


# ---------------------------------------------------------------------------
# ESM-2 — single sequence, mean pooling (default)
# ---------------------------------------------------------------------------


class TestESM2SingleSequence:
    """ESM-2 embedding extraction on a single small protein."""

    def test_full_pipeline(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        """Run ESM-2 end-to-end and verify output embedding."""
        input_data = ESM2Input(
            sequences={"crambin": _CRAMBIN_SEQ},
        )
        runner = get_runner("esm2", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws", mode="embedding")

        assert isinstance(output, EmbeddingOutput)
        assert len(output.embeddings) == 1
        assert output.metadata.tool_name == "esm2"
        assert output.metadata.wall_time_seconds > 0
        assert output.model_name == "esm2_t33_650M_UR50D"
        assert output.embedding_dimension == 1280

        e = output.embeddings[0]
        assert e.sequence_id == "crambin"
        assert e.dimension == 1280
        assert e.embedding_path.exists()

        # Verify the .npy file has the correct shape for mean pooling
        emb = np.load(e.embedding_path)
        assert emb.shape == (1280,)

    def test_per_residue_pooling(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        """Per-residue pooling produces (L, D) shaped embedding."""
        input_data = ESM2Input(
            sequences={"crambin": _CRAMBIN_SEQ},
            pooling="per_residue",
        )
        runner = get_runner("esm2", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws", mode="embedding")

        e = output.embeddings[0]
        emb = np.load(e.embedding_path)
        assert emb.shape == (len(_CRAMBIN_SEQ), 1280)


# ---------------------------------------------------------------------------
# ESM-2 — multiple sequences
# ---------------------------------------------------------------------------


class TestESM2MultipleSequences:
    """ESM-2 embedding extraction on multiple sequences."""

    def test_batch_extraction(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        """Multiple sequences produce one embedding each."""
        input_data = ESM2Input(
            sequences={"crambin": _CRAMBIN_SEQ, "insulin_b": _INSULIN_B},
        )
        runner = get_runner("esm2", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws", mode="embedding")

        assert len(output.embeddings) == 2
        ids = {e.sequence_id for e in output.embeddings}
        assert ids == {"crambin", "insulin_b"}

        for e in output.embeddings:
            assert e.embedding_path.exists()
            emb = np.load(e.embedding_path)
            assert emb.shape == (1280,)


# ---------------------------------------------------------------------------
# ESM-2 — checkpoint selection
# ---------------------------------------------------------------------------


class TestESM2CheckpointSelection:
    """ESM-2 with a non-default checkpoint."""

    def test_8m_checkpoint(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        """8M checkpoint produces 320-dim embeddings."""
        input_data = ESM2Input(
            sequences={"crambin": _CRAMBIN_SEQ},
            checkpoint="8M",
        )
        runner = get_runner("esm2", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws", mode="embedding")

        assert output.embedding_dimension == 320
        e = output.embeddings[0]
        emb = np.load(e.embedding_path)
        assert emb.shape == (320,)


# ---------------------------------------------------------------------------
# ESM-1b
# ---------------------------------------------------------------------------


class TestESM1b:
    """ESM-1b embedding extraction."""

    def test_full_pipeline(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        """Run ESM-1b end-to-end."""
        input_data = ESMEmbedInput(
            sequences={"crambin": _CRAMBIN_SEQ},
        )
        runner = get_runner("esm1b", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws", mode="embedding")

        assert isinstance(output, EmbeddingOutput)
        assert output.model_name == "esm1b_t33_650M_UR50S"
        assert output.embedding_dimension == 1280
        assert output.embeddings[0].embedding_path.exists()
