"""Integration tests for antibody language model runners.

Tests CurrAb, ft-ESM, BALM-paired, and BALM-unpaired end-to-end.
Requires Docker and a GPU.

Run with:
    pytest tests/integration/test_currab_integration.py -v -m docker
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

np = pytest.importorskip("numpy")

from autobio.core.config import AutobioConfig  # noqa: E402
from autobio.schemas.antibody import (  # noqa: E402
    AntibodyEmbeddingInput,
    AntibodyPLLInput,
    AntibodyPLLOutput,
    AntibodySequence,
)
from autobio.schemas.embedding import EmbeddingOutput  # noqa: E402
from autobio.tools import get_runner  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

# All tests in this module require Docker, GPU, and are slow
pytestmark = [pytest.mark.docker, pytest.mark.gpu, pytest.mark.slow]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Trastuzumab variable regions (anti-HER2)
_TRASTUZUMAB_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNG"
    "YTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQ"
    "GTLVTVSS"
)
_TRASTUZUMAB_VL = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLES"
    "GVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)


@pytest.fixture(scope="session")
def autobio_config() -> AutobioConfig:
    return AutobioConfig.resolve()


# ===========================================================================
# CurrAb (ESM-based, single <cls> separator, 1280-dim)
# ===========================================================================


class TestCurrAbPairedEmbedding:
    """CurrAb embedding extraction on paired antibody sequences."""

    def test_full_pipeline(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        input_data = AntibodyEmbeddingInput(
            sequences=[
                AntibodySequence(
                    id="trastuzumab",
                    heavy_chain=_TRASTUZUMAB_VH,
                    light_chain=_TRASTUZUMAB_VL,
                )
            ],
        )
        runner = get_runner("currab", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, EmbeddingOutput)
        assert len(output.embeddings) == 1
        assert output.metadata.tool_name == "currab"
        assert output.metadata.wall_time_seconds > 0
        assert output.embedding_dimension == 1280

        e = output.embeddings[0]
        assert e.sequence_id == "trastuzumab"
        assert e.dimension == 1280
        assert e.embedding_path.exists()

        emb = np.load(e.embedding_path)
        assert emb.shape == (1280,)


class TestCurrAbHeavyOnly:
    def test_heavy_chain_only(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        input_data = AntibodyEmbeddingInput(
            sequences=[AntibodySequence(id="vh_only", heavy_chain=_TRASTUZUMAB_VH)],
        )
        runner = get_runner("currab", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, EmbeddingOutput)
        assert len(output.embeddings) == 1
        assert output.embeddings[0].dimension == 1280


class TestCurrAbLightOnly:
    def test_light_chain_only(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        input_data = AntibodyEmbeddingInput(
            sequences=[AntibodySequence(id="vl_only", light_chain=_TRASTUZUMAB_VL)],
        )
        runner = get_runner("currab", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, EmbeddingOutput)
        assert len(output.embeddings) == 1


class TestCurrAbBatch:
    def test_batch_embedding(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        input_data = AntibodyEmbeddingInput(
            sequences=[
                AntibodySequence(
                    id="paired",
                    heavy_chain=_TRASTUZUMAB_VH,
                    light_chain=_TRASTUZUMAB_VL,
                ),
                AntibodySequence(id="heavy", heavy_chain=_TRASTUZUMAB_VH),
            ],
        )
        runner = get_runner("currab", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert len(output.embeddings) == 2
        ids = {e.sequence_id for e in output.embeddings}
        assert ids == {"paired", "heavy"}


class TestCurrAbPerResidue:
    def test_per_residue_shape(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        input_data = AntibodyEmbeddingInput(
            sequences=[AntibodySequence(id="vh", heavy_chain=_TRASTUZUMAB_VH)],
            pooling="per_residue",
        )
        runner = get_runner("currab", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        e = output.embeddings[0]
        emb = np.load(e.embedding_path)
        assert emb.ndim == 2
        assert emb.shape[1] == 1280
        assert emb.shape[0] == len(_TRASTUZUMAB_VH)


class TestCurrAbPLL:
    def test_pll_pipeline(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        input_data = AntibodyPLLInput(
            sequences=[AntibodySequence(id="vh", heavy_chain=_TRASTUZUMAB_VH)],
        )
        runner = get_runner("currab", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws", mode="pll")

        assert isinstance(output, AntibodyPLLOutput)
        assert len(output.scores) == 1
        score = output.scores[0]
        assert score.sequence_id == "vh"
        assert score.pll < 0
        assert score.sequence_length > 0
        assert score.per_position_pll is None

    def test_pll_per_position(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        input_data = AntibodyPLLInput(
            sequences=[AntibodySequence(id="vh", heavy_chain=_TRASTUZUMAB_VH)],
            per_position=True,
        )
        runner = get_runner("currab", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws", mode="pll")

        score = output.scores[0]
        assert score.per_position_pll is not None
        assert len(score.per_position_pll) == score.sequence_length
        assert all(lp < 0 for lp in score.per_position_pll)
        assert abs(sum(score.per_position_pll) - score.pll) < 1e-4


# ===========================================================================
# ft-ESM (ESM-based, double <cls> separator, 1280-dim)
# ===========================================================================


class TestFtEsmPairedEmbedding:
    """ft-ESM embedding on paired sequences (double <cls> separator)."""

    def test_full_pipeline(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        input_data = AntibodyEmbeddingInput(
            sequences=[
                AntibodySequence(
                    id="trastuzumab",
                    heavy_chain=_TRASTUZUMAB_VH,
                    light_chain=_TRASTUZUMAB_VL,
                )
            ],
        )
        runner = get_runner("ft_esm", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, EmbeddingOutput)
        assert output.embedding_dimension == 1280
        assert output.metadata.tool_name == "ft_esm"

        e = output.embeddings[0]
        assert e.dimension == 1280
        emb = np.load(e.embedding_path)
        assert emb.shape == (1280,)


class TestFtEsmUnpaired:
    def test_heavy_only(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        input_data = AntibodyEmbeddingInput(
            sequences=[AntibodySequence(id="vh", heavy_chain=_TRASTUZUMAB_VH)],
        )
        runner = get_runner("ft_esm", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert len(output.embeddings) == 1
        assert output.embeddings[0].dimension == 1280


class TestFtEsmPLL:
    def test_pll_pipeline(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        input_data = AntibodyPLLInput(
            sequences=[AntibodySequence(id="vh", heavy_chain=_TRASTUZUMAB_VH)],
        )
        runner = get_runner("ft_esm", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws", mode="pll")

        assert isinstance(output, AntibodyPLLOutput)
        score = output.scores[0]
        assert score.pll < 0
        assert score.sequence_length > 0


# ===========================================================================
# BALM-paired (RoBERTa-based, </s> separator, 1024-dim)
# ===========================================================================


class TestBalmPairedEmbedding:
    """BALM-paired embedding (requires both chains)."""

    def test_full_pipeline(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        input_data = AntibodyEmbeddingInput(
            sequences=[
                AntibodySequence(
                    id="trastuzumab",
                    heavy_chain=_TRASTUZUMAB_VH,
                    light_chain=_TRASTUZUMAB_VL,
                )
            ],
        )
        runner = get_runner("balm_paired", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, EmbeddingOutput)
        assert output.embedding_dimension == 1024
        assert output.metadata.tool_name == "balm_paired"

        e = output.embeddings[0]
        assert e.dimension == 1024
        emb = np.load(e.embedding_path)
        assert emb.shape == (1024,)


class TestBalmPairedPLL:
    def test_pll_pipeline(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        input_data = AntibodyPLLInput(
            sequences=[
                AntibodySequence(
                    id="trastuzumab",
                    heavy_chain=_TRASTUZUMAB_VH,
                    light_chain=_TRASTUZUMAB_VL,
                )
            ],
        )
        runner = get_runner("balm_paired", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws", mode="pll")

        assert isinstance(output, AntibodyPLLOutput)
        score = output.scores[0]
        assert score.pll < 0
        assert score.sequence_length > 0


# ===========================================================================
# BALM-unpaired (RoBERTa-based, single chain, 1024-dim)
# ===========================================================================


class TestBalmUnpairedEmbedding:
    """BALM-unpaired embedding (single chain only)."""

    def test_heavy_chain(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        input_data = AntibodyEmbeddingInput(
            sequences=[AntibodySequence(id="vh", heavy_chain=_TRASTUZUMAB_VH)],
        )
        runner = get_runner("balm_unpaired", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, EmbeddingOutput)
        assert output.embedding_dimension == 1024
        assert output.metadata.tool_name == "balm_unpaired"

        e = output.embeddings[0]
        assert e.dimension == 1024
        emb = np.load(e.embedding_path)
        assert emb.shape == (1024,)

    def test_light_chain(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        input_data = AntibodyEmbeddingInput(
            sequences=[AntibodySequence(id="vl", light_chain=_TRASTUZUMAB_VL)],
        )
        runner = get_runner("balm_unpaired", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert len(output.embeddings) == 1
        assert output.embeddings[0].dimension == 1024


class TestBalmUnpairedPLL:
    def test_pll_pipeline(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        input_data = AntibodyPLLInput(
            sequences=[AntibodySequence(id="vh", heavy_chain=_TRASTUZUMAB_VH)],
        )
        runner = get_runner("balm_unpaired", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws", mode="pll")

        assert isinstance(output, AntibodyPLLOutput)
        score = output.scores[0]
        assert score.pll < 0
        assert score.sequence_length > 0
