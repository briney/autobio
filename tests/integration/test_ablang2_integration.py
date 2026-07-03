"""Integration tests for AbLang2 antibody language model runner.

Tests AbLang2 embedding extraction and PLL scoring end-to-end.
Requires Docker and a GPU.

Run with:
    pytest tests/integration/test_ablang2_integration.py -v -m docker
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
# AbLang2 Embedding (12 layers, 480-dim)
# ===========================================================================


class TestAbLang2PairedEmbedding:
    """AbLang2 embedding extraction on paired antibody sequences."""

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
        runner = get_runner("ablang2", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, EmbeddingOutput)
        assert len(output.embeddings) == 1
        assert output.metadata.tool_name == "ablang2"
        assert output.metadata.wall_time_seconds > 0
        assert output.embedding_dimension == 480

        e = output.embeddings[0]
        assert e.sequence_id == "trastuzumab"
        assert e.dimension == 480
        assert e.embedding_path.exists()

        emb = np.load(e.embedding_path)
        assert emb.shape == (480,)


class TestAbLang2HeavyOnly:
    def test_heavy_chain_only(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        input_data = AntibodyEmbeddingInput(
            sequences=[AntibodySequence(id="vh_only", heavy_chain=_TRASTUZUMAB_VH)],
        )
        runner = get_runner("ablang2", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, EmbeddingOutput)
        assert len(output.embeddings) == 1
        assert output.embeddings[0].dimension == 480


class TestAbLang2LightOnly:
    def test_light_chain_only(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        input_data = AntibodyEmbeddingInput(
            sequences=[AntibodySequence(id="vl_only", light_chain=_TRASTUZUMAB_VL)],
        )
        runner = get_runner("ablang2", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert isinstance(output, EmbeddingOutput)
        assert len(output.embeddings) == 1


class TestAbLang2Batch:
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
        runner = get_runner("ablang2", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        assert len(output.embeddings) == 2
        ids = {e.sequence_id for e in output.embeddings}
        assert ids == {"paired", "heavy"}


class TestAbLang2PerResidue:
    def test_per_residue_shape(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        input_data = AntibodyEmbeddingInput(
            sequences=[AntibodySequence(id="vh", heavy_chain=_TRASTUZUMAB_VH)],
            pooling="per_residue",
        )
        runner = get_runner("ablang2", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws")

        e = output.embeddings[0]
        emb = np.load(e.embedding_path)
        assert emb.ndim == 2
        assert emb.shape[1] == 480
        assert emb.shape[0] == len(_TRASTUZUMAB_VH)


# ===========================================================================
# AbLang2 PLL
# ===========================================================================


class TestAbLang2PLL:
    def test_pll_pipeline(self, autobio_config: AutobioConfig, tmp_path: Path) -> None:
        input_data = AntibodyPLLInput(
            sequences=[AntibodySequence(id="vh", heavy_chain=_TRASTUZUMAB_VH)],
        )
        runner = get_runner("ablang2", autobio_config)
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
        runner = get_runner("ablang2", autobio_config)
        output = runner.run(input_data, gpu="auto", output_dir=tmp_path / "ws", mode="pll")

        score = output.scores[0]
        assert score.per_position_pll is not None
        assert len(score.per_position_pll) == score.sequence_length
        assert all(lp < 0 for lp in score.per_position_pll)
        assert abs(sum(score.per_position_pll) - score.pll) < 1e-4
