"""Tests for AntibodyLMRunner — workspace, output parsing, validation, registration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from autobio.core.config import AutobioConfig
from autobio.core.registry import TOOL_REGISTRY, ToolCategory
from autobio.core.result import AutobioError
from autobio.core.workspace import Workspace
from autobio.schemas.antibody import AntibodyInput, AntibodyPLLOutput, AntibodySequence
from autobio.schemas.embedding import EmbeddingOutput
from autobio.tools import TOOL_RUNNERS, get_runner
from autobio.tools.antibody_lm import AntibodyLMRunner

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


def _make_runner(tool_name: str, config: AutobioConfig) -> AntibodyLMRunner:
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        return AntibodyLMRunner(tool_name, config)


@pytest.fixture()
def currab_runner(config: AutobioConfig) -> AntibodyLMRunner:
    return _make_runner("currab", config)


@pytest.fixture()
def currab_pll_runner(config: AutobioConfig) -> AntibodyLMRunner:
    return _make_runner("currab_pll", config)


@pytest.fixture()
def balm_paired_runner(config: AutobioConfig) -> AntibodyLMRunner:
    return _make_runner("balm_paired", config)


@pytest.fixture()
def balm_unpaired_runner(config: AutobioConfig) -> AntibodyLMRunner:
    return _make_runner("balm_unpaired", config)


@pytest.fixture()
def ablang2_runner(config: AutobioConfig) -> AntibodyLMRunner:
    return _make_runner("ablang2", config)


@pytest.fixture()
def ablang2_pll_runner(config: AutobioConfig) -> AntibodyLMRunner:
    return _make_runner("ablang2_pll", config)


@pytest.fixture()
def antiberta2_runner(config: AutobioConfig) -> AntibodyLMRunner:
    return _make_runner("antiberta2", config)


@pytest.fixture()
def antiberta2_pll_runner(config: AutobioConfig) -> AntibodyLMRunner:
    return _make_runner("antiberta2_pll", config)


# Sample sequences — Trastuzumab VH/VL fragments
_VH = "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIH"
_VL = "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVA"


# ---------------------------------------------------------------------------
# TestAntibodyLMPrepareWorkspace
# ---------------------------------------------------------------------------


class TestAntibodyLMPrepareWorkspace:
    """Tests for AntibodyLMRunner.prepare_workspace."""

    def test_paired_json_input(self, currab_runner: AntibodyLMRunner, tmp_path: Path) -> None:
        """Paired sequences are written as JSON with both chains."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(
            sequences=[AntibodySequence(id="ab1", heavy_chain=_VH, light_chain=_VL)]
        )
        currab_runner.prepare_workspace(input_data, workspace)

        json_path = workspace.inputs_dir / "sequences.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert len(data) == 1
        assert data[0]["heavy_chain"] == _VH
        assert data[0]["light_chain"] == _VL

    def test_heavy_only_json(self, currab_runner: AntibodyLMRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)])
        currab_runner.prepare_workspace(input_data, workspace)

        data = json.loads((workspace.inputs_dir / "sequences.json").read_text())
        assert data[0]["heavy_chain"] == _VH
        assert data[0]["light_chain"] is None

    def test_light_only_json(self, currab_runner: AntibodyLMRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(sequences=[AntibodySequence(id="ab1", light_chain=_VL)])
        currab_runner.prepare_workspace(input_data, workspace)

        data = json.loads((workspace.inputs_dir / "sequences.json").read_text())
        assert data[0]["heavy_chain"] is None
        assert data[0]["light_chain"] == _VL

    @pytest.mark.parametrize(
        ("tool_name", "expected_model"),
        [
            ("currab", "brineylab/CurrAb"),
            ("ft_esm", "brineylab/ft-ESM"),
            ("balm_paired", "brineylab/BALM-paired"),
            ("balm_unpaired", "brineylab/BALM-unpaired"),
            ("ablang2", "ablang2-paired"),
            ("antiberta2", "alchemab/antiberta2"),
        ],
    )
    def test_model_name_per_tool(
        self, tool_name: str, expected_model: str, config: AutobioConfig, tmp_path: Path
    ) -> None:
        runner = _make_runner(tool_name, config)
        workspace = Workspace.create(tmp_path / "ws")
        # balm_paired needs both chains; balm_unpaired needs one
        if tool_name == "balm_paired":
            seqs = [AntibodySequence(id="ab1", heavy_chain=_VH, light_chain=_VL)]
        elif tool_name == "balm_unpaired":
            seqs = [AntibodySequence(id="ab1", heavy_chain=_VH)]
        else:
            seqs = [AntibodySequence(id="ab1", heavy_chain=_VH, light_chain=_VL)]
        runner.prepare_workspace(AntibodyInput(sequences=seqs), workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["model_name"] == expected_model

    def test_mode_embedding(self, currab_runner: AntibodyLMRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)])
        currab_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["mode"] == "embedding"

    def test_mode_pll(self, currab_pll_runner: AntibodyLMRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)])
        currab_pll_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["mode"] == "pll"

    @pytest.mark.parametrize(
        ("tool_name", "expected_family", "expected_sep"),
        [
            ("currab", "esm", "single_cls"),
            ("ft_esm", "esm", "double_cls"),
            ("balm_paired", "roberta", "sep"),
            ("balm_unpaired", "roberta", "none"),
            ("ablang2", "ablang2", "pipe"),
            ("antiberta2", "roformer", "sep_prefixed"),
        ],
    )
    def test_model_family_and_separator_in_config(
        self,
        tool_name: str,
        expected_family: str,
        expected_sep: str,
        config: AutobioConfig,
        tmp_path: Path,
    ) -> None:
        runner = _make_runner(tool_name, config)
        workspace = Workspace.create(tmp_path / "ws")
        if tool_name == "balm_paired":
            seqs = [AntibodySequence(id="ab1", heavy_chain=_VH, light_chain=_VL)]
        elif tool_name == "balm_unpaired":
            seqs = [AntibodySequence(id="ab1", heavy_chain=_VH)]
        else:
            seqs = [AntibodySequence(id="ab1", heavy_chain=_VH)]
        runner.prepare_workspace(AntibodyInput(sequences=seqs), workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["model_family"] == expected_family
        assert cfg["chain_separator"] == expected_sep

    def test_layer_in_config(self, currab_runner: AntibodyLMRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(
            sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)], layer=20
        )
        currab_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["layer"] == 20

    def test_pooling_default_mean(self, currab_runner: AntibodyLMRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)])
        currab_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["pooling"] == "mean"

    def test_per_position_default_false(
        self, currab_pll_runner: AntibodyLMRunner, tmp_path: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)])
        currab_pll_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["per_position"] is False

    def test_per_position_opt_in(self, currab_pll_runner: AntibodyLMRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(
            sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)],
            extra={"per_position": True},
        )
        currab_pll_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["per_position"] is True

    def test_extra_dict_merged(self, currab_runner: AntibodyLMRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(
            sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)],
            extra={"batch_size": 16, "seed": 42},
        )
        currab_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["batch_size"] == 16
        assert cfg["seed"] == 42

    def test_consumed_keys_excluded(
        self, currab_pll_runner: AntibodyLMRunner, tmp_path: Path
    ) -> None:
        """Consumed extra key 'per_position' does not appear twice in config."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(
            sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)],
            extra={"per_position": True, "seed": 42},
        )
        currab_pll_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        # per_position appears as a top-level field, not duplicated from extra merge
        assert cfg["per_position"] is True
        assert cfg["seed"] == 42


# ---------------------------------------------------------------------------
# TestAntibodyLMHostValidation
# ---------------------------------------------------------------------------


class TestAntibodyLMHostValidation:
    """Tests for host-side input validation."""

    def test_empty_sequences_raises(self, currab_runner: AntibodyLMRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(sequences=[])
        with pytest.raises(AutobioError, match="sequences must be non-empty"):
            currab_runner.prepare_workspace(input_data, workspace)

    def test_no_chains_raises(self) -> None:
        """Pydantic validator catches both chains being None."""
        with pytest.raises(Exception, match="at least one"):
            AntibodySequence(id="ab1", heavy_chain=None, light_chain=None)

    def test_invalid_heavy_chain_raises(
        self, currab_runner: AntibodyLMRunner, tmp_path: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(sequences=[AntibodySequence(id="ab1", heavy_chain="MVLS123BAD")])
        with pytest.raises(AutobioError, match="heavy_chain.*invalid characters"):
            currab_runner.prepare_workspace(input_data, workspace)

    def test_invalid_light_chain_raises(
        self, currab_runner: AntibodyLMRunner, tmp_path: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(sequences=[AntibodySequence(id="ab1", light_chain="DI@MT!")])
        with pytest.raises(AutobioError, match="light_chain.*invalid characters"):
            currab_runner.prepare_workspace(input_data, workspace)

    def test_sequence_too_long_raises(
        self, currab_runner: AntibodyLMRunner, tmp_path: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        # CurrAb max is 320 tokens
        input_data = AntibodyInput(sequences=[AntibodySequence(id="ab1", heavy_chain="A" * 321)])
        with pytest.raises(AutobioError, match="exceeds maximum"):
            currab_runner.prepare_workspace(input_data, workspace)

    def test_paired_to_balm_unpaired_raises(
        self, balm_unpaired_runner: AntibodyLMRunner, tmp_path: Path
    ) -> None:
        """BALM-unpaired rejects paired (both chains) input."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(
            sequences=[AntibodySequence(id="ab1", heavy_chain=_VH, light_chain=_VL)]
        )
        with pytest.raises(AutobioError, match="does not support paired"):
            balm_unpaired_runner.prepare_workspace(input_data, workspace)

    def test_unpaired_to_balm_paired_raises(
        self, balm_paired_runner: AntibodyLMRunner, tmp_path: Path
    ) -> None:
        """BALM-paired rejects single-chain input."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)])
        with pytest.raises(AutobioError, match="requires both"):
            balm_paired_runner.prepare_workspace(input_data, workspace)

    def test_invalid_layer_raises(self, currab_runner: AntibodyLMRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(
            sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)], layer=50
        )
        with pytest.raises(AutobioError, match="layer must be between 0 and 33"):
            currab_runner.prepare_workspace(input_data, workspace)

    def test_layer_validation_model_aware(self, config: AutobioConfig, tmp_path: Path) -> None:
        """BALM has 24 layers, so layer=30 should fail."""
        runner = _make_runner("balm_unpaired", config)
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(
            sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)], layer=30
        )
        with pytest.raises(AutobioError, match="layer must be between 0 and 24"):
            runner.prepare_workspace(input_data, workspace)

    def test_invalid_pooling_raises(self, currab_runner: AntibodyLMRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(
            sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)], pooling="max"
        )
        with pytest.raises(AutobioError, match="pooling must be one of"):
            currab_runner.prepare_workspace(input_data, workspace)

    def test_valid_layer_zero(self, currab_runner: AntibodyLMRunner, tmp_path: Path) -> None:
        """Layer 0 (input embedding) is valid."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)], layer=0)
        currab_runner.prepare_workspace(input_data, workspace)
        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["layer"] == 0

    def test_ambiguous_residues_accepted(
        self, currab_runner: AntibodyLMRunner, tmp_path: Path
    ) -> None:
        """Ambiguous residue codes (B, X, Z, etc.) are accepted."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(sequences=[AntibodySequence(id="ab1", heavy_chain="EVQLXBZ")])
        currab_runner.prepare_workspace(input_data, workspace)

    def test_invalid_layer_ablang2(self, ablang2_runner: AntibodyLMRunner, tmp_path: Path) -> None:
        """AbLang2 has 12 layers — layer=20 should fail."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(
            sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)], layer=20
        )
        with pytest.raises(AutobioError, match="layer must be between 0 and 12"):
            ablang2_runner.prepare_workspace(input_data, workspace)

    def test_invalid_layer_antiberta2(
        self, antiberta2_runner: AntibodyLMRunner, tmp_path: Path
    ) -> None:
        """AntiBERTa2 has 16 layers — layer=20 should fail."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(
            sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)], layer=20
        )
        with pytest.raises(AutobioError, match="layer must be between 0 and 16"):
            antiberta2_runner.prepare_workspace(input_data, workspace)

    def test_sequence_too_long_antiberta2(
        self, antiberta2_runner: AntibodyLMRunner, tmp_path: Path
    ) -> None:
        """AntiBERTa2 max is 250 tokens — 260 residues should fail."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(sequences=[AntibodySequence(id="ab1", heavy_chain="A" * 260)])
        with pytest.raises(AutobioError, match="exceeds maximum"):
            antiberta2_runner.prepare_workspace(input_data, workspace)

    def test_cache_path_ablang2(self, ablang2_runner: AntibodyLMRunner, tmp_path: Path) -> None:
        """AbLang2 config.json should contain its custom cache path."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)])
        ablang2_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["hf_cache"] == "/app/ablang2/weights"

    def test_cache_path_antiberta2(
        self, antiberta2_runner: AntibodyLMRunner, tmp_path: Path
    ) -> None:
        """AntiBERTa2 config.json should contain its custom cache path."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)])
        antiberta2_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["hf_cache"] == "/app/antiberta2/hf_cache"

    def test_cache_path_existing_models_unchanged(
        self, currab_runner: AntibodyLMRunner, tmp_path: Path
    ) -> None:
        """Existing models should still use the original HF cache path."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)])
        currab_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        assert cfg["hf_cache"] == "/app/antibody-lm/hf_cache"


# ---------------------------------------------------------------------------
# TestAntibodyLMParseOutput
# ---------------------------------------------------------------------------

_SINGLE_EMBEDDING_RESULT = {
    "embeddings": [
        {
            "sequence_id": "ab1",
            "embedding_path": "/workspace/outputs/standardized/ab1.npy",
            "dimension": 1280,
            "layer": 33,
            "pooling": "mean",
        }
    ],
    "model_name": "CurrAb",
    "embedding_dimension": 1280,
}

_MULTI_EMBEDDING_RESULT = {
    "embeddings": [
        {
            "sequence_id": "ab1",
            "embedding_path": "/workspace/outputs/standardized/ab1.npy",
            "dimension": 1280,
            "layer": 33,
            "pooling": "mean",
        },
        {
            "sequence_id": "ab2",
            "embedding_path": "/workspace/outputs/standardized/ab2.npy",
            "dimension": 1280,
            "layer": 33,
            "pooling": "mean",
        },
    ],
    "model_name": "CurrAb",
    "embedding_dimension": 1280,
}

_PLL_RESULT = {
    "scores": [
        {
            "sequence_id": "ab1",
            "pll": -45.23,
            "sequence_length": 34,
        }
    ],
    "model_name": "CurrAb",
}

_PLL_RESULT_WITH_PER_POSITION = {
    "scores": [
        {
            "sequence_id": "ab1",
            "pll": -45.23,
            "per_position_pll": [-1.2, -0.8, -1.5],
            "sequence_length": 3,
        }
    ],
    "model_name": "CurrAb",
}


class TestAntibodyLMParseOutput:
    """Tests for AntibodyLMRunner.parse_output."""

    def test_parse_single_embedding(self, currab_runner: AntibodyLMRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(
            json.dumps(_SINGLE_EMBEDDING_RESULT)
        )

        output = currab_runner.parse_output(workspace)
        assert isinstance(output, EmbeddingOutput)
        assert len(output.embeddings) == 1
        assert output.embeddings[0].sequence_id == "ab1"
        assert output.embeddings[0].dimension == 1280

    def test_parse_multiple_embeddings(
        self, currab_runner: AntibodyLMRunner, tmp_path: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(
            json.dumps(_MULTI_EMBEDDING_RESULT)
        )

        output = currab_runner.parse_output(workspace)
        assert len(output.embeddings) == 2
        ids = {e.sequence_id for e in output.embeddings}
        assert ids == {"ab1", "ab2"}

    def test_model_name_populated(self, currab_runner: AntibodyLMRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(
            json.dumps(_SINGLE_EMBEDDING_RESULT)
        )

        output = currab_runner.parse_output(workspace)
        assert output.model_name == "CurrAb"

    def test_container_paths_resolved(
        self, currab_runner: AntibodyLMRunner, tmp_path: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(
            json.dumps(_SINGLE_EMBEDDING_RESULT)
        )

        output = currab_runner.parse_output(workspace)
        expected = workspace.root / "outputs" / "standardized" / "ab1.npy"
        assert output.embeddings[0].embedding_path == expected

    def test_parse_pll_output(self, currab_pll_runner: AntibodyLMRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(json.dumps(_PLL_RESULT))

        output = currab_pll_runner.parse_output(workspace)
        assert isinstance(output, AntibodyPLLOutput)
        assert len(output.scores) == 1
        assert output.scores[0].pll == pytest.approx(-45.23)
        assert output.scores[0].sequence_length == 34
        assert output.scores[0].per_position_pll is None

    def test_parse_pll_per_position(
        self, currab_pll_runner: AntibodyLMRunner, tmp_path: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.std_output_dir / "result_data.json").write_text(
            json.dumps(_PLL_RESULT_WITH_PER_POSITION)
        )

        output = currab_pll_runner.parse_output(workspace)
        assert output.scores[0].per_position_pll == [-1.2, -0.8, -1.5]


# ---------------------------------------------------------------------------
# TestAntibodyLMRegistration
# ---------------------------------------------------------------------------

_ALL_TOOLS = [
    "currab",
    "currab_pll",
    "ft_esm",
    "ft_esm_pll",
    "balm_paired",
    "balm_paired_pll",
    "balm_unpaired",
    "balm_unpaired_pll",
    "ablang2",
    "ablang2_pll",
    "antiberta2",
    "antiberta2_pll",
]

_EMBEDDING_TOOLS = ["currab", "ft_esm", "balm_paired", "balm_unpaired", "ablang2", "antiberta2"]
_PLL_TOOLS = [
    "currab_pll",
    "ft_esm_pll",
    "balm_paired_pll",
    "balm_unpaired_pll",
    "ablang2_pll",
    "antiberta2_pll",
]


class TestAntibodyLMRegistration:
    """Tests for tool and runner registration."""

    @pytest.mark.parametrize("tool_name", _ALL_TOOLS)
    def test_in_registry(self, tool_name: str) -> None:
        assert tool_name in TOOL_REGISTRY
        assert TOOL_REGISTRY[tool_name].category == ToolCategory.EMBEDDING

    @pytest.mark.parametrize("tool_name", _EMBEDDING_TOOLS)
    def test_embedding_schema(self, tool_name: str) -> None:
        entry = TOOL_REGISTRY[tool_name]
        assert entry.input_schema is AntibodyInput
        assert entry.output_schema is EmbeddingOutput

    @pytest.mark.parametrize("tool_name", _PLL_TOOLS)
    def test_pll_schema(self, tool_name: str) -> None:
        entry = TOOL_REGISTRY[tool_name]
        assert entry.input_schema is AntibodyInput
        assert entry.output_schema is AntibodyPLLOutput

    def test_currab_tools_share_image(self) -> None:
        assert TOOL_REGISTRY["currab"].image_tag == TOOL_REGISTRY["currab_pll"].image_tag

    def test_distinct_image_tags(self) -> None:
        """Each model has its own container image tag."""
        tags = {TOOL_REGISTRY[t].image_tag for t in _EMBEDDING_TOOLS}
        assert len(tags) == 6

    @pytest.mark.parametrize("tool_name", _EMBEDDING_TOOLS)
    def test_embedding_timeout(self, tool_name: str) -> None:
        assert TOOL_REGISTRY[tool_name].default_timeout == 600

    @pytest.mark.parametrize("tool_name", _PLL_TOOLS)
    def test_pll_timeout(self, tool_name: str) -> None:
        assert TOOL_REGISTRY[tool_name].default_timeout == 1800

    @pytest.mark.parametrize("tool_name", _ALL_TOOLS)
    def test_tool_runners_registered(self, tool_name: str) -> None:
        assert tool_name in TOOL_RUNNERS
        assert TOOL_RUNNERS[tool_name] is AntibodyLMRunner

    def test_get_runner_returns_antibody_lm_runner(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner("currab", config)
        assert isinstance(r, AntibodyLMRunner)
        assert r.tool_name == "currab"

    @pytest.mark.parametrize("tool_name", _ALL_TOOLS)
    def test_notes_populated(self, tool_name: str) -> None:
        entry = TOOL_REGISTRY[tool_name]
        assert len(entry.notes) > 0

    @pytest.mark.parametrize("tool_name", _ALL_TOOLS)
    def test_requires_gpu(self, tool_name: str) -> None:
        assert TOOL_REGISTRY[tool_name].requires_gpu is True
