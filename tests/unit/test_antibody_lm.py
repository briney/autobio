"""Tests for the migrated antibody-LM Tools (modes: embedding, pll)."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from autobio.core.catalog import get_tool, list_tools, tool_categories
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

_MODEL_NAMES = ("currab", "ft_esm", "balm_paired", "balm_unpaired", "ablang2", "antiberta2")

_IMAGE_TAGS = {
    "currab": "currab:1.0.0",
    "ft_esm": "ft-esm:1.0.0",
    "balm_paired": "balm-paired:1.0.0",
    "balm_unpaired": "balm-unpaired:1.0.0",
    "ablang2": "ablang2:1.0.0",
    "antiberta2": "antiberta2:1.0.0",
}

# The 6 legacy "*_pll" flat tool names — fully removed by the migration. The 6
# base model names (currab, ft_esm, ...) persist as catalog Tool names, so they
# are checked separately (absent from TOOL_REGISTRY, present in CATALOG/TOOL_RUNNERS).
_OLD_PLL_FLAT_NAMES = (
    "currab_pll",
    "ft_esm_pll",
    "balm_paired_pll",
    "balm_unpaired_pll",
    "ablang2_pll",
    "antiberta2_pll",
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


def _make_runner(
    tool_name: str, config: AutobioConfig, mode: str = "embedding"
) -> AntibodyLMRunner:
    """Create an AntibodyLMRunner with mocked deps, current_mode pinned to *mode*."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        runner = AntibodyLMRunner(tool_name, config)
    runner.current_mode = get_tool(tool_name).modes[mode]
    return runner


@pytest.fixture()
def currab_runner(config: AutobioConfig) -> AntibodyLMRunner:
    return _make_runner("currab", config)


@pytest.fixture()
def currab_pll_runner(config: AutobioConfig) -> AntibodyLMRunner:
    return _make_runner("currab", config, "pll")


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
    return _make_runner("ablang2", config, "pll")


@pytest.fixture()
def antiberta2_runner(config: AutobioConfig) -> AntibodyLMRunner:
    return _make_runner("antiberta2", config)


@pytest.fixture()
def antiberta2_pll_runner(config: AutobioConfig) -> AntibodyLMRunner:
    return _make_runner("antiberta2", config, "pll")


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
        if tool_name == "balm_unpaired":
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
        """per_position is a typed AntibodyInput field (not passed via extra)."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(
            sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)],
            per_position=True,
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


# ---------------------------------------------------------------------------
# TestAntibodyLMApplyExtraRejections — _apply_extra hardening
# ---------------------------------------------------------------------------


class TestAntibodyLMApplyExtraRejections:
    """``extra`` keys that collide with typed fields or config keys are rejected.

    This is intentional hardening over the pre-migration behavior, which
    silently flat-merged ``extra`` and let it clobber earlier config keys.
    """

    def test_extra_model_name_collision_raises(
        self, currab_runner: AntibodyLMRunner, tmp_path: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(
            sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)],
            extra={"model_name": "some-other-model"},
        )
        with pytest.raises(AutobioError, match="collide"):
            currab_runner.prepare_workspace(input_data, workspace)

    def test_extra_layer_shadow_raises(
        self, currab_runner: AntibodyLMRunner, tmp_path: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(
            sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)],
            extra={"layer": 5},
        )
        with pytest.raises(AutobioError, match="collide"):
            currab_runner.prepare_workspace(input_data, workspace)

    def test_extra_per_position_shadow_raises(
        self, currab_pll_runner: AntibodyLMRunner, tmp_path: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(
            sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)],
            extra={"per_position": True},
        )
        with pytest.raises(AutobioError, match="collide"):
            currab_pll_runner.prepare_workspace(input_data, workspace)


# ---------------------------------------------------------------------------
# TestAntibodyLMByteCompatConfig — full-dict config.json equality, per mode
# ---------------------------------------------------------------------------


class TestAntibodyLMByteCompatConfig:
    """Full-dict ``config.json`` equality tests, pinning key order per mode.

    Key order is taken verbatim from ``.superpowers/sdd/recon/antibody_lm.md``.
    """

    def test_embedding_full_config_defaults(
        self, currab_runner: AntibodyLMRunner, tmp_path: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)])
        currab_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        expected = {
            "model_name": "brineylab/CurrAb",
            "model_family": "esm",
            "chain_separator": "single_cls",
            "input_file": "/workspace/inputs/sequences.json",
            "output_dir": "/workspace/outputs/raw",
            "mode": "embedding",
            "layer": None,
            "pooling": "mean",
            "per_position": False,
            "hf_cache": "/app/antibody-lm/hf_cache",
        }
        assert cfg == expected
        assert list(cfg.keys()) == list(expected.keys())

    def test_pll_full_config_defaults(
        self, currab_pll_runner: AntibodyLMRunner, tmp_path: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)])
        currab_pll_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        expected = {
            "model_name": "brineylab/CurrAb",
            "model_family": "esm",
            "chain_separator": "single_cls",
            "input_file": "/workspace/inputs/sequences.json",
            "output_dir": "/workspace/outputs/raw",
            "mode": "pll",
            "layer": None,
            "pooling": "mean",
            "per_position": False,
            "hf_cache": "/app/antibody-lm/hf_cache",
        }
        assert cfg == expected
        assert list(cfg.keys()) == list(expected.keys())

    def test_pll_full_config_with_per_position(
        self, currab_pll_runner: AntibodyLMRunner, tmp_path: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(
            sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)],
            layer=10,
            pooling="cls",
            per_position=True,
        )
        currab_pll_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        expected = {
            "model_name": "brineylab/CurrAb",
            "model_family": "esm",
            "chain_separator": "single_cls",
            "input_file": "/workspace/inputs/sequences.json",
            "output_dir": "/workspace/outputs/raw",
            "mode": "pll",
            "layer": 10,
            "pooling": "cls",
            "per_position": True,
            "hf_cache": "/app/antibody-lm/hf_cache",
        }
        assert cfg == expected
        assert list(cfg.keys()) == list(expected.keys())

    def test_embedding_full_config_with_flat_extra(
        self, currab_runner: AntibodyLMRunner, tmp_path: Path
    ) -> None:
        """A non-consumed extra key flat-merges after the fixed keys."""
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyInput(
            sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)],
            extra={"custom_flag": "value"},
        )
        currab_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        expected = {
            "model_name": "brineylab/CurrAb",
            "model_family": "esm",
            "chain_separator": "single_cls",
            "input_file": "/workspace/inputs/sequences.json",
            "output_dir": "/workspace/outputs/raw",
            "mode": "embedding",
            "layer": None,
            "pooling": "mean",
            "per_position": False,
            "hf_cache": "/app/antibody-lm/hf_cache",
            "custom_flag": "value",
        }
        assert cfg == expected
        assert list(cfg.keys()) == list(expected.keys())


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


class TestAntibodyLMRegistration:
    """Tests for the catalog Tool + runner registration."""

    @pytest.mark.parametrize("model_name", _MODEL_NAMES)
    def test_registered_as_catalog_tool(self, model_name: str) -> None:
        import autobio.tools  # noqa: F401 - populate registries

        tool = get_tool(model_name)
        assert sorted(tool.modes) == ["embedding", "pll"]
        assert tool.default_mode == "embedding"
        assert tool.category == ToolCategory.EMBEDDING
        assert tool.requires_gpu is True
        assert tool.gpu_count == 1
        assert tool.image_tag == _IMAGE_TAGS[model_name]

    @pytest.mark.parametrize("model_name", _MODEL_NAMES)
    def test_tool_categories_embedding_only(self, model_name: str) -> None:
        import autobio.tools  # noqa: F401

        tool = get_tool(model_name)
        assert tool_categories(tool) == (ToolCategory.EMBEDDING,)

    @pytest.mark.parametrize("model_name", _MODEL_NAMES)
    def test_listed_under_embedding(self, model_name: str) -> None:
        import autobio.tools  # noqa: F401

        assert model_name in list_tools(category=ToolCategory.EMBEDDING)

    @pytest.mark.parametrize(
        ("mode_name", "output_schema", "timeout"),
        [
            ("embedding", EmbeddingOutput, 600),
            ("pll", AntibodyPLLOutput, 1800),
        ],
    )
    @pytest.mark.parametrize("model_name", _MODEL_NAMES)
    def test_mode_schemas_and_timeouts(
        self,
        model_name: str,
        mode_name: str,
        output_schema: type,
        timeout: int,
    ) -> None:
        mode = get_tool(model_name).modes[mode_name]
        assert mode.input_schema is AntibodyInput
        assert mode.output_schema is output_schema
        assert mode.default_timeout == timeout
        assert mode.supports_batch is True
        # Modes don't override the Tool's image — both modes share one image.
        assert mode.image_tag is None

    @pytest.mark.parametrize("model_name", _MODEL_NAMES)
    def test_old_base_names_absent_from_tool_registry(self, model_name: str) -> None:
        import autobio.tools  # noqa: F401

        assert model_name not in TOOL_REGISTRY

    @pytest.mark.parametrize("model_name", _MODEL_NAMES)
    def test_base_names_in_tool_runners(self, model_name: str) -> None:
        import autobio.tools  # noqa: F401

        assert model_name in TOOL_RUNNERS
        assert TOOL_RUNNERS[model_name] is AntibodyLMRunner

    @pytest.mark.parametrize("flat_name", _OLD_PLL_FLAT_NAMES)
    def test_old_pll_flat_names_absent_from_tool_registry(self, flat_name: str) -> None:
        import autobio.tools  # noqa: F401

        assert flat_name not in TOOL_REGISTRY

    @pytest.mark.parametrize("flat_name", _OLD_PLL_FLAT_NAMES)
    def test_old_pll_flat_names_absent_from_tool_runners(self, flat_name: str) -> None:
        import autobio.tools  # noqa: F401

        assert flat_name not in TOOL_RUNNERS

    @pytest.mark.parametrize("flat_name", _OLD_PLL_FLAT_NAMES)
    def test_old_pll_flat_names_absent_from_catalog(self, flat_name: str) -> None:
        import autobio.tools  # noqa: F401
        from autobio.core.catalog import CATALOG

        assert flat_name not in CATALOG

    def test_get_runner_returns_antibody_lm_runner(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = get_runner("currab", config)
        assert isinstance(r, AntibodyLMRunner)
        assert r.tool_name == "currab"

    @pytest.mark.parametrize("flat_name", _OLD_PLL_FLAT_NAMES)
    def test_get_runner_removed_pll_flat_name_raises(
        self, flat_name: str, config: AutobioConfig
    ) -> None:
        with pytest.raises(KeyError, match=flat_name):
            get_runner(flat_name, config)

    @pytest.mark.parametrize("model_name", _MODEL_NAMES)
    def test_notes_populated(self, model_name: str) -> None:
        tool = get_tool(model_name)
        assert len(tool.notes) > 0
        assert len(tool.modes["embedding"].notes) > 0
        assert len(tool.modes["pll"].notes) > 0

    def test_distinct_image_tags(self) -> None:
        """Each model has its own container image tag."""
        tags = {get_tool(n).image_tag for n in _MODEL_NAMES}
        assert len(tags) == len(_MODEL_NAMES)


# ---------------------------------------------------------------------------
# TestAntibodyLMInfoSnapshot
# ---------------------------------------------------------------------------


class TestAntibodyLMInfoSnapshot:
    """``autobio info currab`` output — per-mode notes, output_schema, category."""

    def test_info_snapshot(self) -> None:
        import autobio.tools  # noqa: F401
        from autobio.cli.formatters import OutputFormat, format_tool_info_catalog

        parsed = json.loads(format_tool_info_catalog(get_tool("currab"), OutputFormat.JSON))
        assert [m["name"] for m in parsed["modes"]] == ["embedding", "pll"]

        embedding_mode, pll_mode = parsed["modes"]

        assert len(embedding_mode["notes"]) > 0
        assert "output_schema" in embedding_mode
        assert embedding_mode["category"] == "embedding"

        assert len(pll_mode["notes"]) > 0
        assert "output_schema" in pll_mode
        assert pll_mode["category"] == "embedding"


# ---------------------------------------------------------------------------
# TestAntibodyLMRunMetadataMode — full run() lifecycle threads mode into metadata
# ---------------------------------------------------------------------------


class TestAntibodyLMRunMetadataMode:
    """``run(...).metadata.mode`` reflects the selected mode for each mode."""

    def test_run_metadata_mode_embedding(
        self,
        config: AutobioConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import autobio.tools  # noqa: F401

        output_dir = tmp_path / "ws"
        std_dir = output_dir / "outputs" / "standardized"
        std_dir.mkdir(parents=True)
        (std_dir / "result_data.json").write_text(json.dumps(_SINGLE_EMBEDDING_RESULT))

        monkeypatch.setattr(
            "autobio.core.workspace.Workspace.read_result",
            lambda self: SimpleNamespace(
                status="success", phase="run", exit_code=0, error_message=None
            ),
        )

        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = AntibodyLMRunner("currab", config)

        input_data = AntibodyInput(sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)])
        out = r.run(input_data, gpu="none", output_dir=output_dir, mode="embedding")
        assert out.metadata.mode == "embedding"
        assert out.metadata.tool_name == "currab"
        assert isinstance(out, EmbeddingOutput)

    def test_run_metadata_mode_pll(
        self,
        config: AutobioConfig,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import autobio.tools  # noqa: F401

        output_dir = tmp_path / "ws"
        std_dir = output_dir / "outputs" / "standardized"
        std_dir.mkdir(parents=True)
        (std_dir / "result_data.json").write_text(json.dumps(_PLL_RESULT))

        monkeypatch.setattr(
            "autobio.core.workspace.Workspace.read_result",
            lambda self: SimpleNamespace(
                status="success", phase="run", exit_code=0, error_message=None
            ),
        )

        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
        ):
            r = AntibodyLMRunner("currab", config)

        input_data = AntibodyInput(sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)])
        out = r.run(input_data, gpu="none", output_dir=output_dir, mode="pll")
        assert out.metadata.mode == "pll"
        assert out.metadata.tool_name == "currab"
        assert isinstance(out, AntibodyPLLOutput)
