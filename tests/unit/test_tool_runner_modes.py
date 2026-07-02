"""Tests for catalog (Mode-aware) dispatch in ToolRunner, alongside the legacy path."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from autobio.core.catalog import CATALOG, Mode, Tool, register
from autobio.core.config import AutobioConfig
from autobio.core.registry import ToolCategory
from autobio.core.result import AutobioError
from autobio.schemas.base import BaseInput, BaseOutput
from autobio.tools.base import ToolRunner


@pytest.fixture(autouse=True)
def _clean_catalog():
    snapshot = dict(CATALOG)
    CATALOG.clear()
    yield
    CATALOG.clear()
    CATALOG.update(snapshot)


class _Input(BaseInput):
    pass


class _Output(BaseOutput):
    pass


class _CaptureRunner(ToolRunner):
    """Records the mode active during prepare_workspace; minimal parse_output."""

    captured_mode: str | None = None

    def prepare_workspace(self, input_data, workspace) -> None:
        self.captured_mode = self.current_mode.name if self.current_mode else None

    def parse_output(self, workspace) -> _Output:
        return _Output(
            metadata=self._build_metadata(workspace, 0.0, [], ""),
            raw_output_path=workspace.raw_output_dir,
        )


def _register_faketool() -> None:
    register(
        Tool(
            name="faketool",
            display_name="Fake",
            category=ToolCategory.SCORING,
            description="fake",
            version="9.9.9",
            image_tag="fake:1.0.0",
            requires_gpu=False,
            gpu_count=0,
            default_mode="alpha",
            modes={
                "alpha": Mode("alpha", "Alpha", "a", _Input, _Output, default_timeout=111),
                "beta": Mode(
                    "beta",
                    "Beta",
                    "b",
                    _Input,
                    _Output,
                    default_timeout=222,
                    image_tag="fake-beta:1.0.0",
                ),
            },
        )
    )


def _make_runner(tool_name: str) -> _CaptureRunner:
    with patch("autobio.tools.base.ContainerManager"), patch("autobio.tools.base.GPUManager"):
        return _CaptureRunner(tool_name, AutobioConfig.resolve())


def test_init_resolves_catalog_tool() -> None:
    _register_faketool()
    runner = _make_runner("faketool")
    assert runner.tool is not None
    assert runner.entry is None
    assert runner.current_mode is None


def test_init_unknown_name_lists_available() -> None:
    _register_faketool()
    with pytest.raises(KeyError, match="faketool"):
        _make_runner("does-not-exist")


def test_resolve_mode_default_and_explicit() -> None:
    _register_faketool()
    runner = _make_runner("faketool")
    assert runner._resolve_mode(None).name == "alpha"
    assert runner._resolve_mode("beta").name == "beta"


def test_resolve_mode_unknown_raises() -> None:
    _register_faketool()
    runner = _make_runner("faketool")
    with pytest.raises(AutobioError, match="Unknown mode"):
        runner._resolve_mode("gamma")


def test_image_and_timeout_use_mode_override() -> None:
    _register_faketool()
    runner = _make_runner("faketool")
    runner.current_mode = runner._resolve_mode("beta")
    assert runner._image_tag() == "fake-beta:1.0.0"
    assert runner._default_timeout() == 222
    runner.current_mode = runner._resolve_mode("alpha")
    assert runner._image_tag() == "fake:1.0.0"  # falls back to Tool.image_tag
    assert runner._default_timeout() == 111


def test_run_sets_current_mode_and_mode_metadata(tmp_path, monkeypatch) -> None:
    _register_faketool()
    runner = _make_runner("faketool")
    monkeypatch.setattr(
        "autobio.core.workspace.Workspace.read_result",
        lambda self: SimpleNamespace(
            status="success", phase="run", exit_code=0, error_message=None
        ),
    )
    out = runner.run(_Input(), gpu="none", output_dir=tmp_path, mode="beta")
    assert runner.captured_mode == "beta"
    assert out.metadata.tool_version == "9.9.9"
    assert out.metadata.image_uri.endswith("fake-beta:1.0.0")


def test_run_rejects_mode_for_legacy_tool() -> None:
    # 'prodigy' is a legacy flat tool (in TOOL_REGISTRY, not CATALOG) — imported for real.
    import autobio.tools  # noqa: F401 - populate registries

    with patch("autobio.tools.base.ContainerManager"), patch("autobio.tools.base.GPUManager"):
        runner = _CaptureRunner("prodigy", AutobioConfig.resolve())
    assert runner.entry is not None
    assert runner.tool is None
    with pytest.raises(AutobioError, match="does not support modes"):
        runner.run(_Input(), gpu="none", mode="whatever")
