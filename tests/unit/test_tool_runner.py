"""Tests for ToolRunner ABC, _resolve_gpu, _build_metadata, run lifecycle, get_runner."""

from __future__ import annotations

import dataclasses
import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from autobio.core.catalog import CATALOG, Mode, Tool, get_tool, register
from autobio.core.config import AutobioConfig
from autobio.core.registry import ToolCategory
from autobio.core.result import (
    ContainerResult,
    GPUNotAvailableError,
    ToolExecutionError,
)
from autobio.core.workspace import Workspace
from autobio.schemas.base import BaseInput, BaseOutput, RunMetadata
from autobio.tools import TOOL_RUNNERS, get_runner
from autobio.tools.base import ToolRunner

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Mock runner subclass
# ---------------------------------------------------------------------------


class MockRunner(ToolRunner):
    """Concrete subclass for testing the ABC."""

    prepare_called = False
    parse_called = False

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        self.prepare_called = True
        workspace.write_config({"mock": True, **input_data.extra})

    def parse_output(self, workspace: Workspace) -> BaseOutput:
        self.parse_called = True
        # Return a minimal BaseOutput — metadata will be overwritten by run()
        return BaseOutput(
            metadata=RunMetadata(
                tool_name="placeholder",
                tool_version="0.0.0",
                image_uri="placeholder",
                wall_time_seconds=0,
                workspace_path=workspace.root,
                timestamp="2026-01-01T00:00:00Z",
            ),
            raw_output_path=workspace.raw_output_dir,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_MOCK_TOOL_NAME = "_test_mock_tool"
_MOCK_MODE_NAME = "default"


def _make_mock_tool(*, requires_gpu: bool = True, gpu_count: int = 1) -> Tool:
    """Build a mock catalog Tool with a single mode for testing."""
    return Tool(
        name=_MOCK_TOOL_NAME,
        display_name="Mock Tool",
        category=ToolCategory.SCORING,
        description="Mock tool for testing.",
        version="0.1.0",
        image_tag="mock-tool:0.1.0",
        requires_gpu=requires_gpu,
        gpu_count=gpu_count,
        default_mode=_MOCK_MODE_NAME,
        modes={
            _MOCK_MODE_NAME: Mode(
                name=_MOCK_MODE_NAME,
                display_name="Default",
                description="Mock mode for testing.",
                input_schema=BaseInput,
                output_schema=BaseOutput,
                default_timeout=600,
            ),
        },
    )


@pytest.fixture(autouse=True)
def _register_mock_tool():
    """Register and unregister a mock catalog Tool for every test."""
    catalog_snapshot = dict(CATALOG)
    CATALOG.clear()
    register(_make_mock_tool())
    yield
    CATALOG.clear()
    CATALOG.update(catalog_snapshot)


@pytest.fixture()
def config() -> AutobioConfig:
    return AutobioConfig.resolve()


@pytest.fixture()
def runner(config: AutobioConfig) -> MockRunner:
    """Create a MockRunner with mocked ContainerManager and GPUManager."""
    with (
        patch("autobio.tools.base.ContainerManager") as cm_cls,
        patch("autobio.tools.base.GPUManager") as gm_cls,
    ):
        mock_cm = MagicMock()
        mock_gm = MagicMock()
        mock_gm.allocate.return_value = [0]
        cm_cls.return_value = mock_cm
        gm_cls.return_value = mock_gm

        r = MockRunner(_MOCK_TOOL_NAME, config)
    return r


# ---------------------------------------------------------------------------
# __init__ tests
# ---------------------------------------------------------------------------


class TestToolRunnerInit:
    """Tests for ToolRunner.__init__."""

    def test_init_sets_attributes(self, runner: MockRunner) -> None:
        assert runner.tool_name == _MOCK_TOOL_NAME
        assert runner.tool is get_tool(_MOCK_TOOL_NAME)
        assert runner.config is not None

    def test_init_unknown_tool_raises(self, config: AutobioConfig) -> None:
        with (
            patch("autobio.tools.base.ContainerManager"),
            patch("autobio.tools.base.GPUManager"),
            pytest.raises(KeyError, match="Unknown tool.*no_such_tool"),
        ):
            MockRunner("no_such_tool", config)


# ---------------------------------------------------------------------------
# _resolve_gpu tests
# ---------------------------------------------------------------------------


class TestResolveGpu:
    """Tests for ToolRunner._resolve_gpu."""

    def test_auto_with_gpu_tool(self, runner: MockRunner) -> None:
        runner._gpu.allocate.return_value = [0]
        result = runner._resolve_gpu("auto")
        runner._gpu.allocate.assert_called_once_with(count=1)
        assert result == [0]

    def test_auto_with_no_gpu_tool(self, runner: MockRunner) -> None:
        runner.tool = dataclasses.replace(runner.tool, requires_gpu=False, gpu_count=0)
        result = runner._resolve_gpu("auto")
        runner._gpu.allocate.assert_not_called()
        assert result == []

    def test_none_returns_empty(self, runner: MockRunner) -> None:
        result = runner._resolve_gpu("none")
        runner._gpu.allocate.assert_not_called()
        assert result == []

    def test_list_of_ids(self, runner: MockRunner) -> None:
        runner._gpu.allocate.return_value = [1, 2]
        result = runner._resolve_gpu([1, 2])
        runner._gpu.allocate.assert_called_once_with(device_ids=[1, 2])
        assert result == [1, 2]

    def test_empty_list_returns_empty(self, runner: MockRunner) -> None:
        result = runner._resolve_gpu([])
        runner._gpu.allocate.assert_not_called()
        assert result == []

    def test_comma_string(self, runner: MockRunner) -> None:
        runner._gpu.allocate.return_value = [0, 3]
        result = runner._resolve_gpu("0,3")
        runner._gpu.allocate.assert_called_once_with(device_ids=[0, 3])
        assert result == [0, 3]

    def test_gpu_unavailable_propagates(self, runner: MockRunner) -> None:
        runner._gpu.allocate.side_effect = GPUNotAvailableError("no GPUs")
        with pytest.raises(GPUNotAvailableError):
            runner._resolve_gpu("auto")


# ---------------------------------------------------------------------------
# _build_metadata tests
# ---------------------------------------------------------------------------


class TestBuildMetadata:
    """Tests for ToolRunner._build_metadata."""

    def test_metadata_fields(self, runner: MockRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        meta = runner._build_metadata(workspace, wall_time=12.5, gpu_ids=[0, 1], image_uri="img:1")
        assert isinstance(meta, RunMetadata)
        assert meta.tool_name == _MOCK_TOOL_NAME
        assert meta.tool_version == "0.1.0"
        assert meta.image_uri == "img:1"
        assert meta.wall_time_seconds == 12.5
        assert meta.gpu_ids == [0, 1]
        assert meta.workspace_path == workspace.root

    def test_no_gpus_yields_none(self, runner: MockRunner, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        meta = runner._build_metadata(workspace, wall_time=1.0, gpu_ids=[], image_uri="img:1")
        assert meta.gpu_ids is None


# ---------------------------------------------------------------------------
# _read_logs tests
# ---------------------------------------------------------------------------


class TestReadLogs:
    """Tests for ToolRunner._read_logs."""

    def test_reads_stderr(self, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        (workspace.logs_dir / "stderr.log").write_text("error output here")
        assert ToolRunner._read_logs(workspace) == "error output here"

    def test_missing_log_returns_empty(self, tmp_path: Path) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        assert ToolRunner._read_logs(workspace) == ""


# ---------------------------------------------------------------------------
# run() lifecycle tests
# ---------------------------------------------------------------------------


def _write_success_result(workspace_root: Path) -> None:
    """Write a successful result.json into the workspace."""
    result = {
        "status": "success",
        "exit_code": 0,
        "phase": "complete",
        "wall_time_seconds": 5.0,
        "gpu_ids": [0],
        "completed": 1,
        "total": 1,
        "outputs": {"standardized_files": [], "raw_files": []},
    }
    (workspace_root / "result.json").write_text(json.dumps(result))


def _write_failure_result(workspace_root: Path) -> None:
    """Write a failed result.json into the workspace."""
    result = {
        "status": "failed",
        "exit_code": 1,
        "phase": "execution",
        "error_type": "RuntimeError",
        "error_message": "Tool crashed",
        "wall_time_seconds": 2.0,
        "outputs": {"standardized_files": [], "raw_files": []},
    }
    (workspace_root / "result.json").write_text(json.dumps(result))


class TestRunLifecycle:
    """Tests for the full run() method."""

    def test_successful_run(self, runner: MockRunner, tmp_path: Path) -> None:
        """Verify prepare → container run → parse_output → metadata attachment."""

        def fake_container_run(
            image_uri: str,
            workspace: Path,
            gpu_ids: list[int] | None = None,
            timeout: int | None = None,
            memory_limit: str | None = None,
        ) -> ContainerResult:
            _write_success_result(workspace)
            return ContainerResult(
                exit_code=0,
                stdout_log=workspace / "logs" / "stdout.log",
                stderr_log=workspace / "logs" / "stderr.log",
            )

        runner._container.run.side_effect = fake_container_run
        runner._container.ensure_image.return_value = None
        runner._gpu.allocate.return_value = [0]

        output_dir = tmp_path / "output"
        input_data = BaseInput()
        result = runner.run(input_data, gpu="auto", output_dir=output_dir)

        assert runner.prepare_called
        assert runner.parse_called
        assert isinstance(result, BaseOutput)
        assert result.metadata.tool_name == _MOCK_TOOL_NAME
        runner._container.ensure_image.assert_called_once()
        runner._container.run.assert_called_once()

    def test_failure_raises_tool_execution_error(self, runner: MockRunner, tmp_path: Path) -> None:
        """Container failure raises ToolExecutionError with correct fields."""

        def fake_container_run(
            image_uri: str,
            workspace: Path,
            gpu_ids: list[int] | None = None,
            timeout: int | None = None,
            memory_limit: str | None = None,
        ) -> ContainerResult:
            _write_failure_result(workspace)
            return ContainerResult(
                exit_code=1,
                stdout_log=workspace / "logs" / "stdout.log",
                stderr_log=workspace / "logs" / "stderr.log",
            )

        runner._container.run.side_effect = fake_container_run
        runner._container.ensure_image.return_value = None
        runner._gpu.allocate.return_value = [0]

        with pytest.raises(ToolExecutionError) as exc_info:
            runner.run(BaseInput(), gpu="auto", output_dir=tmp_path / "out")

        err = exc_info.value
        assert err.phase == "execution"
        assert err.exit_code == 1
        assert "Tool crashed" in err.error_message

    def test_gpu_released_on_success(self, runner: MockRunner, tmp_path: Path) -> None:
        """GPUs are released after a successful run."""

        def fake_container_run(
            image_uri: str,
            workspace: Path,
            **kwargs: object,
        ) -> ContainerResult:
            _write_success_result(workspace)
            return ContainerResult(
                exit_code=0,
                stdout_log=workspace / "logs" / "stdout.log",
                stderr_log=workspace / "logs" / "stderr.log",
            )

        runner._container.run.side_effect = fake_container_run
        runner._container.ensure_image.return_value = None
        runner._gpu.allocate.return_value = [0]

        runner.run(BaseInput(), gpu="auto", output_dir=tmp_path / "out")
        runner._gpu.release.assert_called_once_with([0])

    def test_gpu_released_on_failure(self, runner: MockRunner, tmp_path: Path) -> None:
        """GPUs are released even when the run fails."""

        def fake_container_run(
            image_uri: str,
            workspace: Path,
            **kwargs: object,
        ) -> ContainerResult:
            _write_failure_result(workspace)
            return ContainerResult(
                exit_code=1,
                stdout_log=workspace / "logs" / "stdout.log",
                stderr_log=workspace / "logs" / "stderr.log",
            )

        runner._container.run.side_effect = fake_container_run
        runner._container.ensure_image.return_value = None
        runner._gpu.allocate.return_value = [0]

        with pytest.raises(ToolExecutionError):
            runner.run(BaseInput(), gpu="auto", output_dir=tmp_path / "out")

        runner._gpu.release.assert_called_once_with([0])

    def test_temp_workspace_cleaned_up(self, runner: MockRunner, tmp_path: Path) -> None:
        """Temp workspace is removed after successful run."""
        created_roots: list[Path] = []

        original_create = Workspace.create

        @classmethod  # type: ignore[misc]
        def patched_create(cls: type, output_dir: Path | None = None) -> Workspace:
            ws = original_create.__func__(cls, output_dir)  # type: ignore[attr-defined]
            created_roots.append(ws.root)
            return ws

        def fake_container_run(
            image_uri: str,
            workspace: Path,
            **kwargs: object,
        ) -> ContainerResult:
            _write_success_result(workspace)
            return ContainerResult(
                exit_code=0,
                stdout_log=workspace / "logs" / "stdout.log",
                stderr_log=workspace / "logs" / "stderr.log",
            )

        runner._container.run.side_effect = fake_container_run
        runner._container.ensure_image.return_value = None
        runner._gpu.allocate.return_value = []

        with patch.object(Workspace, "create", patched_create):
            runner.run(BaseInput(), gpu="none", output_dir=None)

        assert len(created_roots) == 1
        assert not created_roots[0].exists()

    def test_user_workspace_preserved(self, runner: MockRunner, tmp_path: Path) -> None:
        """User-specified workspace persists after run."""

        def fake_container_run(
            image_uri: str,
            workspace: Path,
            **kwargs: object,
        ) -> ContainerResult:
            _write_success_result(workspace)
            return ContainerResult(
                exit_code=0,
                stdout_log=workspace / "logs" / "stdout.log",
                stderr_log=workspace / "logs" / "stderr.log",
            )

        runner._container.run.side_effect = fake_container_run
        runner._container.ensure_image.return_value = None
        runner._gpu.allocate.return_value = []

        output_dir = tmp_path / "kept"
        runner.run(BaseInput(), gpu="none", output_dir=output_dir)

        assert output_dir.exists()

    def test_default_timeout_from_mode(self, runner: MockRunner, tmp_path: Path) -> None:
        """When no timeout is given, uses the active mode's default_timeout."""

        def fake_container_run(
            image_uri: str,
            workspace: Path,
            **kwargs: object,
        ) -> ContainerResult:
            _write_success_result(workspace)
            return ContainerResult(
                exit_code=0,
                stdout_log=workspace / "logs" / "stdout.log",
                stderr_log=workspace / "logs" / "stderr.log",
            )

        runner._container.run.side_effect = fake_container_run
        runner._container.ensure_image.return_value = None
        runner._gpu.allocate.return_value = []

        runner.run(BaseInput(), gpu="none", output_dir=tmp_path / "out")

        assert runner.current_mode is not None
        assert runner.current_mode.default_timeout == 600
        call_kwargs = runner._container.run.call_args
        assert call_kwargs.kwargs.get("timeout") == 600 or call_kwargs[1].get("timeout") == 600


# ---------------------------------------------------------------------------
# get_runner tests
# ---------------------------------------------------------------------------


class TestGetRunner:
    """Tests for tools.get_runner."""

    def test_get_runner_success(self, config: AutobioConfig) -> None:
        TOOL_RUNNERS[_MOCK_TOOL_NAME] = MockRunner
        try:
            with (
                patch("autobio.tools.base.ContainerManager"),
                patch("autobio.tools.base.GPUManager"),
            ):
                r = get_runner(_MOCK_TOOL_NAME, config)
            assert isinstance(r, MockRunner)
            assert r.tool_name == _MOCK_TOOL_NAME
        finally:
            TOOL_RUNNERS.pop(_MOCK_TOOL_NAME, None)

    def test_get_runner_unknown_raises(self, config: AutobioConfig) -> None:
        with pytest.raises(KeyError, match="No runner registered"):
            get_runner("nonexistent_tool", config)
