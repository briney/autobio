"""Tests for autobio.core.result."""

from __future__ import annotations

from pathlib import Path

import pytest

from autobio.core.result import (
    AutobioError,
    ContainerNotFoundError,
    ContainerResult,
    GPUNotAvailableError,
    RunResult,
    ToolExecutionError,
    ToolTimeoutError,
)

# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    def test_all_inherit_from_autobio_error(self) -> None:
        assert issubclass(ContainerNotFoundError, AutobioError)
        assert issubclass(GPUNotAvailableError, AutobioError)
        assert issubclass(ToolExecutionError, AutobioError)
        assert issubclass(ToolTimeoutError, AutobioError)

    def test_autobio_error_inherits_from_exception(self) -> None:
        assert issubclass(AutobioError, Exception)

    def test_tool_execution_error_stores_fields(self) -> None:
        err = ToolExecutionError(
            phase="execution",
            exit_code=1,
            error_message="segfault",
            logs="stderr output",
            wall_time=12.5,
        )
        assert err.phase == "execution"
        assert err.exit_code == 1
        assert err.error_message == "segfault"
        assert err.logs == "stderr output"
        assert err.wall_time == 12.5

    def test_tool_execution_error_str(self) -> None:
        err = ToolExecutionError(
            phase="setup",
            exit_code=2,
            error_message="bad config",
            logs="",
            wall_time=0.1,
        )
        assert "setup" in str(err)
        assert "bad config" in str(err)

    def test_catch_as_autobio_error(self) -> None:
        with pytest.raises(AutobioError):
            raise ToolExecutionError(
                phase="execution",
                exit_code=1,
                error_message="fail",
                logs="",
                wall_time=0.0,
            )


# ---------------------------------------------------------------------------
# RunResult deserialization
# ---------------------------------------------------------------------------


class TestRunResult:
    def test_success_result(self) -> None:
        data = {
            "status": "success",
            "exit_code": 0,
            "phase": "complete",
            "wall_time_seconds": 42.5,
            "gpu_ids": [0],
            "completed": 1,
            "total": 1,
            "outputs": {
                "standardized_files": ["outputs/standardized/result.json"],
                "raw_files": ["outputs/raw/model.pdb"],
            },
        }
        result = RunResult.model_validate(data)
        assert result.status == "success"
        assert result.exit_code == 0
        assert result.phase == "complete"
        assert result.wall_time_seconds == 42.5
        assert result.gpu_ids == [0]
        assert result.completed == 1
        assert len(result.outputs.standardized_files) == 1
        assert len(result.outputs.raw_files) == 1

    def test_failure_result(self) -> None:
        data = {
            "status": "failed",
            "exit_code": 1,
            "phase": "execution",
            "error_type": "runtime",
            "error_message": "Tool exited with code 1",
            "wall_time_seconds": 3.2,
            "completed": 0,
            "total": 1,
        }
        result = RunResult.model_validate(data)
        assert result.status == "failed"
        assert result.error_type == "runtime"
        assert result.error_message == "Tool exited with code 1"

    def test_minimal_result(self) -> None:
        data = {
            "status": "success",
            "exit_code": 0,
            "phase": "complete",
            "wall_time_seconds": 1.0,
        }
        result = RunResult.model_validate(data)
        assert result.gpu_ids is None
        assert result.completed == 0
        assert result.total == 1
        assert result.outputs.standardized_files == []

    def test_round_trip_serialization(self) -> None:
        data = {
            "status": "success",
            "exit_code": 0,
            "phase": "complete",
            "wall_time_seconds": 10.0,
            "gpu_ids": [0, 1],
            "completed": 5,
            "total": 5,
            "outputs": {
                "standardized_files": ["a.json"],
                "raw_files": ["b.pdb"],
            },
        }
        result = RunResult.model_validate(data)
        dumped = result.model_dump()
        restored = RunResult.model_validate(dumped)
        assert restored == result


# ---------------------------------------------------------------------------
# ContainerResult
# ---------------------------------------------------------------------------


class TestContainerResult:
    def test_fields(self) -> None:
        cr = ContainerResult(
            exit_code=0,
            stdout_log=Path("/tmp/ws/logs/stdout.log"),
            stderr_log=Path("/tmp/ws/logs/stderr.log"),
        )
        assert cr.exit_code == 0
        assert cr.stdout_log == Path("/tmp/ws/logs/stdout.log")
        assert cr.stderr_log == Path("/tmp/ws/logs/stderr.log")
