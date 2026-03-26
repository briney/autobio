"""Exception hierarchy, run-result model, and container-result dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from pathlib import Path

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AutobioError(Exception):
    """Base exception for all autobio errors."""


class ContainerNotFoundError(AutobioError):
    """Image not available and could not be pulled."""


class GPUNotAvailableError(AutobioError):
    """Requested GPU(s) not available."""


class ToolExecutionError(AutobioError):
    """Container exited with non-zero status."""

    def __init__(
        self,
        phase: str,
        exit_code: int,
        error_message: str,
        logs: str,
        wall_time: float,
    ) -> None:
        self.phase = phase
        self.exit_code = exit_code
        self.error_message = error_message
        self.logs = logs
        self.wall_time = wall_time
        super().__init__(f"Tool failed during {phase} (exit {exit_code}): {error_message}")


class ToolTimeoutError(AutobioError):
    """Container exceeded timeout."""


# ---------------------------------------------------------------------------
# RunResult — deserialized result.json written by containers
# ---------------------------------------------------------------------------


class _RunOutputs(BaseModel):
    """Files listed inside ``result.json``."""

    standardized_files: list[str] = []
    raw_files: list[str] = []


class RunResult(BaseModel):
    """Structured representation of a container's ``result.json``."""

    status: str
    exit_code: int
    phase: str
    error_type: str | None = None
    error_message: str | None = None
    wall_time_seconds: float
    gpu_ids: list[int] | None = None
    completed: int = 0
    total: int = 1
    outputs: _RunOutputs = _RunOutputs()


# ---------------------------------------------------------------------------
# ContainerResult — returned by ContainerManager.run()
# ---------------------------------------------------------------------------


@dataclass
class ContainerResult:
    """Lightweight result of a container execution."""

    exit_code: int
    stdout_log: Path
    stderr_log: Path
