"""autobio — Unified, agentic-friendly interface to computational biology tools."""

from __future__ import annotations

from autobio.core.config import AutobioConfig
from autobio.core.result import (
    AutobioError,
    ContainerNotFoundError,
    GPUNotAvailableError,
    RunResult,
    ToolExecutionError,
    ToolTimeoutError,
)
from autobio.schemas.base import BaseInput, BaseOutput, RunMetadata

__version__ = "0.0.1"

__all__ = [
    "__version__",
    # core
    "AutobioConfig",
    # exceptions
    "AutobioError",
    "ContainerNotFoundError",
    "GPUNotAvailableError",
    "ToolExecutionError",
    "ToolTimeoutError",
    # result
    "RunResult",
    # base schemas
    "BaseInput",
    "BaseOutput",
    "RunMetadata",
]
