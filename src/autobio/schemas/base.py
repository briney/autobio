"""Base schema types shared by all tool categories."""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 - Pydantic needs at runtime
from pathlib import Path  # noqa: TC003 - Pydantic needs at runtime
from typing import Any

from pydantic import BaseModel, Field


class RunMetadata(BaseModel):
    """Metadata attached to every tool output.

    Populated by the host runner, not the container.
    """

    tool_name: str
    mode: str | None = None
    tool_version: str
    image_uri: str
    wall_time_seconds: float
    gpu_ids: list[int] | None = None
    workspace_path: Path
    timestamp: datetime


class BaseInput(BaseModel):
    """Base class for all tool input schemas."""

    extra: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Tool-specific parameters passed through to the container. "
            "Keys and values are forwarded as-is into config.json."
        ),
    )


class BaseOutput(BaseModel):
    """Base class for all tool output schemas."""

    metadata: RunMetadata
    raw_output_path: Path = Field(
        description="Path to the outputs/raw/ directory containing unmodified tool outputs."
    )
