"""Tests for autobio.schemas.base."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from autobio.schemas.base import BaseInput, BaseOutput, RunMetadata

# ---------------------------------------------------------------------------
# RunMetadata
# ---------------------------------------------------------------------------


class TestRunMetadata:
    def _make(self, **overrides: object) -> RunMetadata:
        defaults: dict[str, object] = {
            "tool_name": "test-tool",
            "tool_version": "1.0.0",
            "image_uri": "ghcr.io/briney/autobio-test:1.0.0",
            "wall_time_seconds": 10.0,
            "workspace_path": Path("/tmp/ws"),
            "timestamp": datetime(2025, 1, 1, tzinfo=UTC),
        }
        defaults.update(overrides)
        return RunMetadata.model_validate(defaults)

    def test_required_fields(self) -> None:
        meta = self._make()
        assert meta.tool_name == "test-tool"
        assert meta.tool_version == "1.0.0"

    def test_gpu_ids_optional(self) -> None:
        meta = self._make()
        assert meta.gpu_ids is None

        meta_gpu = self._make(gpu_ids=[0, 1])
        assert meta_gpu.gpu_ids == [0, 1]

    def test_round_trip(self) -> None:
        meta = self._make(gpu_ids=[0])
        dumped = meta.model_dump()
        restored = RunMetadata.model_validate(dumped)
        assert restored.tool_name == meta.tool_name
        assert restored.gpu_ids == meta.gpu_ids

    def test_missing_required_raises(self) -> None:
        with pytest.raises(ValidationError):
            RunMetadata.model_validate({"tool_name": "x"})


# ---------------------------------------------------------------------------
# BaseInput
# ---------------------------------------------------------------------------


class TestBaseInput:
    def test_default_extra_is_empty_dict(self) -> None:
        inp = BaseInput()
        assert inp.extra == {}

    def test_extra_passthrough(self) -> None:
        inp = BaseInput(extra={"custom_param": 42, "flag": True})
        assert inp.extra["custom_param"] == 42
        assert inp.extra["flag"] is True

    def test_round_trip(self) -> None:
        inp = BaseInput(extra={"key": "value"})
        dumped = inp.model_dump()
        restored = BaseInput.model_validate(dumped)
        assert restored.extra == inp.extra


# ---------------------------------------------------------------------------
# BaseOutput
# ---------------------------------------------------------------------------


class TestBaseOutput:
    def _make_metadata(self) -> dict[str, object]:
        return {
            "tool_name": "test",
            "tool_version": "1.0",
            "image_uri": "img:1",
            "wall_time_seconds": 1.0,
            "workspace_path": "/tmp/ws",
            "timestamp": datetime(2025, 1, 1, tzinfo=UTC).isoformat(),
        }

    def test_required_metadata(self) -> None:
        out = BaseOutput(
            metadata=self._make_metadata(),  # type: ignore[arg-type]
            raw_output_path=Path("/tmp/ws/outputs/raw"),
        )
        assert out.metadata.tool_name == "test"
        assert out.raw_output_path == Path("/tmp/ws/outputs/raw")

    def test_missing_metadata_raises(self) -> None:
        with pytest.raises(ValidationError):
            BaseOutput.model_validate({"raw_output_path": "/tmp"})

    def test_round_trip(self) -> None:
        out = BaseOutput(
            metadata=self._make_metadata(),  # type: ignore[arg-type]
            raw_output_path=Path("/tmp/ws/outputs/raw"),
        )
        dumped = out.model_dump()
        restored = BaseOutput.model_validate(dumped)
        assert restored.metadata.tool_name == out.metadata.tool_name
