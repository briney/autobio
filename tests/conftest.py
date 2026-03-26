"""Shared test fixtures for the autobio test suite."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - used in fixture type hints
from unittest.mock import patch

import pytest

from autobio.core.config import AutobioConfig
from autobio.core.workspace import Workspace


@pytest.fixture()
def tmp_workspace(tmp_path: Path) -> Workspace:
    """An initialised workspace in a temporary directory."""
    return Workspace.create(tmp_path / "workspace")


@pytest.fixture()
def sample_config() -> AutobioConfig:
    """An AutobioConfig with default values."""
    return AutobioConfig.resolve()


@pytest.fixture()
def monkeypatch_no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch pynvml so that no GPUs are discovered."""
    with patch("autobio.core.gpu.GPUManager._discover_gpus", return_value=[]):
        yield  # type: ignore[func-returns-value]
