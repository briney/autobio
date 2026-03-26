"""Tests for autobio.core.gpu."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from autobio.core.gpu import GPUManager
from autobio.core.result import GPUNotAvailableError


@pytest.fixture()
def two_gpu_manager() -> GPUManager:
    """A GPUManager that sees 2 GPUs (mocked)."""
    with patch.object(GPUManager, "_discover_gpus", return_value=[0, 1]):
        return GPUManager()


@pytest.fixture()
def no_gpu_manager() -> GPUManager:
    """A GPUManager with no GPUs."""
    with patch.object(GPUManager, "_discover_gpus", return_value=[]):
        return GPUManager()


class TestDiscovery:
    def test_graceful_no_gpu(self, no_gpu_manager: GPUManager) -> None:
        assert no_gpu_manager.available_gpus == []
        assert no_gpu_manager.has_gpus is False

    def test_two_gpus_discovered(self, two_gpu_manager: GPUManager) -> None:
        assert two_gpu_manager.available_gpus == [0, 1]
        assert two_gpu_manager.has_gpus is True


class TestAllocateRelease:
    def test_allocate_one(self, two_gpu_manager: GPUManager) -> None:
        ids = two_gpu_manager.allocate(count=1)
        assert len(ids) == 1
        assert ids[0] in (0, 1)

    def test_allocate_specific_devices(self, two_gpu_manager: GPUManager) -> None:
        ids = two_gpu_manager.allocate(device_ids=[1])
        assert ids == [1]
        assert 1 not in two_gpu_manager.available_gpus

    def test_allocate_all(self, two_gpu_manager: GPUManager) -> None:
        ids = two_gpu_manager.allocate(count=2)
        assert sorted(ids) == [0, 1]
        assert two_gpu_manager.available_gpus == []

    def test_release(self, two_gpu_manager: GPUManager) -> None:
        ids = two_gpu_manager.allocate(count=2)
        two_gpu_manager.release(ids)
        assert sorted(two_gpu_manager.available_gpus) == [0, 1]

    def test_double_allocate_raises(self, two_gpu_manager: GPUManager) -> None:
        two_gpu_manager.allocate(device_ids=[0])
        with pytest.raises(GPUNotAvailableError):
            two_gpu_manager.allocate(device_ids=[0])

    def test_allocate_more_than_available_raises(
        self,
        two_gpu_manager: GPUManager,
    ) -> None:
        with pytest.raises(GPUNotAvailableError):
            two_gpu_manager.allocate(count=3)

    def test_allocate_on_no_gpu_raises(self, no_gpu_manager: GPUManager) -> None:
        with pytest.raises(GPUNotAvailableError):
            no_gpu_manager.allocate(count=1)

    def test_allocate_nonexistent_device_raises(
        self,
        two_gpu_manager: GPUManager,
    ) -> None:
        with pytest.raises(GPUNotAvailableError):
            two_gpu_manager.allocate(device_ids=[5])


class TestProperties:
    def test_available_gpus_after_partial_alloc(
        self,
        two_gpu_manager: GPUManager,
    ) -> None:
        two_gpu_manager.allocate(device_ids=[0])
        assert two_gpu_manager.available_gpus == [1]
