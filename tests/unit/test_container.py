"""Tests for autobio.core.container."""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from autobio.core.config import AutobioConfig
from autobio.core.container import ContainerManager, ImageInfo
from autobio.core.result import ContainerNotFoundError, ContainerResult, ToolTimeoutError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_docker_client():
    """Yield a mocked DockerClient, patching the import in container.py."""
    with patch("autobio.core.container.DockerClient") as mock_cls:
        client = MagicMock()
        mock_cls.return_value = client
        yield client


@pytest.fixture()
def manager(mock_docker_client: MagicMock) -> ContainerManager:
    """A ContainerManager backed by the mocked Docker client."""
    return ContainerManager(AutobioConfig.resolve())


# ---------------------------------------------------------------------------
# ImageInfo
# ---------------------------------------------------------------------------


class TestImageInfo:
    def test_construction(self) -> None:
        info = ImageInfo(
            uri="ghcr.io/briney/autobio-tool:1.0",
            tag="1.0",
            size=500_000_000,
            created=datetime(2025, 6, 1),
        )
        assert info.uri == "ghcr.io/briney/autobio-tool:1.0"
        assert info.tag == "1.0"
        assert info.size == 500_000_000
        assert info.created == datetime(2025, 6, 1)


# ---------------------------------------------------------------------------
# ContainerManager.__init__
# ---------------------------------------------------------------------------


class TestContainerManagerInit:
    def test_default_config(self, mock_docker_client: MagicMock) -> None:
        with patch("autobio.core.container.DockerClient") as mock_cls:
            ContainerManager(AutobioConfig.resolve())
            mock_cls.assert_called_once_with(host=None)

    def test_custom_docker_host(self, mock_docker_client: MagicMock) -> None:
        with patch("autobio.core.container.DockerClient") as mock_cls:
            cfg = AutobioConfig(docker_host="tcp://remote:2375")
            ContainerManager(cfg)
            mock_cls.assert_called_once_with(host="tcp://remote:2375")


# ---------------------------------------------------------------------------
# ensure_image / pull_image
# ---------------------------------------------------------------------------


class TestEnsureImage:
    def test_skips_pull_when_image_exists(
        self, manager: ContainerManager, mock_docker_client: MagicMock
    ) -> None:
        mock_docker_client.image.exists.return_value = True
        manager.ensure_image("ghcr.io/briney/autobio-tool:1.0")
        mock_docker_client.image.pull.assert_not_called()

    def test_pulls_when_image_missing(
        self, manager: ContainerManager, mock_docker_client: MagicMock
    ) -> None:
        mock_docker_client.image.exists.return_value = False
        manager.ensure_image("ghcr.io/briney/autobio-tool:1.0")
        mock_docker_client.image.pull.assert_called_once_with("ghcr.io/briney/autobio-tool:1.0")

    def test_raises_container_not_found_on_pull_failure(
        self, manager: ContainerManager, mock_docker_client: MagicMock
    ) -> None:
        from python_on_whales.exceptions import DockerException

        mock_docker_client.image.exists.return_value = False
        mock_docker_client.image.pull.side_effect = DockerException(
            ["docker", "pull", "bad:image"], 1
        )
        with pytest.raises(ContainerNotFoundError, match="Failed to pull"):
            manager.ensure_image("bad:image")


class TestPullImage:
    def test_successful_pull(
        self, manager: ContainerManager, mock_docker_client: MagicMock
    ) -> None:
        manager.pull_image("ghcr.io/briney/autobio-tool:2.0")
        mock_docker_client.image.pull.assert_called_once_with("ghcr.io/briney/autobio-tool:2.0")

    def test_pull_failure_wraps_docker_exception(
        self, manager: ContainerManager, mock_docker_client: MagicMock
    ) -> None:
        from python_on_whales.exceptions import DockerException

        mock_docker_client.image.pull.side_effect = DockerException(["docker", "pull"], 1)
        with pytest.raises(ContainerNotFoundError):
            manager.pull_image("nosuch:latest")


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


class TestRun:
    def _setup_container_mock(self, mock_docker_client: MagicMock, exit_code: int = 0) -> MagicMock:
        container = MagicMock()
        mock_docker_client.container.create.return_value = container
        mock_docker_client.container.wait.return_value = exit_code
        return container

    def test_bind_mount_construction(
        self, manager: ContainerManager, mock_docker_client: MagicMock, tmp_path: Path
    ) -> None:
        self._setup_container_mock(mock_docker_client)
        manager.run("image:tag", tmp_path)

        call_kwargs = mock_docker_client.container.create.call_args
        volumes = call_kwargs.kwargs.get("volumes") or call_kwargs[1].get("volumes")
        assert (str(tmp_path), "/workspace", "rw") in volumes

    def test_gpu_spec_string(
        self, manager: ContainerManager, mock_docker_client: MagicMock, tmp_path: Path
    ) -> None:
        self._setup_container_mock(mock_docker_client)
        manager.run("image:tag", tmp_path, gpu_ids=[0, 2])

        call_kwargs = mock_docker_client.container.create.call_args
        gpus = call_kwargs.kwargs.get("gpus") or call_kwargs[1].get("gpus")
        assert gpus == "device=0,2"

    def test_no_gpu_when_none(
        self, manager: ContainerManager, mock_docker_client: MagicMock, tmp_path: Path
    ) -> None:
        self._setup_container_mock(mock_docker_client)
        manager.run("image:tag", tmp_path, gpu_ids=None)

        call_kwargs = mock_docker_client.container.create.call_args
        gpus = call_kwargs.kwargs.get("gpus") or call_kwargs[1].get("gpus")
        assert gpus is None

    def test_memory_limit_passed(
        self, manager: ContainerManager, mock_docker_client: MagicMock, tmp_path: Path
    ) -> None:
        self._setup_container_mock(mock_docker_client)
        manager.run("image:tag", tmp_path, memory_limit="16g")

        call_kwargs = mock_docker_client.container.create.call_args
        memory = call_kwargs.kwargs.get("memory") or call_kwargs[1].get("memory")
        assert memory == "16g"

    def test_returns_container_result(
        self, manager: ContainerManager, mock_docker_client: MagicMock, tmp_path: Path
    ) -> None:
        self._setup_container_mock(mock_docker_client, exit_code=0)
        result = manager.run("image:tag", tmp_path)

        assert isinstance(result, ContainerResult)
        assert result.exit_code == 0
        assert result.stdout_log == tmp_path / "logs" / "stdout.log"
        assert result.stderr_log == tmp_path / "logs" / "stderr.log"

    def test_nonzero_exit_code_returned(
        self, manager: ContainerManager, mock_docker_client: MagicMock, tmp_path: Path
    ) -> None:
        self._setup_container_mock(mock_docker_client, exit_code=137)
        result = manager.run("image:tag", tmp_path)
        assert result.exit_code == 137

    def test_container_started_and_removed(
        self, manager: ContainerManager, mock_docker_client: MagicMock, tmp_path: Path
    ) -> None:
        container = self._setup_container_mock(mock_docker_client)
        manager.run("image:tag", tmp_path)

        container.start.assert_called_once()
        container.remove.assert_called_once_with(force=True)

    def test_container_removed_on_exception(
        self, manager: ContainerManager, mock_docker_client: MagicMock, tmp_path: Path
    ) -> None:
        container = self._setup_container_mock(mock_docker_client)
        container.start.side_effect = RuntimeError("docker broke")

        with pytest.raises(RuntimeError, match="docker broke"):
            manager.run("image:tag", tmp_path)

        container.remove.assert_called_once_with(force=True)

    def test_timeout_raises_tool_timeout_error(
        self, manager: ContainerManager, mock_docker_client: MagicMock, tmp_path: Path
    ) -> None:
        container = MagicMock()
        mock_docker_client.container.create.return_value = container
        mock_docker_client.container.wait.side_effect = lambda _: time.sleep(5)

        with pytest.raises(ToolTimeoutError):
            manager.run("image:tag", tmp_path, timeout=0.1)

        container.stop.assert_called_once_with(time=10)
        container.remove.assert_called_once_with(force=True)


# ---------------------------------------------------------------------------
# list_images
# ---------------------------------------------------------------------------


class TestListImages:
    def test_filters_by_prefix(
        self, manager: ContainerManager, mock_docker_client: MagicMock
    ) -> None:
        img_match = MagicMock()
        img_match.repo_tags = ["ghcr.io/briney/autobio-alphafold:2.3.2"]
        img_match.size = 5_000_000_000
        img_match.created = datetime(2025, 3, 15)

        img_other = MagicMock()
        img_other.repo_tags = ["ubuntu:22.04"]
        img_other.size = 80_000_000
        img_other.created = datetime(2025, 1, 1)

        mock_docker_client.image.list.return_value = [img_match, img_other]

        results = manager.list_images("ghcr.io/briney/autobio-")
        assert len(results) == 1
        assert results[0].uri == "ghcr.io/briney/autobio-alphafold:2.3.2"
        assert results[0].tag == "2.3.2"
        assert results[0].size == 5_000_000_000
        assert results[0].created == datetime(2025, 3, 15)

    def test_multiple_tags_on_single_image(
        self, manager: ContainerManager, mock_docker_client: MagicMock
    ) -> None:
        img = MagicMock()
        img.repo_tags = [
            "ghcr.io/briney/autobio-esm2:2.0",
            "ghcr.io/briney/autobio-esm2:latest",
        ]
        img.size = 3_000_000_000
        img.created = datetime(2025, 6, 1)

        mock_docker_client.image.list.return_value = [img]

        results = manager.list_images("ghcr.io/briney/autobio-")
        assert len(results) == 2
        tags = {r.tag for r in results}
        assert tags == {"2.0", "latest"}

    def test_no_matches(self, manager: ContainerManager, mock_docker_client: MagicMock) -> None:
        img = MagicMock()
        img.repo_tags = ["unrelated:v1"]
        img.size = 100
        img.created = datetime(2025, 1, 1)

        mock_docker_client.image.list.return_value = [img]

        assert manager.list_images("ghcr.io/briney/autobio-") == []

    def test_empty_image_list(
        self, manager: ContainerManager, mock_docker_client: MagicMock
    ) -> None:
        mock_docker_client.image.list.return_value = []
        assert manager.list_images("ghcr.io/briney/autobio-") == []

    def test_image_without_tag_gets_latest(
        self, manager: ContainerManager, mock_docker_client: MagicMock
    ) -> None:
        img = MagicMock()
        img.repo_tags = ["ghcr.io/briney/autobio-tool"]
        img.size = 200
        img.created = datetime(2025, 1, 1)

        mock_docker_client.image.list.return_value = [img]

        results = manager.list_images("ghcr.io/briney/autobio-")
        assert len(results) == 1
        assert results[0].tag == "latest"
