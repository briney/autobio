"""Docker container lifecycle management via python-on-whales."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

from python_on_whales import DockerClient
from python_on_whales.exceptions import DockerException

from autobio.core.result import ContainerNotFoundError, ContainerResult, ToolTimeoutError

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

    from python_on_whales import Container

    from autobio.core.config import AutobioConfig

logger = logging.getLogger(__name__)


@dataclass
class ImageInfo:
    """Summary of a locally cached Docker image."""

    uri: str
    tag: str
    size: int
    created: datetime


class ContainerManager:
    """Manages Docker container lifecycle for tool execution.

    This is the sole interface to the Docker SDK.  No other module in the
    codebase imports Docker libraries directly.
    """

    def __init__(self, config: AutobioConfig) -> None:
        self._client = DockerClient(host=config.docker_host)
        self._config = config

    def ensure_image(self, uri: str) -> None:
        """Pull the image if it is not available locally.

        Args:
            uri: Full image URI (e.g. ``ghcr.io/briney/autobio-tool:1.0``).

        Raises:
            ContainerNotFoundError: If the pull fails.
        """
        if self._client.image.exists(uri):
            return
        self.pull_image(uri)

    def pull_image(self, uri: str) -> None:
        """Pull an image from the registry.

        Args:
            uri: Full image URI.

        Raises:
            ContainerNotFoundError: If the pull fails.
        """
        try:
            self._client.image.pull(uri)
        except DockerException as exc:
            raise ContainerNotFoundError(f"Failed to pull image {uri!r}: {exc}") from exc

    def run(
        self,
        image_uri: str,
        workspace: Path,
        gpu_ids: list[int] | None = None,
        timeout: int | None = None,
        memory_limit: str | None = None,
    ) -> ContainerResult:
        """Run a container with *workspace* bind-mounted at ``/workspace``.

        The container is always removed after execution, regardless of outcome.

        Args:
            image_uri: Full image URI.
            workspace: Host path to bind-mount as ``/workspace``.
            gpu_ids: GPU device IDs to expose, or *None* for no GPUs.
            timeout: Maximum wall-clock seconds.  Raises
                :class:`ToolTimeoutError` if exceeded.
            memory_limit: Docker memory limit string (e.g. ``"8g"``).

        Returns:
            A :class:`ContainerResult` with exit code and log file paths.

        Raises:
            ToolTimeoutError: If *timeout* is exceeded.
        """
        volumes = [(str(workspace), "/workspace", "rw")]
        gpus: str | None = None
        if gpu_ids:
            gpus = f"device={','.join(str(g) for g in gpu_ids)}"

        container = self._client.container.create(
            image_uri,
            volumes=volumes,
            gpus=gpus,
            memory=memory_limit,
        )
        try:
            container.start()
            exit_code = self._wait_for_exit(container, timeout)
        except ToolTimeoutError:
            try:
                container.stop(time=10)
            except Exception:
                logger.warning("Failed to stop timed-out container", exc_info=True)
            raise
        finally:
            try:
                container.remove(force=True)
            except Exception:
                logger.warning("Failed to remove container", exc_info=True)

        stdout_log = workspace / "logs" / "stdout.log"
        stderr_log = workspace / "logs" / "stderr.log"
        return ContainerResult(
            exit_code=exit_code,
            stdout_log=stdout_log,
            stderr_log=stderr_log,
        )

    def list_images(self, prefix: str) -> list[ImageInfo]:
        """List locally cached images whose repository starts with *prefix*.

        Args:
            prefix: Image name prefix to filter on
                (e.g. ``"ghcr.io/briney/autobio-"``).

        Returns:
            A list of :class:`ImageInfo` for matching images.
        """
        images = self._client.image.list()
        result: list[ImageInfo] = []
        for img in images:
            for tag in img.repo_tags:
                if tag.startswith(prefix):
                    result.append(
                        ImageInfo(
                            uri=tag,
                            tag=tag.split(":")[-1] if ":" in tag else "latest",
                            size=img.size,
                            created=img.created,
                        )
                    )
        return result

    def _wait_for_exit(self, container: Container, timeout: int | None) -> int:
        """Block until *container* exits or *timeout* seconds elapse.

        Args:
            container: A started Docker container.
            timeout: Seconds to wait, or *None* for no limit.

        Returns:
            The container exit code.

        Raises:
            ToolTimeoutError: If the container does not exit within *timeout*.
        """
        wait_result: dict[str, object] = {"exit_code": None, "error": None}

        def _do_wait() -> None:
            try:
                wait_result["exit_code"] = self._client.container.wait(container)
            except Exception as exc:
                wait_result["error"] = exc

        thread = threading.Thread(target=_do_wait, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            raise ToolTimeoutError(f"Container exceeded {timeout}s timeout")

        if wait_result["error"] is not None:
            raise wait_result["error"]  # type: ignore[misc]

        return wait_result["exit_code"]  # type: ignore[return-value]
