"""ToolRunner abstract base class — execution lifecycle for all tool runners."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from autobio.core.container import ContainerManager
from autobio.core.gpu import GPUManager
from autobio.core.registry import TOOL_REGISTRY, ToolEntry
from autobio.core.result import ToolExecutionError
from autobio.core.workspace import Workspace
from autobio.schemas.base import BaseInput, BaseOutput, RunMetadata
from autobio.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from autobio.core.config import AutobioConfig

logger = get_logger("tools.base")


class ToolRunner(ABC):
    """Abstract base for all tool runners.

    Subclasses implement ``prepare_workspace`` and ``parse_output`` only.
    The concrete ``run`` method orchestrates the full execution lifecycle:
    workspace creation, input preparation, GPU allocation, container launch,
    result collection, output parsing, and cleanup.
    """

    def __init__(self, tool_name: str, config: AutobioConfig) -> None:
        try:
            self.entry: ToolEntry = TOOL_REGISTRY[tool_name]
        except KeyError:
            available = ", ".join(sorted(TOOL_REGISTRY)) or "(none)"
            raise KeyError(f"Unknown tool {tool_name!r}. Available tools: {available}") from None
        self.tool_name = tool_name
        self.config = config
        self._container = ContainerManager(config)
        self._gpu = GPUManager()

    @abstractmethod
    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json and input files to the workspace.

        Translates the standardized input schema into the tool-specific
        configuration and file layout expected by the container's ``run.sh``.

        Args:
            input_data: Validated input conforming to the tool's category schema.
            workspace: Initialised workspace with directories created.
        """
        ...

    @abstractmethod
    def parse_output(self, workspace: Workspace) -> BaseOutput:
        """Read standardised outputs from the workspace into a Pydantic model.

        The container's ``standardize.sh`` has already coerced raw outputs into
        the schema format. This method reads ``outputs/standardized/`` and
        returns a populated output model.

        Args:
            workspace: Workspace after successful container execution.

        Returns:
            Populated output model (e.g., ``StructurePredictionOutput``).
        """
        ...

    def run(
        self,
        input_data: BaseInput,
        gpu: str | list[int] = "auto",
        timeout: int | None = None,
        output_dir: Path | None = None,
    ) -> BaseOutput:
        """Full execution lifecycle. Do NOT override this method.

        Steps:
            1. Create workspace
            2. Prepare workspace (subclass hook)
            3. Resolve GPU allocation
            4. Ensure container image is available
            5. Run container
            6. Read result.json and check status
            7. Parse output (subclass hook)
            8. Attach metadata to output
            9. Return output

        Finally: release GPUs, clean up temp workspace.

        Args:
            input_data: Validated input for this tool.
            gpu: GPU specification — ``"auto"``, ``"none"``, a list of device
                IDs, or a comma-separated string of IDs.
            timeout: Maximum wall-clock seconds. Falls back to the tool's
                ``default_timeout`` if *None*.
            output_dir: Persist workspace here instead of a temp directory.

        Returns:
            Populated output model with metadata attached.

        Raises:
            ToolExecutionError: If the container reports a failure.
        """
        gpu_ids: list[int] = []
        workspace: Workspace | None = None
        start = time.monotonic()

        try:
            # 1. Create workspace
            workspace = Workspace.create(output_dir)

            # 2. Prepare workspace (subclass hook)
            self.prepare_workspace(input_data, workspace)

            # 3. Resolve GPUs
            gpu_ids = self._resolve_gpu(gpu)

            # 4. Ensure image
            image_uri = f"{self.config.image_prefix}{self.entry.image_tag}"
            self._container.ensure_image(image_uri)

            # 5. Run container
            effective_timeout = timeout if timeout is not None else self.entry.default_timeout
            self._container.run(
                image_uri=image_uri,
                workspace=workspace.root,
                gpu_ids=gpu_ids or None,
                timeout=effective_timeout,
            )

            # 6. Read result.json and check status
            wall_time = time.monotonic() - start
            run_result = workspace.read_result()

            if run_result.status != "success":
                logs = self._read_logs(workspace)
                raise ToolExecutionError(
                    phase=run_result.phase,
                    exit_code=run_result.exit_code,
                    error_message=run_result.error_message or "Unknown error",
                    logs=logs,
                    wall_time=wall_time,
                )

            # 7. Parse output (subclass hook)
            output = self.parse_output(workspace)

            # 8. Attach metadata
            output.metadata = self._build_metadata(workspace, wall_time, gpu_ids, image_uri)

            return output

        finally:
            # Always release GPUs
            if gpu_ids:
                self._gpu.release(gpu_ids)

            # Clean up temp workspace (user-specified dirs are preserved)
            if workspace is not None and workspace._is_temp:
                workspace.cleanup()

    def _resolve_gpu(self, gpu: str | list[int]) -> list[int]:
        """Translate the user-facing gpu parameter into a list of device IDs.

        Args:
            gpu: One of:
                - ``"auto"`` — allocate the tool's default ``gpu_count``
                  (0 if the tool doesn't require a GPU).
                - ``"none"`` — no GPUs.
                - A ``list[int]`` — specific device IDs.
                - A comma-separated string of IDs (e.g., ``"0,1"``).

        Returns:
            Sorted list of device IDs (may be empty).
        """
        if isinstance(gpu, list):
            if not gpu:
                return []
            return self._gpu.allocate(device_ids=gpu)

        if gpu == "none":
            return []

        if gpu == "auto":
            if not self.entry.requires_gpu:
                return []
            return self._gpu.allocate(count=self.entry.gpu_count)

        # Comma-separated string: "0,1,2"
        device_ids = [int(x.strip()) for x in gpu.split(",")]
        return self._gpu.allocate(device_ids=device_ids)

    def _build_metadata(
        self,
        workspace: Workspace,
        wall_time: float,
        gpu_ids: list[int],
        image_uri: str,
    ) -> RunMetadata:
        """Construct a ``RunMetadata`` instance for the completed run."""
        return RunMetadata(
            tool_name=self.tool_name,
            tool_version=self.entry.version,
            image_uri=image_uri,
            wall_time_seconds=wall_time,
            gpu_ids=gpu_ids or None,
            workspace_path=workspace.root,
            timestamp=datetime.now(tz=UTC),
        )

    @staticmethod
    def _read_logs(workspace: Workspace) -> str:
        """Read stderr log from workspace, returning empty string on failure."""
        stderr_log = workspace.logs_dir / "stderr.log"
        try:
            return stderr_log.read_text()
        except OSError:
            return ""
