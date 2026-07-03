"""ToolRunner abstract base class — execution lifecycle for all tool runners."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from autobio.core.catalog import CATALOG, Mode, Tool, get_tool
from autobio.core.container import ContainerManager
from autobio.core.gpu import GPUManager
from autobio.core.result import AutobioError, ToolExecutionError
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
        if tool_name not in CATALOG:
            available = ", ".join(sorted(CATALOG)) or "(none)"
            raise KeyError(f"Unknown tool {tool_name!r}. Available tools: {available}")
        self.tool: Tool = get_tool(tool_name)
        self.tool_name = tool_name
        self.config = config
        self.current_mode: Mode | None = None
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
        mode: str | None = None,
    ) -> BaseOutput:
        """Full execution lifecycle. Do NOT override this method.

        *mode* selects a :class:`Mode` by name (defaulting to the Tool's
        ``default_mode``).

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
            mode: Name of the :class:`Mode` to run. Defaults to the Tool's
                ``default_mode``.

        Returns:
            Populated output model with metadata attached.

        Raises:
            ToolExecutionError: If the container reports a failure.
            AutobioError: For an unknown mode.
        """
        gpu_ids: list[int] = []
        workspace: Workspace | None = None
        start = time.monotonic()

        self.current_mode = self._resolve_mode(mode)

        try:
            # 1. Create workspace
            workspace = Workspace.create(output_dir)

            # 2. Prepare workspace (subclass hook)
            self.prepare_workspace(input_data, workspace)

            # 3. Resolve GPUs
            gpu_ids = self._resolve_gpu(gpu)

            # 4. Ensure image
            image_uri = f"{self.config.image_prefix}{self._image_tag()}"
            self._container.ensure_image(image_uri)

            # 5. Run container
            effective_timeout = timeout if timeout is not None else self._default_timeout()
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

    def _resolve_mode(self, mode: str | None) -> Mode:
        """Resolve the selected Mode by name, defaulting to the tool's default mode."""
        name = mode if mode is not None else self.tool.default_mode
        try:
            return self.tool.modes[name]
        except KeyError:
            available = ", ".join(sorted(self.tool.modes))
            raise AutobioError(
                f"Unknown mode {name!r} for tool {self.tool_name!r}. Available modes: {available}"
            ) from None

    def _apply_extra(self, config: dict[str, Any], input_data: BaseInput) -> None:
        """Merge ``input_data.extra`` into *config*, rejecting key collisions.

        ``extra`` is the escape hatch for parameters not promoted to typed fields
        on a mode's input schema. A key in ``extra`` is rejected fail-fast (rather
        than silently applied via ``config.update``) if it either:

        - names a typed field on the active mode's input schema, or
        - already exists in *config* — i.e. it collides with a runner-derived
          config key (e.g. ``output_dir``, ``pdb_path``) written earlier in
          ``prepare_workspace``.

        Args:
            config: The mapping being assembled for ``config.json``, as it stands
                at call time (i.e. with all runner-derived keys already present);
                mutated in place with the accepted ``extra`` keys.
            input_data: The validated input whose ``extra`` dict is merged.

        Raises:
            AutobioError: If ``extra`` contains a key that collides with a typed
                field on the active mode's input schema or with a runner-derived
                key already present in *config*.
        """
        assert self.current_mode is not None
        typed_fields = set(self.current_mode.input_schema.model_fields) - {"extra"}
        collisions = sorted(key for key in input_data.extra if key in typed_fields or key in config)
        if collisions:
            raise AutobioError(
                "extra must not contain keys that collide with typed input fields or "
                f"runner-derived config keys: {', '.join(collisions)}. Pass tool-specific "
                "parameters under new keys; set typed parameters as top-level input fields."
            )
        config.update(input_data.extra)

    def _image_tag(self) -> str:
        """Container image tag for the current run (mode override, else tool default)."""
        assert self.current_mode is not None
        return self.current_mode.image_tag or self.tool.image_tag

    def _default_timeout(self) -> int:
        """Default timeout for the current run (per-mode)."""
        assert self.current_mode is not None
        return self.current_mode.default_timeout

    def _requires_gpu(self) -> bool:
        """Whether the active tool requires a GPU."""
        return self.tool.requires_gpu

    def _gpu_count(self) -> int:
        """Number of GPUs the active tool requests under ``gpu='auto'``."""
        return self.tool.gpu_count

    def _tool_version(self) -> str:
        """Version string of the active tool."""
        return self.tool.version

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
            if not self._requires_gpu():
                return []
            return self._gpu.allocate(count=self._gpu_count())

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
            mode=self.current_mode.name if self.current_mode is not None else None,
            tool_version=self._tool_version(),
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
