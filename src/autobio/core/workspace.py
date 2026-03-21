"""Workspace directory lifecycle management."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from autobio.core.result import RunResult


class Workspace:
    """Manages the standardised workspace directory mounted into containers.

    A workspace has the layout::

        <root>/
            config.json
            inputs/
            outputs/raw/
            outputs/standardized/
            logs/
            result.json
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.config_path = root / "config.json"
        self.inputs_dir = root / "inputs"
        self.raw_output_dir = root / "outputs" / "raw"
        self.std_output_dir = root / "outputs" / "standardized"
        self.logs_dir = root / "logs"
        self.result_path = root / "result.json"
        self._is_temp = False

    @classmethod
    def create(cls, output_dir: Path | None = None) -> Workspace:
        """Create a workspace with the required subdirectories.

        Args:
            output_dir: Explicit root directory.  When *None* a temporary
                directory is created and flagged for cleanup.
        """
        if output_dir is not None:
            root = output_dir
            is_temp = False
        else:
            root = Path(tempfile.mkdtemp(prefix="autobio-"))
            is_temp = True

        for subdir in ("inputs", "outputs/raw", "outputs/standardized", "logs"):
            (root / subdir).mkdir(parents=True, exist_ok=True)

        ws = cls(root)
        ws._is_temp = is_temp
        return ws

    def write_config(self, config: dict[str, object]) -> None:
        """Serialise *config* as JSON to ``config.json``."""
        self.config_path.write_text(json.dumps(config, indent=2))

    def write_input_file(self, filename: str, content: str | bytes) -> Path:
        """Write an input file into the ``inputs/`` directory.

        Args:
            filename: Name of the file to create.
            content: Text or binary content.

        Returns:
            Path to the written file.
        """
        dest = self.inputs_dir / filename
        if isinstance(content, bytes):
            dest.write_bytes(content)
        else:
            dest.write_text(content)
        return dest

    def read_result(self) -> RunResult:
        """Parse ``result.json`` into a :class:`RunResult`."""
        data = json.loads(self.result_path.read_text())
        return RunResult.model_validate(data)

    def cleanup(self) -> None:
        """Remove the workspace if it was auto-created as a temp directory."""
        if self._is_temp and self.root.exists():
            shutil.rmtree(self.root)
