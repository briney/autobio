"""Tool runner package — runner registry and lookup."""

from __future__ import annotations

from typing import TYPE_CHECKING

from autobio.tools.boltz import BoltzRunner
from autobio.tools.mpnn import MPNNRunner
from autobio.tools.rfd3 import RFD3Runner

if TYPE_CHECKING:
    from autobio.core.config import AutobioConfig
    from autobio.tools.base import ToolRunner

TOOL_RUNNERS: dict[str, type[ToolRunner]] = {
    "proteinmpnn": MPNNRunner,
    "ligandmpnn": MPNNRunner,
    "rfd3": RFD3Runner,
    "boltz1": BoltzRunner,
    "boltz2": BoltzRunner,
}
"""Maps tool name to its runner class. Populated when tool modules are loaded."""


def get_runner(tool_name: str, config: AutobioConfig) -> ToolRunner:
    """Look up and instantiate a tool runner by name.

    Args:
        tool_name: Registered tool name (e.g., ``"alphafold"``).
        config: Configuration to pass to the runner.

    Returns:
        Instantiated runner ready to call ``run()``.

    Raises:
        KeyError: If no runner is registered for *tool_name*.
    """
    try:
        runner_cls = TOOL_RUNNERS[tool_name]
    except KeyError:
        available = ", ".join(sorted(TOOL_RUNNERS)) or "(none)"
        raise KeyError(
            f"No runner registered for tool {tool_name!r}. Available runners: {available}"
        ) from None
    return runner_cls(tool_name, config)
