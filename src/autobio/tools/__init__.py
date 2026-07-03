"""Tool runner package — runner registry and lookup."""

from __future__ import annotations

from typing import TYPE_CHECKING

from autobio.tools.antibody_lm import AntibodyLMRunner
from autobio.tools.antifold import AntiFoldRunner
from autobio.tools.antipasti import AntipastiRunner
from autobio.tools.baddg import BAddGRunner
from autobio.tools.boltz import BoltzRunner
from autobio.tools.chai import ChaiRunner
from autobio.tools.complexa import ComplexaRunner
from autobio.tools.esm import ESMRunner
from autobio.tools.esm_if1 import ESMIF1Runner
from autobio.tools.esmfold import ESMFoldRunner
from autobio.tools.evoef2 import EvoEF2Runner
from autobio.tools.freesasa import FreeSASARunner
from autobio.tools.ligandmpnn_packer import LigandMPNNPackerRunner
from autobio.tools.mpnn import MPNNRunner
from autobio.tools.openfold3 import OpenFold3Runner
from autobio.tools.openmm import OpenMMRunner
from autobio.tools.prodigy import ProdigyRunner
from autobio.tools.rfd3 import RFD3Runner
from autobio.tools.rosetta import RosettaRunner
from autobio.tools.stabddg import StaBddGRunner

if TYPE_CHECKING:
    from autobio.core.config import AutobioConfig
    from autobio.tools.base import ToolRunner

TOOL_RUNNERS: dict[str, type[ToolRunner]] = {
    "antifold": AntiFoldRunner,
    "antipasti": AntipastiRunner,
    "currab": AntibodyLMRunner,
    "ft_esm": AntibodyLMRunner,
    "balm_paired": AntibodyLMRunner,
    "balm_unpaired": AntibodyLMRunner,
    "ablang2": AntibodyLMRunner,
    "antiberta2": AntibodyLMRunner,
    "esm1b": ESMRunner,
    "esm2": ESMRunner,
    "esm_if1": ESMIF1Runner,
    "esmfold": ESMFoldRunner,
    "proteinmpnn": MPNNRunner,
    "ligandmpnn": MPNNRunner,
    "rfd3": RFD3Runner,
    "complexa": ComplexaRunner,
    "boltz1": BoltzRunner,
    "boltz2": BoltzRunner,
    "chai1": ChaiRunner,
    "openfold3": OpenFold3Runner,
    "rosetta": RosettaRunner,
    "prodigy": ProdigyRunner,
    "openmm": OpenMMRunner,
    "stabddg": StaBddGRunner,
    "baddg": BAddGRunner,
    "evoef2": EvoEF2Runner,
    "freesasa": FreeSASARunner,
    "ligandmpnn_build_mutant": LigandMPNNPackerRunner,
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
