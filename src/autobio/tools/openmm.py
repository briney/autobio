"""OpenMM tool runners — energy minimization and future MD tools.

All OpenMM tools share a single ``OpenMMRunner`` class that dispatches by
``tool_name`` using the ``_VARIANT_CONFIG`` dict.  Each tool maps to a
different Docker image (thin layer on top of the shared
``autobio-openmm-base`` image).

OpenMM tools are **CPU-only** for minimization. Future MD tools may require GPU.

Supported tools:

- ``openmm_amber_minimize`` — Amber force field energy minimization with
  iterative violation checking (AlphaFold-style).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from autobio.core.registry import TOOL_REGISTRY, ToolCategory, ToolEntry
from autobio.core.result import AutobioError
from autobio.schemas.base import BaseInput  # noqa: TC001 - needed at runtime for isinstance
from autobio.schemas.scoring import ScoredStructure, ScoringInput, ScoringOutput
from autobio.tools.base import ToolRunner

if TYPE_CHECKING:
    from autobio.core.workspace import Workspace

# ---------------------------------------------------------------------------
# Variant configuration — maps tool name to protocol-specific settings
# ---------------------------------------------------------------------------

_VARIANT_CONFIG: dict[str, dict[str, Any]] = {
    "openmm_amber_minimize": {
        "protocol": "amber_minimize",
        "produces_structure": True,
        "default_force_field": "amber14-all.xml",
        "default_tolerance": 2.39,
        "default_max_iterations": 0,
        "default_restraint_set": "none",
        "default_restraint_stiffness": 10.0,
        "default_implicit_solvent": True,
        "default_max_outer_iterations": 20,
    },
}

# Keys in ``extra`` that are consumed by the runner and should NOT be
# flat-merged into config.json.
_CONSUMED_EXTRA_KEYS = frozenset(
    {
        "force_field",
        "tolerance",
        "max_iterations",
        "restraint_set",
        "restraint_stiffness",
        "implicit_solvent",
        "max_outer_iterations",
        "violation_tolerance",
    }
)

# Allowed values for validated extra keys
_ALLOWED_FORCE_FIELDS = frozenset({"amber14-all.xml", "amber99sb.xml", "charmm36.xml"})
_ALLOWED_RESTRAINT_SETS = frozenset({"none", "ca", "heavy_atoms"})


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class OpenMMRunner(ToolRunner):
    """Runner for OpenMM molecular simulation tools.

    ``prepare_workspace`` validates inputs on the host side, copies the input
    structure into the workspace, and writes ``config.json``.

    ``parse_output`` reads the standardised ``result_data.json`` produced by
    the container's ``standardize.py`` script.
    """

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json and copy input structure into the workspace."""
        assert isinstance(input_data, ScoringInput)

        # -- Host-side validation (fail fast before container launch) --------
        self._validate_inputs(input_data)

        # -- Resolve variant config -----------------------------------------
        variant_cfg = _VARIANT_CONFIG[self.tool_name]

        # -- Copy input structure into workspace ----------------------------
        src_path = input_data.structure_path
        dest_name = src_path.name
        shutil.copy2(src_path, workspace.inputs_dir / dest_name)
        container_structure_path = f"/workspace/inputs/{dest_name}"

        # -- Build config.json ----------------------------------------------
        config: dict[str, Any] = {
            "protocol": variant_cfg["protocol"],
            "structure_path": container_structure_path,
            "out_dir": "/workspace/outputs/raw",
        }

        # Parameters with defaults from variant config
        config["force_field"] = input_data.extra.get(
            "force_field", variant_cfg["default_force_field"]
        )
        config["tolerance"] = input_data.extra.get("tolerance", variant_cfg["default_tolerance"])
        config["max_iterations"] = input_data.extra.get(
            "max_iterations", variant_cfg["default_max_iterations"]
        )
        config["restraint_set"] = input_data.extra.get(
            "restraint_set", variant_cfg["default_restraint_set"]
        )
        config["restraint_stiffness"] = input_data.extra.get(
            "restraint_stiffness", variant_cfg["default_restraint_stiffness"]
        )
        config["implicit_solvent"] = input_data.extra.get(
            "implicit_solvent", variant_cfg["default_implicit_solvent"]
        )
        config["max_outer_iterations"] = input_data.extra.get(
            "max_outer_iterations", variant_cfg["default_max_outer_iterations"]
        )

        # Flat-merge extra dict (excluding consumed keys)
        for key, value in input_data.extra.items():
            if key not in _CONSUMED_EXTRA_KEYS:
                config[key] = value

        workspace.write_config(config)

    def parse_output(self, workspace: Workspace) -> ScoringOutput:
        """Read standardised outputs and return a ``ScoringOutput``."""
        result_data_path = workspace.std_output_dir / "result_data.json"
        data = json.loads(result_data_path.read_text())

        scores = []
        for s in data["scores"]:
            structure_path = None
            if s.get("structure_path"):
                structure_path = self._resolve_container_path(s["structure_path"], workspace)

            scores.append(
                ScoredStructure(
                    total_score=s["total_score"],
                    per_residue_scores=s.get("per_residue_scores"),
                    score_breakdown=s.get("score_breakdown"),
                    units=s.get("units"),
                    structure_path=structure_path,
                    ddg=s.get("ddg"),
                    mutations=s.get("mutations"),
                )
            )

        # Placeholder metadata — overwritten by base class run()
        return ScoringOutput(
            scores=scores,
            metadata=self._build_metadata(workspace, 0.0, [], ""),
            raw_output_path=workspace.raw_output_dir,
        )

    # -- Private helpers ----------------------------------------------------

    @staticmethod
    def _resolve_container_path(container_path_str: str, workspace: Workspace) -> Path:
        """Map a container-internal ``/workspace/...`` path to the host workspace."""
        container_path = Path(container_path_str)
        try:
            relative = container_path.relative_to("/workspace")
        except ValueError:
            return container_path
        return workspace.root / relative

    def _validate_inputs(self, input_data: ScoringInput) -> None:
        """Host-side validation — catch errors before container launch."""
        if not input_data.structure_path.exists():
            raise AutobioError(f"Input structure file does not exist: {input_data.structure_path}")

        # Validate force field if specified
        force_field = input_data.extra.get("force_field")
        if force_field is not None and force_field not in _ALLOWED_FORCE_FIELDS:
            raise AutobioError(
                f"Invalid force_field {force_field!r}. "
                f"Allowed: {', '.join(sorted(_ALLOWED_FORCE_FIELDS))}"
            )

        # Validate restraint set if specified
        restraint_set = input_data.extra.get("restraint_set")
        if restraint_set is not None and restraint_set not in _ALLOWED_RESTRAINT_SETS:
            raise AutobioError(
                f"Invalid restraint_set {restraint_set!r}. "
                f"Allowed: {', '.join(sorted(_ALLOWED_RESTRAINT_SETS))}"
            )


# ---------------------------------------------------------------------------
# Registry entries — populated when this module is imported
# ---------------------------------------------------------------------------

_AMBER_MINIMIZE_NOTES = (
    "Energy minimization using the Amber14 force field (amber14-all.xml) "
    "with iterative violation checking (AlphaFold-style). Minimizes energy, "
    "checks for steric clashes and bond violations, excludes violating "
    "residues from restraints, and re-minimizes until clean.",
    "Default tolerance is 2.39 kJ/mol/nm (matching AlphaFold's amber "
    "relaxation). Override with extra['tolerance']. Set extra['max_iterations'] "
    "to limit minimization steps per round (0 = unlimited).",
    "Implicit solvent (OBC2) is enabled by default, matching AlphaFold-style "
    "relaxation. Disable with extra['implicit_solvent'] = False for vacuum.",
    "Optional harmonic restraints on CA atoms or heavy atoms via "
    "extra['restraint_set'] ('ca' or 'heavy_atoms'). Restraint stiffness "
    "defaults to 10.0 kJ/mol/nm^2, override with extra['restraint_stiffness'].",
    "Output energy is in kJ/mol (physical units). The score_breakdown "
    "contains per-force-type energies (HarmonicBondForce, NonbondedForce, etc.), "
    "initial_energy for comparison, and violation-checking metadata.",
    "Force field alternatives: extra['force_field'] accepts 'amber99sb.xml' "
    "(legacy, used by AlphaFold) or 'charmm36.xml'.",
)

_AMBER_MINIMIZE_INPUT_FORMAT = (
    "Provide a PDB file via structure_path. The structure should have "
    "standard amino acid residues. Missing hydrogens are added "
    "automatically by OpenMM's Modeller. Non-standard residues or "
    "ligands are not supported in the current Amber force field workflow.",
)

TOOL_REGISTRY["openmm_amber_minimize"] = ToolEntry(
    image_tag="openmm-amber-minimize:1.0.0",
    category=ToolCategory.SCORING,
    requires_gpu=False,
    gpu_count=0,
    input_schema=ScoringInput,
    output_schema=ScoringOutput,
    default_timeout=600,
    supports_batch=False,
    description=(
        "Minimize a protein structure using OpenMM with the Amber force "
        "field and iterative violation checking (AlphaFold-style). "
        "Resolves steric clashes and refines geometry through repeated "
        "minimization rounds, excluding violating residues from restraints "
        "at each iteration. Reports final energy in kJ/mol with a "
        "per-force-type breakdown. Produces a refined PDB structure."
    ),
    version="1.0.0",
    notes=_AMBER_MINIMIZE_NOTES,
    input_format=_AMBER_MINIMIZE_INPUT_FORMAT,
)
