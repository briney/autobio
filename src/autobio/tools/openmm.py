"""OpenMM tool — energy minimization, relaxation, and MD simulation.

A single catalog Tool, ``openmm``, exposes three Modes sharing the
``OpenMMRunner`` runner class, dispatching by ``self.current_mode.name``
using the ``_VARIANT_CONFIG`` dict. Each mode maps to a different Docker
image (thin layer on top of the shared ``autobio-openmm-base`` image).

Modes:

- ``amber_minimize`` (default) — Amber force field energy minimization with
  iterative violation checking (AlphaFold-style).
- ``amber_relax`` — Full relaxation with explicit solvent (default),
  including solvation, heating, NVT/NPT equilibration, and short production.
- ``md_simulate`` — Production molecular dynamics simulation with
  trajectory and energy time series output.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from autobio.core.catalog import Mode, Tool, register
from autobio.core.registry import ToolCategory
from autobio.core.result import AutobioError
from autobio.schemas.base import BaseInput, BaseOutput  # noqa: TC001 - needed at runtime
from autobio.schemas.scoring import ScoredStructure, ScoringInput, ScoringOutput
from autobio.schemas.simulation import (
    EnergyRecord,
    SimulationInput,
    SimulationOutput,
    SimulationSummary,
)
from autobio.tools.base import ToolRunner

if TYPE_CHECKING:
    from autobio.core.workspace import Workspace

# ---------------------------------------------------------------------------
# Variant configuration — maps tool name to protocol-specific settings
# ---------------------------------------------------------------------------

_VARIANT_CONFIG: dict[str, dict[str, Any]] = {
    "amber_minimize": {
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
    "amber_relax": {
        "protocol": "amber_relax",
        "produces_structure": True,
        "default_force_field": "amber14-all.xml",
        "default_implicit_solvent": False,
        "default_water_model": "tip3p",
        "default_box_shape": "cubic",
        "default_box_padding": 1.0,
        "default_ion_type": "NaCl",
        "default_ion_concentration": 0.15,
        "default_temperature": 300.0,
        "default_pressure": 1.0,
        "default_restraint_set": "heavy_atoms",
        "default_restraint_stiffness": 10.0,
        "default_timestep": 2.0,
        "default_minimize_max_iterations": 0,
        "default_heating_steps": 25000,
        "default_nvt_steps": 25000,
        "default_npt_steps": 50000,
        "default_production_steps": 25000,
    },
    "md_simulate": {
        "protocol": "md_simulate",
        "produces_structure": True,
        "produces_trajectory": True,
        "default_force_field": "amber14-all.xml",
        "default_implicit_solvent": False,
        "default_water_model": "tip3p",
        "default_box_shape": "cubic",
        "default_box_padding": 1.0,
        "default_ion_type": "NaCl",
        "default_ion_concentration": 0.15,
        "default_temperature": 300.0,
        "default_pressure": 1.0,
        "default_timestep": 2.0,
        "default_total_time_ns": 10.0,
        "default_reporting_interval_steps": 5000,
        "default_trajectory_format": "dcd",
        "default_restraint_set": "none",
        "default_restraint_stiffness": 10.0,
        "default_minimize_max_iterations": 0,
        "default_equilibration_nvt_steps": 50000,
        "default_equilibration_npt_steps": 100000,
        "default_platform": "CUDA",
    },
}

# Keys in ``extra`` that are consumed by the runner and should NOT be
# flat-merged into config.json.
_CONSUMED_EXTRA_KEYS = frozenset(
    {
        # Shared
        "force_field",
        "restraint_set",
        "restraint_stiffness",
        "implicit_solvent",
        # Minimize-specific
        "tolerance",
        "max_iterations",
        "max_outer_iterations",
        "violation_tolerance",
        # Solvation (relax + md_simulate)
        "water_model",
        "box_shape",
        "box_padding",
        "ion_type",
        "ion_concentration",
        # Dynamics (relax + md_simulate)
        "temperature",
        "pressure",
        "timestep",
        "minimize_max_iterations",
        # Relax-specific
        "heating_steps",
        "nvt_steps",
        "npt_steps",
        "production_steps",
        # MD-specific
        "total_time_ns",
        "n_steps",
        "reporting_interval_steps",
        "trajectory_format",
        "equilibration_nvt_steps",
        "equilibration_npt_steps",
        "platform",
    }
)

# Allowed values for validated extra keys
_ALLOWED_FORCE_FIELDS = frozenset({"amber14-all.xml", "amber99sb.xml", "charmm36.xml"})
_ALLOWED_RESTRAINT_SETS = frozenset({"none", "ca", "heavy_atoms"})
_ALLOWED_WATER_MODELS = frozenset({"tip3p", "tip4pew", "spce"})
_ALLOWED_BOX_SHAPES = frozenset({"cubic", "dodecahedron", "truncated_octahedron"})
_ALLOWED_ION_TYPES = frozenset({"NaCl", "KCl"})
_ALLOWED_TRAJECTORY_FORMATS = frozenset({"dcd", "xtc", "pdb"})


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class OpenMMRunner(ToolRunner):
    """Runner for the OpenMM amber_minimize/amber_relax/md_simulate modes.

    Dispatches by ``self.current_mode.name`` to support minimize, relax, and
    MD simulate modes. Scoring-type modes (minimize, relax) use
    :class:`ScoringInput`/:class:`ScoringOutput`. The MD simulation mode uses
    :class:`SimulationInput`/:class:`SimulationOutput`.
    """

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json and copy input structure into the workspace."""
        assert self.current_mode is not None
        variant_cfg = _VARIANT_CONFIG[self.current_mode.name]

        if variant_cfg.get("produces_trajectory"):
            assert isinstance(input_data, SimulationInput)
            self._validate_simulation_inputs(input_data)
            self._prepare_simulation_workspace(input_data, workspace, variant_cfg)
        else:
            assert isinstance(input_data, ScoringInput)
            self._validate_scoring_inputs(input_data)
            self._prepare_scoring_workspace(input_data, workspace, variant_cfg)

    def parse_output(self, workspace: Workspace) -> BaseOutput:
        """Read standardised outputs and return the appropriate output model."""
        assert self.current_mode is not None
        variant_cfg = _VARIANT_CONFIG[self.current_mode.name]

        if variant_cfg.get("produces_trajectory"):
            return self._parse_simulation_output(workspace)
        return self._parse_scoring_output(workspace)

    # -- Scoring workspace (minimize + relax) --------------------------------

    def _prepare_scoring_workspace(
        self,
        input_data: ScoringInput,
        workspace: Workspace,
        variant_cfg: dict[str, Any],
    ) -> None:
        """Build config.json for scoring-type tools (minimize, relax)."""
        # Copy input structure into workspace
        src_path = input_data.structure_path
        dest_name = src_path.name
        shutil.copy2(src_path, workspace.inputs_dir / dest_name)
        container_structure_path = f"/workspace/inputs/{dest_name}"

        # Base config
        config: dict[str, Any] = {
            "protocol": variant_cfg["protocol"],
            "structure_path": container_structure_path,
            "out_dir": "/workspace/outputs/raw",
        }

        # Extract all variant defaults, overridden by extra dict
        for key, default in variant_cfg.items():
            if key.startswith("default_"):
                param_name = key.removeprefix("default_")
                config[param_name] = input_data.extra.get(param_name, default)

        # Flat-merge extra dict (excluding consumed keys)
        for key, value in input_data.extra.items():
            if key not in _CONSUMED_EXTRA_KEYS:
                config[key] = value

        workspace.write_config(config)

    def _parse_scoring_output(self, workspace: Workspace) -> ScoringOutput:
        """Read standardised scoring output."""
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

        return ScoringOutput(
            scores=scores,
            metadata=self._build_metadata(workspace, 0.0, [], ""),
            raw_output_path=workspace.raw_output_dir,
        )

    # -- Simulation workspace (md_simulate) ----------------------------------

    def _prepare_simulation_workspace(
        self,
        input_data: SimulationInput,
        workspace: Workspace,
        variant_cfg: dict[str, Any],
    ) -> None:
        """Build config.json for simulation-type tools (md_simulate)."""
        # Copy input structure into workspace
        src_path = input_data.structure_path
        dest_name = src_path.name
        shutil.copy2(src_path, workspace.inputs_dir / dest_name)
        container_structure_path = f"/workspace/inputs/{dest_name}"

        # Base config
        config: dict[str, Any] = {
            "protocol": variant_cfg["protocol"],
            "structure_path": container_structure_path,
            "out_dir": "/workspace/outputs/raw",
        }

        # Extract all variant defaults, overridden by extra dict
        for key, default in variant_cfg.items():
            if key.startswith("default_"):
                param_name = key.removeprefix("default_")
                config[param_name] = input_data.extra.get(param_name, default)

        # Allow n_steps as alternative to total_time_ns
        if "n_steps" in input_data.extra:
            config["n_steps"] = input_data.extra["n_steps"]

        # Flat-merge extra dict (excluding consumed keys)
        for key, value in input_data.extra.items():
            if key not in _CONSUMED_EXTRA_KEYS:
                config[key] = value

        workspace.write_config(config)

    def _parse_simulation_output(self, workspace: Workspace) -> SimulationOutput:
        """Read standardised simulation output."""
        result_data_path = workspace.std_output_dir / "result_data.json"
        data = json.loads(result_data_path.read_text())

        # Parse trajectory path
        trajectory_path = self._resolve_container_path(data["trajectory_path"], workspace)

        # Parse final structure path
        final_structure_path = None
        if data.get("final_structure_path"):
            final_structure_path = self._resolve_container_path(
                data["final_structure_path"], workspace
            )

        # Parse energy timeseries
        energy_timeseries = [EnergyRecord(**record) for record in data["energy_timeseries"]]

        # Parse summary
        summary = SimulationSummary(**data["summary"])

        return SimulationOutput(
            trajectory_path=trajectory_path,
            final_structure_path=final_structure_path,
            energy_timeseries=energy_timeseries,
            summary=summary,
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

    def _validate_scoring_inputs(self, input_data: ScoringInput) -> None:
        """Host-side validation for scoring tools (minimize, relax)."""
        if not input_data.structure_path.exists():
            raise AutobioError(f"Input structure file does not exist: {input_data.structure_path}")

        self._validate_common_enums(input_data.extra)
        self._validate_solvation_params(input_data.extra)

    def _validate_simulation_inputs(self, input_data: SimulationInput) -> None:
        """Host-side validation for simulation tools (md_simulate)."""
        if not input_data.structure_path.exists():
            raise AutobioError(f"Input structure file does not exist: {input_data.structure_path}")

        self._validate_common_enums(input_data.extra)
        self._validate_solvation_params(input_data.extra)

        # Simulation-specific validations
        total_time_ns = input_data.extra.get("total_time_ns")
        if total_time_ns is not None and total_time_ns <= 0:
            raise AutobioError(f"total_time_ns must be positive, got {total_time_ns}")

        n_steps = input_data.extra.get("n_steps")
        if n_steps is not None and n_steps <= 0:
            raise AutobioError(f"n_steps must be positive, got {n_steps}")

        reporting_interval = input_data.extra.get("reporting_interval_steps")
        if reporting_interval is not None and reporting_interval <= 0:
            raise AutobioError(
                f"reporting_interval_steps must be positive, got {reporting_interval}"
            )

        trajectory_format = input_data.extra.get("trajectory_format")
        if trajectory_format is not None and trajectory_format not in _ALLOWED_TRAJECTORY_FORMATS:
            raise AutobioError(
                f"Invalid trajectory_format {trajectory_format!r}. "
                f"Allowed: {', '.join(sorted(_ALLOWED_TRAJECTORY_FORMATS))}"
            )

    @staticmethod
    def _validate_common_enums(extra: dict[str, Any]) -> None:
        """Validate enum-type parameters shared across tools."""
        force_field = extra.get("force_field")
        if force_field is not None and force_field not in _ALLOWED_FORCE_FIELDS:
            raise AutobioError(
                f"Invalid force_field {force_field!r}. "
                f"Allowed: {', '.join(sorted(_ALLOWED_FORCE_FIELDS))}"
            )

        restraint_set = extra.get("restraint_set")
        if restraint_set is not None and restraint_set not in _ALLOWED_RESTRAINT_SETS:
            raise AutobioError(
                f"Invalid restraint_set {restraint_set!r}. "
                f"Allowed: {', '.join(sorted(_ALLOWED_RESTRAINT_SETS))}"
            )

    @staticmethod
    def _validate_solvation_params(extra: dict[str, Any]) -> None:
        """Validate solvation-related parameters."""
        water_model = extra.get("water_model")
        if water_model is not None and water_model not in _ALLOWED_WATER_MODELS:
            raise AutobioError(
                f"Invalid water_model {water_model!r}. "
                f"Allowed: {', '.join(sorted(_ALLOWED_WATER_MODELS))}"
            )

        box_shape = extra.get("box_shape")
        if box_shape is not None and box_shape not in _ALLOWED_BOX_SHAPES:
            raise AutobioError(
                f"Invalid box_shape {box_shape!r}. "
                f"Allowed: {', '.join(sorted(_ALLOWED_BOX_SHAPES))}"
            )

        ion_type = extra.get("ion_type")
        if ion_type is not None and ion_type not in _ALLOWED_ION_TYPES:
            raise AutobioError(
                f"Invalid ion_type {ion_type!r}. Allowed: {', '.join(sorted(_ALLOWED_ION_TYPES))}"
            )

        temperature = extra.get("temperature")
        if temperature is not None and temperature <= 0:
            raise AutobioError(f"temperature must be positive, got {temperature}")

        pressure = extra.get("pressure")
        if pressure is not None and pressure <= 0:
            raise AutobioError(f"pressure must be positive, got {pressure}")

        box_padding = extra.get("box_padding")
        if box_padding is not None and box_padding <= 0:
            raise AutobioError(f"box_padding must be positive, got {box_padding}")

        ion_concentration = extra.get("ion_concentration")
        if ion_concentration is not None and ion_concentration < 0:
            raise AutobioError(f"ion_concentration must be non-negative, got {ion_concentration}")

        timestep = extra.get("timestep")
        if timestep is not None and (timestep < 0.5 or timestep > 4.0):
            raise AutobioError(f"timestep must be between 0.5 and 4.0 fs, got {timestep}")


# ---------------------------------------------------------------------------
# Catalog registration — populated when this module is imported
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
    "GPU (CUDA) is required. OpenMM 8.5 runs the energy minimizer entirely "
    "on GPU for dramatically faster performance.",
)

_AMBER_MINIMIZE_INPUT_FORMAT = (
    "Provide a PDB file via structure_path. The structure should have "
    "standard amino acid residues. Missing hydrogens are added "
    "automatically by OpenMM's Modeller. Non-standard residues or "
    "ligands are not supported in the current Amber force field workflow.",
)

_MINIMIZE_MODE_NOTES = _AMBER_MINIMIZE_NOTES + _AMBER_MINIMIZE_INPUT_FORMAT

# ---------------------------------------------------------------------------

_AMBER_RELAX_NOTES = (
    "Full relaxation protocol with explicit solvent (default) or implicit "
    "solvent. Includes energy minimization, gradual heating, NVT and NPT "
    "equilibration, and a short production run. Returns a refined, solvent-"
    "stripped protein structure with energy in kJ/mol.",
    "Default solvent: explicit TIP3P water in a cubic box with 1.0 nm "
    "padding and 0.15 M NaCl. Override with extra['water_model'] "
    "('tip3p', 'tip4pew', 'spce'), extra['box_shape'] ('cubic', "
    "'dodecahedron', 'truncated_octahedron'), extra['box_padding'], "
    "extra['ion_type'] ('NaCl', 'KCl'), extra['ion_concentration'].",
    "For faster processing, set extra['implicit_solvent'] = True to use "
    "OBC2 implicit solvent instead of explicit solvation.",
    "Default temperature is 300 K and pressure is 1 atm. Override with "
    "extra['temperature'] and extra['pressure']. Heating ramp goes from "
    "10 K to target temperature over extra['heating_steps'] (default 25000).",
    "Heavy-atom restraints (10.0 kJ/mol/nm^2) are applied during heating "
    "and NVT equilibration, then gradually released during NPT. Override "
    "with extra['restraint_set'] and extra['restraint_stiffness'].",
    "Protocol step counts: extra['heating_steps'] (50 ps), "
    "extra['nvt_steps'] (50 ps), extra['npt_steps'] (100 ps), "
    "extra['production_steps'] (50 ps). All at 2 fs timestep by default.",
    "GPU (CUDA) is required. OpenMM 8.5 runs energy minimization and "
    "dynamics entirely on GPU for dramatically faster performance.",
)

_AMBER_RELAX_INPUT_FORMAT = (
    "Provide a PDB file via structure_path. The structure should have "
    "standard amino acid residues. Missing hydrogens, solvent, and ions "
    "are added automatically. Non-standard residues or ligands are not "
    "supported in the current Amber force field workflow.",
)

_RELAX_MODE_NOTES = _AMBER_RELAX_NOTES + _AMBER_RELAX_INPUT_FORMAT

# ---------------------------------------------------------------------------

_MD_SIMULATE_NOTES = (
    "Production molecular dynamics simulation using the Amber force field "
    "with explicit solvent. Builds a solvated system, minimizes, "
    "equilibrates (NVT + NPT), and runs production MD. Produces a DCD "
    "trajectory, energy time series, and a final protein-only PDB.",
    "Default: 10 ns production at 300 K and 1 atm with TIP3P water in "
    "a cubic box (1.0 nm padding, 0.15 M NaCl). Override via extra dict.",
    "Trajectory format: extra['trajectory_format'] — 'dcd' (default), "
    "'xtc', or 'pdb'. Reporting interval: extra['reporting_interval_steps'] "
    "(default 5000, i.e., every 10 ps at 2 fs timestep).",
    "Solvation control: extra['water_model'] ('tip3p', 'tip4pew', 'spce'), "
    "extra['box_shape'] ('cubic', 'dodecahedron', 'truncated_octahedron'), "
    "extra['box_padding'] (nm), extra['ion_type'] ('NaCl', 'KCl'), "
    "extra['ion_concentration'] (M).",
    "Simulation length: extra['total_time_ns'] (default 10.0) or "
    "extra['n_steps'] for exact step count. Timestep: extra['timestep'] "
    "(default 2.0 fs).",
    "GPU (CUDA) is required by default. The OPENMM_DEFAULT_PLATFORM "
    "environment variable is set to CUDA in the container image.",
    "Equilibration: extra['equilibration_nvt_steps'] (default 50000) and "
    "extra['equilibration_npt_steps'] (default 100000).",
)

_MD_SIMULATE_INPUT_FORMAT = (
    "Provide a PDB file via structure_path. Ideally a pre-relaxed "
    "structure (e.g., from openmm_amber_relax), but the tool handles "
    "raw structures with PDBFixer cleanup and full solvation. Non-standard "
    "residues or ligands are not supported.",
)

_SIMULATE_MODE_NOTES = _MD_SIMULATE_NOTES + _MD_SIMULATE_INPUT_FORMAT

OPENMM_TOOL = Tool(
    name="openmm",
    display_name="OpenMM",
    category=ToolCategory.SCORING,
    description=(
        "OpenMM molecular mechanics engine (Amber force field). Modes: "
        "amber_minimize (AlphaFold-style energy minimization), amber_relax "
        "(full relaxation with explicit solvent), and md_simulate (production "
        "molecular dynamics with trajectory output)."
    ),
    version="1.1.0",
    image_tag="openmm-amber-minimize:1.1.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="amber_minimize",
    modes={
        "amber_minimize": Mode(
            name="amber_minimize",
            display_name="Amber minimize",
            description=(
                "Amber force-field energy minimization with iterative violation "
                "checking (AlphaFold-style). Produces a refined PDB and energy in kJ/mol."
            ),
            input_schema=ScoringInput,
            output_schema=ScoringOutput,
            default_timeout=600,
            image_tag="openmm-amber-minimize:1.1.0",
            notes=_MINIMIZE_MODE_NOTES,
        ),
        "amber_relax": Mode(
            name="amber_relax",
            display_name="Amber relax",
            description=(
                "Full relaxation with explicit solvent (default): solvation, "
                "minimization, heating, NVT/NPT equilibration, and short production. "
                "Returns a refined, solvent-stripped structure with energy in kJ/mol."
            ),
            input_schema=ScoringInput,
            output_schema=ScoringOutput,
            default_timeout=3600,
            image_tag="openmm-amber-relax:1.1.0",
            notes=_RELAX_MODE_NOTES,
        ),
        "md_simulate": Mode(
            name="md_simulate",
            display_name="MD simulate",
            description=(
                "Production molecular dynamics with the Amber force field and "
                "explicit solvent. Produces a trajectory (DCD/XTC/PDB), energy "
                "time series, and a final protein-only PDB."
            ),
            input_schema=SimulationInput,
            output_schema=SimulationOutput,
            default_timeout=86400,
            image_tag="openmm-md-simulate:1.1.0",
            category=ToolCategory.SIMULATION,
            notes=_SIMULATE_MODE_NOTES,
        ),
    },
    keywords=(
        "openmm",
        "molecular dynamics",
        "md",
        "minimize",
        "relax",
        "simulation",
        "amber",
        "force field",
        "energy",
    ),
)
"""Catalog Tool for OpenMM (amber_minimize + amber_relax + md_simulate modes)."""

register(OPENMM_TOOL)
