"""Input/output schemas for molecular dynamics simulation tools."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - Pydantic needs at runtime
from typing import Any

from pydantic import BaseModel, Field

from autobio.schemas.base import BaseInput, BaseOutput


class SimulationInput(BaseInput):
    """Input schema for molecular dynamics simulation tools (e.g., OpenMM MD).

    Tool-specific parameters are passed via the ``extra`` dict.  Common keys:

    - ``temperature`` (float): Target temperature in Kelvin (default 300.0).
    - ``pressure`` (float): Target pressure in atm (default 1.0).
    - ``timestep`` (float): Integration timestep in femtoseconds (default 2.0).
    - ``total_time_ns`` (float): Total production simulation time in nanoseconds.
    - ``n_steps`` (int): Alternative to ``total_time_ns`` — exact step count.
    - ``reporting_interval_steps`` (int): Steps between trajectory/energy reports.
    - ``trajectory_format`` (str): Output format — ``"dcd"`` (default), ``"xtc"``, ``"pdb"``.
    - ``force_field`` (str): Force field XML (default ``"amber14-all.xml"``).
    - ``implicit_solvent`` (bool): Use implicit solvent instead of explicit (default False).
    - ``water_model`` (str): Water model — ``"tip3p"``, ``"tip4pew"``, ``"spce"``.
    - ``box_shape`` (str): ``"cubic"``, ``"dodecahedron"``, ``"truncated_octahedron"``.
    - ``box_padding`` (float): Padding around solute in nm (default 1.0).
    - ``ion_type`` (str): Counter-ion pair — ``"NaCl"`` or ``"KCl"``.
    - ``ion_concentration`` (float): Ion concentration in M (default 0.15).
    - ``restraint_set`` (str): Position restraints — ``"none"``, ``"ca"``, ``"heavy_atoms"``.
    - ``restraint_stiffness`` (float): Restraint spring constant in kJ/mol/nm^2.
    - ``platform`` (str): OpenMM platform — ``"CUDA"``, ``"OpenCL"``, ``"CPU"``.
    """

    structure_path: Path = Field(description="Path to the input structure (PDB or mmCIF).")
    sequences: dict[str, str] | None = Field(
        default=None,
        description=(
            "Optional mapping of chain ID to amino acid sequence. "
            "If provided, threads these sequences onto the input backbone."
        ),
    )


class EnergyRecord(BaseModel):
    """A single time-point in the simulation energy time series."""

    step: int = Field(description="Simulation step number.")
    time_ps: float = Field(description="Simulation time in picoseconds.")
    potential_energy_kj_mol: float = Field(description="Potential energy in kJ/mol.")
    kinetic_energy_kj_mol: float = Field(description="Kinetic energy in kJ/mol.")
    total_energy_kj_mol: float = Field(description="Total energy (potential + kinetic) in kJ/mol.")
    temperature_K: float = Field(description="Instantaneous temperature in Kelvin.")
    volume_nm3: float | None = Field(
        default=None,
        description="Box volume in nm^3 (NPT ensembles only).",
    )
    density_kg_m3: float | None = Field(
        default=None,
        description="System density in kg/m^3 (NPT ensembles only).",
    )


class SimulationSummary(BaseModel):
    """Aggregate statistics and metadata for a completed simulation."""

    n_steps_completed: int = Field(description="Total number of production steps completed.")
    total_time_ns: float = Field(description="Total production simulation time in nanoseconds.")
    initial_potential_energy_kj_mol: float = Field(
        description="Potential energy at the start of production in kJ/mol."
    )
    final_potential_energy_kj_mol: float = Field(
        description="Potential energy at the end of production in kJ/mol."
    )
    mean_temperature_K: float = Field(description="Mean temperature during production in Kelvin.")
    mean_potential_energy_kj_mol: float = Field(
        description="Mean potential energy during production in kJ/mol."
    )
    platform_used: str = Field(description="OpenMM platform used (e.g., 'CUDA', 'CPU').")
    force_field: str = Field(description="Force field XML used (e.g., 'amber14-all.xml').")
    water_model: str | None = Field(
        default=None,
        description="Water model used (e.g., 'tip3p'), None if implicit solvent.",
    )
    box_shape: str | None = Field(
        default=None,
        description="Solvent box shape (e.g., 'cubic'), None if implicit solvent.",
    )
    ion_concentration_M: float | None = Field(
        default=None,
        description="Ion concentration in M, None if implicit solvent.",
    )
    equilibration_protocol: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Summary of the equilibration protocol "
            "(e.g., {'nvt_steps': 50000, 'npt_steps': 100000})."
        ),
    )


class SimulationOutput(BaseOutput):
    """Output schema for molecular dynamics simulation tools."""

    trajectory_path: Path = Field(description="Path to the trajectory file (DCD, XTC, or PDB).")
    final_structure_path: Path | None = Field(
        default=None,
        description="Path to the final-frame PDB (protein-only, solvent stripped).",
    )
    energy_timeseries: list[EnergyRecord] = Field(
        description="Energy and thermodynamic data recorded during production."
    )
    summary: SimulationSummary = Field(
        description="Aggregate statistics and metadata for the simulation."
    )
