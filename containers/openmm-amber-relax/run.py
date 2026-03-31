#!/usr/bin/env python3
"""Amber force field full relaxation protocol.

Implements a multi-phase relaxation workflow:
  1. Load structure, clean with PDBFixer, add hydrogens
  2. Set up Amber force field with solvent model
  3. Solvate (explicit) or configure implicit solvent
  4. Energy minimization
  5. Heating ramp: 10 K -> target temperature with position restraints
  6. NVT equilibration with restraints
  7. NPT equilibration with barostat and restraints
  8. Short production dynamics without restraints
  9. Strip solvent, write relaxed PDB and energy data

Uses modern OpenMM API (no deprecated simtk imports).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import openmm
from openmm import unit
from openmm.app import (
    HBonds,
    NoCutoff,
    PME,
    PDBFile,
    Simulation,
)

from openmm_utils import (
    add_hydrogens,
    add_restraints,
    cleanup_structure,
    create_forcefield,
    get_energy_decomposition,
    solvate_system,
    strip_solvent,
    write_pdb,
)


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------


def _create_system(
    topology: openmm.app.Topology,
    forcefield: openmm.app.ForceField,
    explicit_solvent: bool,
) -> openmm.System:
    """Create an OpenMM system with appropriate nonbonded method.

    Args:
        topology: Molecular topology.
        forcefield: Configured force field.
        explicit_solvent: If True, use PME with 1.0 nm cutoff. Otherwise NoCutoff.

    Returns:
        Configured OpenMM System.
    """
    if explicit_solvent:
        system = forcefield.createSystem(
            topology,
            nonbondedMethod=PME,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=HBonds,
        )
    else:
        system = forcefield.createSystem(
            topology,
            nonbondedMethod=NoCutoff,
            constraints=HBonds,
        )
    return system


def _create_simulation(
    topology: openmm.app.Topology,
    system: openmm.System,
    positions: list[openmm.Vec3],
    temperature: float,
    timestep: float,
) -> Simulation:
    """Create a simulation with LangevinMiddleIntegrator.

    Args:
        topology: Molecular topology.
        system: OpenMM system.
        positions: Atom positions.
        temperature: Temperature in kelvin.
        timestep: Integration timestep in femtoseconds.

    Returns:
        Configured Simulation.
    """
    integrator = openmm.LangevinMiddleIntegrator(
        temperature * unit.kelvin,
        1.0 / unit.picosecond,
        timestep * unit.femtoseconds,
    )
    simulation = Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    return simulation


# ---------------------------------------------------------------------------
# Relaxation phases
# ---------------------------------------------------------------------------


def _run_heating(
    simulation: Simulation,
    start_temp: float,
    target_temp: float,
    heating_steps: int,
) -> None:
    """Ramp temperature from start_temp to target_temp over heating_steps.

    Updates the integrator temperature linearly in 10 K increments.

    Args:
        simulation: Active simulation.
        start_temp: Starting temperature in kelvin.
        target_temp: Target temperature in kelvin.
        heating_steps: Total number of steps for the heating phase.
    """
    if heating_steps <= 0:
        return

    temp_increment = 10.0  # K
    n_increments = max(1, int((target_temp - start_temp) / temp_increment))
    steps_per_increment = max(1, heating_steps // n_increments)

    current_temp = start_temp
    for i in range(n_increments):
        current_temp = start_temp + (i + 1) * (target_temp - start_temp) / n_increments
        simulation.integrator.setTemperature(current_temp * unit.kelvin)
        simulation.step(steps_per_increment)

    # Ensure we reach the exact target temperature
    simulation.integrator.setTemperature(target_temp * unit.kelvin)

    # Run any remaining steps
    steps_done = steps_per_increment * n_increments
    if steps_done < heating_steps:
        simulation.step(heating_steps - steps_done)

    print(
        f"[openmm-amber-relax] Heating complete: "
        f"{start_temp:.0f} K -> {target_temp:.0f} K ({heating_steps} steps)"
    )


def _run_nvt(simulation: Simulation, nvt_steps: int) -> None:
    """Run NVT equilibration.

    Args:
        simulation: Active simulation at target temperature.
        nvt_steps: Number of NVT steps.
    """
    if nvt_steps <= 0:
        return
    simulation.step(nvt_steps)
    print(f"[openmm-amber-relax] NVT equilibration complete ({nvt_steps} steps)")


def _run_npt(
    simulation: Simulation,
    system: openmm.System,
    npt_steps: int,
    pressure: float,
    temperature: float,
) -> int:
    """Run NPT equilibration with MonteCarloBarostat.

    Args:
        simulation: Active simulation.
        system: OpenMM system (barostat will be added).
        npt_steps: Number of NPT steps.
        pressure: Pressure in atmospheres.
        temperature: Temperature in kelvin.

    Returns:
        Index of the barostat force in the system (for later removal).
    """
    if npt_steps <= 0:
        return -1

    barostat = openmm.MonteCarloBarostat(
        pressure * unit.atmospheres,
        temperature * unit.kelvin,
        25,
    )
    barostat_idx = system.addForce(barostat)
    simulation.context.reinitialize(preserveState=True)
    simulation.step(npt_steps)
    print(
        f"[openmm-amber-relax] NPT equilibration complete "
        f"({npt_steps} steps, {pressure} atm)"
    )
    return barostat_idx


# ---------------------------------------------------------------------------
# Main relaxation
# ---------------------------------------------------------------------------


def relax(workspace: Path) -> None:
    """Run full amber relaxation protocol.

    Args:
        workspace: Path to the workspace directory.
    """
    config_path = workspace / "config.json"
    config = json.loads(config_path.read_text())

    structure_path = config["structure_path"]
    out_dir = Path(config.get("out_dir", str(workspace / "outputs" / "raw")))

    # Protocol parameters
    temperature = config.get("temperature", 300.0)
    pressure = config.get("pressure", 1.0)
    timestep = config.get("timestep", 2.0)
    minimize_max_iterations = config.get("minimize_max_iterations", 0)
    heating_steps = config.get("heating_steps", 5000)
    nvt_steps = config.get("nvt_steps", 25000)
    npt_steps = config.get("npt_steps", 25000)
    production_steps = config.get("production_steps", 50000)
    restraint_set = config.get("restraint_set", "heavy_atoms")
    restraint_stiffness = config.get("restraint_stiffness", 10.0)
    implicit_solvent = config.get("implicit_solvent", False)
    force_field_name = config.get("force_field", "amber14-all.xml")
    water_model = config.get("water_model", "tip3p")
    explicit_solvent = not implicit_solvent

    phases_completed: list[str] = []

    # --- Phase 1: Structure preparation ---
    print("[openmm-amber-relax] Phase 1: Structure preparation")
    raw_topology, raw_positions = cleanup_structure(structure_path)

    forcefield = create_forcefield(
        force_field_name,
        implicit_solvent=implicit_solvent,
        water_model=water_model if explicit_solvent else None,
    )

    modeller = add_hydrogens(raw_topology, raw_positions, forcefield)

    if explicit_solvent:
        modeller = solvate_system(modeller, forcefield, config)

    topology = modeller.topology
    positions = modeller.positions
    phases_completed.append("preparation")

    # --- Phase 2: Energy minimization ---
    print("[openmm-amber-relax] Phase 2: Energy minimization")
    system = _create_system(topology, forcefield, explicit_solvent)

    # Add restraints for minimization
    if restraint_set != "none":
        add_restraints(
            system, topology, positions, restraint_set, restraint_stiffness
        )

    simulation = _create_simulation(topology, system, positions, 10.0, timestep)

    # Record initial energy
    initial_state = simulation.context.getState(getEnergy=True)
    initial_energy = initial_state.getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )
    print(f"[openmm-amber-relax] Initial energy: {initial_energy:.1f} kJ/mol")

    simulation.minimizeEnergy(maxIterations=minimize_max_iterations)

    min_state = simulation.context.getState(getEnergy=True)
    min_energy = min_state.getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )
    print(f"[openmm-amber-relax] Post-minimization energy: {min_energy:.1f} kJ/mol")
    phases_completed.append("minimization")

    # --- Phase 3: Heating ramp ---
    print("[openmm-amber-relax] Phase 3: Heating ramp")
    _run_heating(simulation, start_temp=10.0, target_temp=temperature, heating_steps=heating_steps)
    phases_completed.append("heating")

    # --- Phase 4: NVT equilibration ---
    print("[openmm-amber-relax] Phase 4: NVT equilibration")
    _run_nvt(simulation, nvt_steps)
    phases_completed.append("nvt_equilibration")

    # --- Phase 5: NPT equilibration ---
    print("[openmm-amber-relax] Phase 5: NPT equilibration")
    barostat_idx = _run_npt(simulation, system, npt_steps, pressure, temperature)
    if npt_steps > 0:
        phases_completed.append("npt_equilibration")

    # --- Phase 6: Production dynamics (no restraints) ---
    print("[openmm-amber-relax] Phase 6: Production dynamics")
    if production_steps > 0:
        # Remove restraints by setting stiffness to 0
        for i in range(system.getNumForces()):
            force = system.getForce(i)
            if isinstance(force, openmm.CustomExternalForce):
                force.setGlobalParameterDefaultValue(0, 0.0)
        simulation.context.reinitialize(preserveState=True)

        simulation.step(production_steps)
        print(
            f"[openmm-amber-relax] Production complete ({production_steps} steps)"
        )
        phases_completed.append("production")

    # --- Collect results ---
    final_state = simulation.context.getState(
        getPositions=True, getEnergy=True
    )
    final_positions = final_state.getPositions(asNumpy=False)
    final_energy = final_state.getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )

    # Remove barostat before energy decomposition (avoids spurious force group)
    if barostat_idx >= 0:
        system.removeForce(barostat_idx)
        simulation.context.reinitialize(preserveState=True)

    energy_terms = get_energy_decomposition(simulation, system)

    # Strip solvent for output
    if explicit_solvent:
        protein_topology, protein_positions = strip_solvent(topology, final_positions)
    else:
        protein_topology, protein_positions = topology, final_positions

    # Write outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_out_path = out_dir / "relaxed.pdb"
    write_pdb(protein_topology, protein_positions, str(pdb_out_path))

    energy_data: dict[str, Any] = {
        "initial_energy_kj_mol": initial_energy,
        "final_energy_kj_mol": final_energy,
        "energy_terms": energy_terms,
        "protocol": {
            "phases_completed": phases_completed,
            "temperature_kelvin": temperature,
            "pressure_atm": pressure,
            "timestep_fs": timestep,
            "heating_steps": heating_steps,
            "nvt_steps": nvt_steps,
            "npt_steps": npt_steps,
            "production_steps": production_steps,
            "force_field": force_field_name,
            "solvent": "implicit" if implicit_solvent else f"explicit ({water_model})",
            "restraint_set": restraint_set,
        },
    }
    (out_dir / "energy.json").write_text(json.dumps(energy_data, indent=2))

    print(
        f"[openmm-amber-relax] Complete: "
        f"{initial_energy:.1f} -> {final_energy:.1f} kJ/mol "
        f"({len(phases_completed)} phases completed)"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <workspace_path>", file=sys.stderr)
        sys.exit(1)
    relax(Path(sys.argv[1]))
