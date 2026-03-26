#!/usr/bin/env python3
"""Molecular dynamics simulation with OpenMM.

Implements a full MD protocol:
  1. Clean structure and add hydrogens
  2. Set up force field with explicit or implicit solvent
  3. Solvate (if explicit solvent) and add ions
  4. Energy minimization
  5. NVT equilibration (velocity generation at target temperature)
  6. NPT equilibration (pressure coupling)
  7. Production MD with trajectory and energy reporting
  8. Strip solvent and write final frame

Uses modern OpenMM API (no deprecated simtk imports).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import openmm
from openmm import unit
from openmm.app import (
    DCDReporter,
    HBonds,
    NoCutoff,
    PDBFile,
    PDBReporter,
    PME,
    Simulation,
    StateDataReporter,
)

from openmm_utils import (
    add_hydrogens,
    add_restraints,
    cleanup_structure,
    create_forcefield,
    solvate_system,
    strip_solvent,
    write_pdb,
)


# ---------------------------------------------------------------------------
# State data CSV parsing
# ---------------------------------------------------------------------------


def _parse_state_data_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Parse a StateDataReporter CSV file into a list of energy records.

    Args:
        csv_path: Path to the CSV file written by StateDataReporter.

    Returns:
        List of dicts with keys matching EnergyRecord field names.
    """
    records: list[dict[str, Any]] = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            record: dict[str, Any] = {
                "step": int(row['#"Step"']),
                "time_ps": float(row["Time (ps)"]),
                "potential_energy_kj_mol": float(row["Potential Energy (kJ/mole)"]),
                "kinetic_energy_kj_mol": float(row["Kinetic Energy (kJ/mole)"]),
                "total_energy_kj_mol": float(row["Total Energy (kJ/mole)"]),
                "temperature_K": float(row["Temperature (K)"]),
            }
            # Volume and density are only present for periodic systems
            if "Box Volume (nm^3)" in row:
                record["volume_nm3"] = float(row["Box Volume (nm^3)"])
            if "Density (g/mL)" in row:
                # Convert g/mL to kg/m^3
                record["density_kg_m3"] = float(row["Density (g/mL)"]) * 1000.0
            records.append(record)
    return records


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


def simulate(workspace: Path) -> None:
    """Run a full MD simulation protocol.

    Args:
        workspace: Path to the workspace directory.
    """
    config_path = workspace / "config.json"
    config = json.loads(config_path.read_text())

    structure_path = config["structure_path"]
    out_dir = Path(config.get("out_dir", str(workspace / "outputs" / "raw")))
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Configuration defaults ─────────────────────────────────────────
    force_field_name = config.get("force_field", "amber14-all.xml")
    implicit_solvent = config.get("implicit_solvent", False)
    water_model = config.get("water_model", "tip3p")
    temperature = config.get("temperature", 300.0)
    pressure = config.get("pressure", 1.0)
    timestep_fs = config.get("timestep", 2.0)
    total_time_ns = config.get("total_time_ns", None)
    n_steps = config.get("n_steps", None)
    reporting_interval = config.get("reporting_interval_steps", 1000)
    trajectory_format = config.get("trajectory_format", "dcd")
    minimize_max_iterations = config.get("minimize_max_iterations", 1000)
    equilibration_nvt_steps = config.get("equilibration_nvt_steps", 50000)
    equilibration_npt_steps = config.get("equilibration_npt_steps", 100000)
    restraint_set = config.get("restraint_set", "none")
    restraint_stiffness = config.get("restraint_stiffness", 10.0)

    timestep = timestep_fs * unit.femtoseconds

    # ── Step 1: Clean structure ────────────────────────────────────────
    print("[openmm-md-simulate] Cleaning structure...")
    topology, positions = cleanup_structure(structure_path)

    # ── Step 2: Set up force field ─────────────────────────────────────
    print("[openmm-md-simulate] Creating force field...")
    forcefield = create_forcefield(
        force_field_name,
        implicit_solvent=implicit_solvent,
        water_model=None if implicit_solvent else water_model,
    )

    # ── Step 3: Add hydrogens ──────────────────────────────────────────
    print("[openmm-md-simulate] Adding hydrogens...")
    modeller = add_hydrogens(topology, positions, forcefield)

    # ── Step 4: Solvate (explicit solvent only) ────────────────────────
    if not implicit_solvent:
        print("[openmm-md-simulate] Solvating system...")
        modeller = solvate_system(modeller, forcefield, config)

    topology = modeller.topology
    positions = modeller.positions

    # ── Step 5: Create system ──────────────────────────────────────────
    print("[openmm-md-simulate] Creating system...")
    if implicit_solvent:
        system = forcefield.createSystem(
            topology,
            nonbondedMethod=NoCutoff,
            constraints=HBonds,
        )
    else:
        system = forcefield.createSystem(
            topology,
            nonbondedMethod=PME,
            nonbondedCutoff=1.0 * unit.nanometer,
            constraints=HBonds,
        )

    # Add restraints if requested
    if restraint_set != "none":
        add_restraints(
            system, topology, positions, restraint_set, restraint_stiffness
        )

    # ── Step 6: Energy minimization ────────────────────────────────────
    print("[openmm-md-simulate] Minimizing energy...")
    integrator = openmm.LangevinMiddleIntegrator(
        temperature * unit.kelvin,
        1.0 / unit.picosecond,
        timestep,
    )
    simulation = Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy(maxIterations=minimize_max_iterations)

    minimized_state = simulation.context.getState(
        getPositions=True, getEnergy=True
    )
    positions = minimized_state.getPositions(asNumpy=False)
    minimized_energy = minimized_state.getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )
    print(
        f"[openmm-md-simulate] Minimized energy: {minimized_energy:.1f} kJ/mol"
    )

    # ── Step 7: NVT equilibration ──────────────────────────────────────
    if equilibration_nvt_steps > 0:
        print(
            f"[openmm-md-simulate] NVT equilibration: "
            f"{equilibration_nvt_steps} steps..."
        )
        simulation.context.setVelocitiesToTemperature(
            temperature * unit.kelvin
        )
        simulation.step(equilibration_nvt_steps)

    # ── Step 8: NPT equilibration ──────────────────────────────────────
    if equilibration_npt_steps > 0 and not implicit_solvent:
        print(
            f"[openmm-md-simulate] NPT equilibration: "
            f"{equilibration_npt_steps} steps..."
        )
        system.addForce(
            openmm.MonteCarloBarostat(
                pressure * unit.atmospheres,
                temperature * unit.kelvin,
            )
        )
        simulation.context.reinitialize(preserveState=True)
        simulation.step(equilibration_npt_steps)

    # ── Step 9: Production MD ──────────────────────────────────────────
    # Determine step count
    if n_steps is None and total_time_ns is not None:
        n_steps = int(total_time_ns * 1e6 / timestep_fs)
    elif n_steps is None:
        raise ValueError(
            "Either 'n_steps' or 'total_time_ns' must be specified in config."
        )

    # Compute actual total time
    actual_total_time_ns = (n_steps * timestep_fs) / 1e6

    print(
        f"[openmm-md-simulate] Production MD: {n_steps} steps "
        f"({actual_total_time_ns:.3f} ns)..."
    )

    # Add trajectory reporter
    trajectory_ext = trajectory_format if trajectory_format != "pdb" else "pdb"
    trajectory_path = out_dir / f"trajectory.{trajectory_ext}"
    if trajectory_format == "pdb":
        simulation.reporters.append(
            PDBReporter(str(trajectory_path), reporting_interval)
        )
    else:
        simulation.reporters.append(
            DCDReporter(str(trajectory_path), reporting_interval)
        )

    # Add state data reporter (energy, temperature, etc.)
    state_data_path = out_dir / "state_data.csv"
    report_volume = not implicit_solvent
    simulation.reporters.append(
        StateDataReporter(
            str(state_data_path),
            reporting_interval,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=report_volume,
            density=report_volume,
        )
    )

    # Get initial production energy
    initial_state = simulation.context.getState(getEnergy=True)
    initial_production_energy = initial_state.getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )

    # Run production
    simulation.step(n_steps)

    # Get platform used
    platform_used = simulation.context.getPlatform().getName()

    # ── Step 10: Final state ───────────────────────────────────────────
    final_state = simulation.context.getState(
        getPositions=True, getEnergy=True
    )
    final_positions = final_state.getPositions(asNumpy=False)
    final_energy = final_state.getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )

    # ── Step 11: Strip solvent and write final frame ───────────────────
    print("[openmm-md-simulate] Writing final frame...")
    protein_topology, protein_positions = strip_solvent(
        topology, final_positions
    )
    final_pdb_path = out_dir / "final.pdb"
    write_pdb(protein_topology, protein_positions, str(final_pdb_path))

    # ── Step 12: Parse energy timeseries ───────────────────────────────
    print("[openmm-md-simulate] Parsing energy timeseries...")
    energy_records = _parse_state_data_csv(state_data_path)
    (out_dir / "energy_timeseries.json").write_text(
        json.dumps(energy_records, indent=2)
    )

    # Compute summary statistics from timeseries
    temperatures = [r["temperature_K"] for r in energy_records]
    potential_energies = [r["potential_energy_kj_mol"] for r in energy_records]
    mean_temperature = sum(temperatures) / len(temperatures) if temperatures else 0.0
    mean_potential_energy = (
        sum(potential_energies) / len(potential_energies)
        if potential_energies
        else 0.0
    )

    # ── Step 13: Write simulation summary ──────────────────────────────
    summary = {
        "n_steps_completed": n_steps,
        "total_time_ns": actual_total_time_ns,
        "initial_potential_energy_kj_mol": initial_production_energy,
        "final_potential_energy_kj_mol": final_energy,
        "mean_temperature_K": mean_temperature,
        "mean_potential_energy_kj_mol": mean_potential_energy,
        "platform_used": platform_used,
        "force_field": force_field_name,
        "water_model": None if implicit_solvent else water_model,
        "box_shape": None if implicit_solvent else config.get("box_shape", "cubic"),
        "ion_concentration_M": (
            None if implicit_solvent else config.get("ion_concentration", 0.15)
        ),
        "equilibration_protocol": {
            "nvt_steps": equilibration_nvt_steps,
            "npt_steps": equilibration_npt_steps if not implicit_solvent else 0,
        },
    }
    (out_dir / "simulation_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(
        f"[openmm-md-simulate] Complete: {n_steps} steps, "
        f"{actual_total_time_ns:.3f} ns, "
        f"final energy {final_energy:.1f} kJ/mol, "
        f"platform {platform_used}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <workspace_path>", file=sys.stderr)
        sys.exit(1)
    simulate(Path(sys.argv[1]))
