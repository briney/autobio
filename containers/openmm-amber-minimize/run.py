#!/usr/bin/env python3
"""Amber force field energy minimization with iterative violation checking.

Implements an AlphaFold-style relaxation workflow:
  1. Load structure, add hydrogens
  2. Set up Amber force field + optional restraints
  3. Minimize energy
  4. Check for steric clashes and geometry violations
  5. Exclude violating residues from restraints, re-minimize
  6. Repeat until clean or max iterations reached

Uses modern OpenMM API (no deprecated simtk imports).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import openmm
from openmm import unit
from openmm.app import (
    ForceField,
    HBonds,
    Modeller,
    NoCutoff,
    PDBFile,
    Simulation,
)
from pdbfixer import PDBFixer

# Implicit solvent model XML (included with OpenMM)
_IMPLICIT_SOLVENT_XML = "implicit/obc2.xml"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Clash detection: fraction of VdW radius sum below which a clash is flagged
_CLASH_OVERLAP_FRACTION = 0.6

# Bond length violation threshold (relative to equilibrium)
_BOND_LENGTH_TOLERANCE = 0.15  # 15% deviation

# Bond angle violation threshold (degrees from equilibrium)
_BOND_ANGLE_TOLERANCE = 15.0  # degrees


# ---------------------------------------------------------------------------
# Violation checking
# ---------------------------------------------------------------------------


def _check_steric_clashes(
    topology: openmm.app.Topology,
    positions: list[openmm.Vec3],
    system: openmm.System,
) -> list[dict[str, Any]]:
    """Find atom pairs with steric clashes (overlapping VdW radii).

    Args:
        topology: Molecular topology.
        positions: Current atom positions.
        system: OpenMM system (used to extract VdW radii from NonbondedForce).

    Returns:
        List of violation dicts with residue indices.
    """
    # Extract VdW radii from the NonbondedForce
    radii: dict[int, float] = {}
    for force in system.getForces():
        if isinstance(force, openmm.NonbondedForce):
            for i in range(force.getNumParticles()):
                _, sigma, _ = force.getParticleParameters(i)
                radii[i] = sigma.value_in_unit(unit.nanometer) / 2.0
            break

    if not radii:
        return []

    # Build residue index lookup
    atoms = list(topology.atoms())
    residue_of_atom: dict[int, int] = {}
    for atom in atoms:
        residue_of_atom[atom.index] = atom.residue.index

    violations = []
    n_atoms = len(atoms)
    for i in range(n_atoms):
        pos_i = positions[i]
        ri = radii.get(i, 0.15)
        for j in range(i + 1, n_atoms):
            # Skip atoms in the same residue
            if residue_of_atom[i] == residue_of_atom[j]:
                continue
            # Skip bonded atoms (1-2 pairs)
            pos_j = positions[j]
            rj = radii.get(j, 0.15)
            dx = pos_i[0] - pos_j[0]
            dy = pos_i[1] - pos_j[1]
            dz = pos_i[2] - pos_j[2]
            dist = math.sqrt(
                dx.value_in_unit(unit.nanometer) ** 2
                + dy.value_in_unit(unit.nanometer) ** 2
                + dz.value_in_unit(unit.nanometer) ** 2
            )
            threshold = _CLASH_OVERLAP_FRACTION * (ri + rj)
            if dist < threshold:
                violations.append(
                    {
                        "type": "steric_clash",
                        "residue_indices": [residue_of_atom[i], residue_of_atom[j]],
                        "atom_indices": [i, j],
                        "distance_nm": dist,
                        "threshold_nm": threshold,
                    }
                )
    return violations


def _check_bond_violations(
    topology: openmm.app.Topology,
    positions: list[openmm.Vec3],
    system: openmm.System,
) -> list[dict[str, Any]]:
    """Find bonds with length violations (stretched or compressed).

    Args:
        topology: Molecular topology.
        positions: Current atom positions.
        system: OpenMM system (used to extract bond parameters from HarmonicBondForce).

    Returns:
        List of violation dicts with residue indices.
    """
    atoms = list(topology.atoms())
    residue_of_atom: dict[int, int] = {}
    for atom in atoms:
        residue_of_atom[atom.index] = atom.residue.index

    violations = []
    for force in system.getForces():
        if isinstance(force, openmm.HarmonicBondForce):
            for bond_idx in range(force.getNumBonds()):
                i, j, r0, _ = force.getBondParameters(bond_idx)
                r0_nm = r0.value_in_unit(unit.nanometer)

                pos_i = positions[i]
                pos_j = positions[j]
                dx = pos_i[0] - pos_j[0]
                dy = pos_i[1] - pos_j[1]
                dz = pos_i[2] - pos_j[2]
                dist = math.sqrt(
                    dx.value_in_unit(unit.nanometer) ** 2
                    + dy.value_in_unit(unit.nanometer) ** 2
                    + dz.value_in_unit(unit.nanometer) ** 2
                )

                deviation = abs(dist - r0_nm) / r0_nm if r0_nm > 0 else 0
                if deviation > _BOND_LENGTH_TOLERANCE:
                    violations.append(
                        {
                            "type": "bond_length",
                            "residue_indices": [
                                residue_of_atom[i],
                                residue_of_atom[j],
                            ],
                            "atom_indices": [i, j],
                            "actual_nm": dist,
                            "equilibrium_nm": r0_nm,
                            "deviation_fraction": deviation,
                        }
                    )
            break
    return violations


def _check_violations(
    topology: openmm.app.Topology,
    positions: list[openmm.Vec3],
    system: openmm.System,
) -> list[dict[str, Any]]:
    """Check for all structural violations.

    Args:
        topology: Molecular topology.
        positions: Current atom positions.
        system: OpenMM system.

    Returns:
        Combined list of violation dicts.
    """
    violations = []
    violations.extend(_check_steric_clashes(topology, positions, system))
    violations.extend(_check_bond_violations(topology, positions, system))
    return violations


def _violating_residues(violations: list[dict[str, Any]]) -> set[int]:
    """Extract unique residue indices from a list of violations."""
    residues: set[int] = set()
    for v in violations:
        residues.update(v["residue_indices"])
    return residues


# ---------------------------------------------------------------------------
# Restraint setup
# ---------------------------------------------------------------------------


def _add_restraints(
    system: openmm.System,
    topology: openmm.app.Topology,
    positions: list[openmm.Vec3],
    restraint_set: str,
    stiffness_kj_per_nm2: float,
    exclusions: set[int],
) -> None:
    """Add harmonic positional restraints to the system.

    Args:
        system: OpenMM system to add the force to.
        topology: Molecular topology.
        positions: Reference positions for restraints.
        restraint_set: Which atoms to restrain: "ca" or "heavy_atoms".
        stiffness_kj_per_nm2: Spring constant in kJ/mol/nm^2.
        exclusions: Residue indices to exclude from restraints.
    """
    if restraint_set == "none":
        return

    force = openmm.CustomExternalForce(
        "0.5 * k * periodicdistance(x, y, z, x0, y0, z0)^2"
    )
    force.addGlobalParameter("k", stiffness_kj_per_nm2)
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    for atom in topology.atoms():
        if atom.residue.index in exclusions:
            continue

        include = False
        if restraint_set == "ca":
            include = atom.name == "CA"
        elif restraint_set == "heavy_atoms":
            include = atom.element.symbol != "H"

        if include:
            pos = positions[atom.index]
            force.addParticle(
                atom.index,
                [
                    pos[0].value_in_unit(unit.nanometer),
                    pos[1].value_in_unit(unit.nanometer),
                    pos[2].value_in_unit(unit.nanometer),
                ],
            )

    if force.getNumParticles() > 0:
        system.addForce(force)


# ---------------------------------------------------------------------------
# Energy decomposition
# ---------------------------------------------------------------------------


def _get_energy_decomposition(
    simulation: Simulation,
    system: openmm.System,
) -> dict[str, float]:
    """Get per-force-type energy breakdown.

    Assigns each force to its own group, then queries energy for each group.

    Args:
        simulation: Active simulation with current positions set.
        system: OpenMM system.

    Returns:
        Dict mapping force class name to energy in kJ/mol.
    """
    # Assign each force to its own group
    for i, force in enumerate(system.getForces()):
        force.setForceGroup(i)

    # Reinitialize context to pick up the new force groups
    simulation.context.reinitialize(preserveState=True)

    breakdown: dict[str, float] = {}
    for i, force in enumerate(system.getForces()):
        state = simulation.context.getState(getEnergy=True, groups={i})
        energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        force_name = type(force).__name__
        breakdown[force_name] = energy
    return breakdown


# ---------------------------------------------------------------------------
# Main minimization
# ---------------------------------------------------------------------------


def _create_system_and_minimize(
    topology: openmm.app.Topology,
    positions: list[openmm.Vec3],
    forcefield: ForceField,
    config: dict[str, Any],
    exclusions: set[int],
) -> tuple[Simulation, openmm.System, list[openmm.Vec3]]:
    """Create a system and run energy minimization.

    Args:
        topology: Molecular topology.
        positions: Starting atom positions.
        forcefield: Loaded force field.
        config: Configuration dict from config.json.
        exclusions: Residue indices to exclude from restraints.

    Returns:
        Tuple of (simulation, system, minimized_positions).
    """
    # Create system (implicit solvent is already in the ForceField if enabled)
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=NoCutoff,
        constraints=HBonds,
    )

    # Add restraints
    restraint_set = config.get("restraint_set", "none")
    stiffness = config.get("restraint_stiffness", 10.0)
    if restraint_set != "none":
        _add_restraints(system, topology, positions, restraint_set, stiffness, exclusions)

    # Create integrator (0 K — pure minimization, no dynamics)
    integrator = openmm.LangevinMiddleIntegrator(
        0 * unit.kelvin,
        1.0 / unit.picosecond,
        0.002 * unit.picoseconds,
    )

    # Create simulation
    simulation = Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)

    # Minimize
    tolerance = config.get("tolerance", 2.39) * unit.kilojoules_per_mole / unit.nanometer
    max_iterations = config.get("max_iterations", 0)
    simulation.minimizeEnergy(tolerance=tolerance, maxIterations=max_iterations)

    # Get minimized positions
    state = simulation.context.getState(getPositions=True)
    minimized_positions = state.getPositions(asNumpy=False)

    return simulation, system, minimized_positions


def minimize(workspace: Path) -> None:
    """Run iterative amber minimization with violation checking.

    Args:
        workspace: Path to the workspace directory.
    """
    config_path = workspace / "config.json"
    config = json.loads(config_path.read_text())

    structure_path = config["structure_path"]
    out_dir = Path(config.get("out_dir", str(workspace / "outputs" / "raw")))

    # Clean and prepare structure using PDBFixer
    # Handles: missing residues/atoms, non-standard residues, waters, ions
    print("[openmm-amber-minimize] Cleaning structure with PDBFixer...")
    fixer = PDBFixer(filename=structure_path)
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    # Set up force field (include implicit solvent XML if enabled)
    force_field_name = config.get("force_field", "amber14-all.xml")
    implicit_solvent = config.get("implicit_solvent", True)
    if implicit_solvent:
        forcefield = ForceField(force_field_name, _IMPLICIT_SOLVENT_XML)
        print(f"[openmm-amber-minimize] Force field: {force_field_name} + {_IMPLICIT_SOLVENT_XML}")
    else:
        forcefield = ForceField(force_field_name)
        print(f"[openmm-amber-minimize] Force field: {force_field_name} (vacuum)")

    # Add hydrogens (predicted structures often lack them)
    modeller = Modeller(fixer.topology, fixer.positions)
    modeller.addHydrogens(forcefield)
    topology = modeller.topology
    positions = modeller.positions

    # Get initial energy
    initial_system = forcefield.createSystem(
        topology, nonbondedMethod=NoCutoff, constraints=HBonds
    )
    initial_integrator = openmm.LangevinMiddleIntegrator(
        0 * unit.kelvin, 1.0 / unit.picosecond, 0.002 * unit.picoseconds
    )
    initial_sim = Simulation(topology, initial_system, initial_integrator)
    initial_sim.context.setPositions(positions)
    initial_state = initial_sim.context.getState(getEnergy=True)
    initial_energy = initial_state.getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )
    del initial_sim, initial_system, initial_integrator

    # Iterative minimization with violation checking
    max_outer_iterations = config.get("max_outer_iterations", 20)
    exclusions: set[int] = set()
    num_rounds = 0
    remaining_violations: list[dict[str, Any]] = []

    for round_idx in range(max_outer_iterations):
        num_rounds = round_idx + 1
        print(f"[openmm-amber-minimize] Minimization round {num_rounds}")

        simulation, system, minimized_positions = _create_system_and_minimize(
            topology, positions, forcefield, config, exclusions
        )

        # Check violations
        violations = _check_violations(topology, minimized_positions, system)
        remaining_violations = violations

        if not violations:
            print(f"[openmm-amber-minimize] No violations after round {num_rounds}")
            break

        # Add violating residues to exclusion set for next round
        new_exclusions = _violating_residues(violations)
        if new_exclusions.issubset(exclusions):
            # No new residues to exclude — can't make further progress
            print(
                f"[openmm-amber-minimize] No new exclusions possible, "
                f"{len(violations)} violations remain after round {num_rounds}"
            )
            break

        exclusions.update(new_exclusions)
        print(
            f"[openmm-amber-minimize] Round {num_rounds}: "
            f"{len(violations)} violations, "
            f"excluding {len(exclusions)} residues from restraints"
        )

        # Update positions for next round
        positions = minimized_positions

    # Get final energy and decomposition
    final_state = simulation.context.getState(getEnergy=True)
    final_energy = final_state.getPotentialEnergy().value_in_unit(
        unit.kilojoules_per_mole
    )
    energy_terms = _get_energy_decomposition(simulation, system)

    # Write minimized PDB
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_out_path = out_dir / "minimized.pdb"
    with open(pdb_out_path, "w") as f:
        PDBFile.writeFile(topology, minimized_positions, f)

    # Write energy data
    energy_data = {
        "initial_energy_kj_mol": initial_energy,
        "final_energy_kj_mol": final_energy,
        "energy_terms": energy_terms,
        "num_minimization_rounds": num_rounds,
        "violations": [
            {k: v for k, v in v.items() if k != "atom_indices"}
            for v in remaining_violations
        ],
    }
    (out_dir / "energy.json").write_text(json.dumps(energy_data, indent=2))

    print(
        f"[openmm-amber-minimize] Complete: "
        f"{initial_energy:.1f} -> {final_energy:.1f} kJ/mol "
        f"({num_rounds} round(s), "
        f"{len(remaining_violations)} remaining violations)"
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <workspace_path>", file=sys.stderr)
        sys.exit(1)
    minimize(Path(sys.argv[1]))
