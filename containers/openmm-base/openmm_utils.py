"""Shared OpenMM utility functions for all autobio OpenMM containers.

Provides structure preparation, force field setup, solvation, restraint
management, energy decomposition, and solvent stripping used by
openmm-amber-minimize, openmm-amber-relax, and openmm-md-simulate.
"""

from __future__ import annotations

from typing import Any

import openmm
from openmm import unit
from openmm.app import (
    ForceField,
    Modeller,
    PDBFile,
    Simulation,
)
from pdbfixer import PDBFixer


# ---------------------------------------------------------------------------
# Water model XML mapping
# ---------------------------------------------------------------------------

# Maps user-facing water model names to OpenMM XML files bundled with
# the Amber14 force field distribution.
WATER_MODEL_XML: dict[str, str] = {
    "tip3p": "amber14/tip3p.xml",
    "tip4pew": "amber14/tip4pew.xml",
    "spce": "amber14/spce.xml",
}

# Implicit solvent model XML (included with OpenMM)
IMPLICIT_SOLVENT_XML = "implicit/obc2.xml"

# Maps user-facing box shape names to OpenMM Modeller.addSolvent() values
BOX_SHAPE_MAP: dict[str, str] = {
    "cubic": "cube",
    "dodecahedron": "dodecahedron",
    "truncated_octahedron": "octahedron",
}


# ---------------------------------------------------------------------------
# Structure cleanup
# ---------------------------------------------------------------------------


def cleanup_structure(
    structure_path: str,
    *,
    keep_water: bool = False,
) -> tuple[openmm.app.Topology, list[openmm.Vec3]]:
    """Clean a PDB structure using PDBFixer and return topology + positions.

    Removes heterogens (optionally keeps water), finds and adds missing
    residues and atoms.

    Args:
        structure_path: Path to the input PDB file.
        keep_water: If True, retain crystallographic water molecules.

    Returns:
        Tuple of (topology, positions) after cleanup.
    """
    print(f"[openmm] Cleaning structure with PDBFixer: {structure_path}")
    fixer = PDBFixer(filename=structure_path)
    fixer.removeHeterogens(keepWater=keep_water)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    return fixer.topology, fixer.positions


# ---------------------------------------------------------------------------
# Force field setup
# ---------------------------------------------------------------------------


def create_forcefield(
    force_field: str,
    *,
    implicit_solvent: bool = False,
    water_model: str | None = None,
) -> ForceField:
    """Create an OpenMM ForceField with appropriate solvent model.

    Args:
        force_field: Primary force field XML (e.g., ``"amber14-all.xml"``).
        implicit_solvent: If True, add OBC2 implicit solvent model.
        water_model: Explicit water model name (e.g., ``"tip3p"``). Ignored
            if *implicit_solvent* is True.

    Returns:
        Configured :class:`ForceField` instance.
    """
    xmls = [force_field]
    if implicit_solvent:
        xmls.append(IMPLICIT_SOLVENT_XML)
        print(f"[openmm] Force field: {force_field} + {IMPLICIT_SOLVENT_XML}")
    elif water_model:
        water_xml = WATER_MODEL_XML.get(water_model)
        if water_xml is None:
            raise ValueError(
                f"Unknown water model {water_model!r}. "
                f"Allowed: {', '.join(sorted(WATER_MODEL_XML))}"
            )
        xmls.append(water_xml)
        print(f"[openmm] Force field: {force_field} + {water_xml}")
    else:
        print(f"[openmm] Force field: {force_field} (vacuum)")

    return ForceField(*xmls)


def add_hydrogens(
    topology: openmm.app.Topology,
    positions: list[openmm.Vec3],
    forcefield: ForceField,
) -> Modeller:
    """Add hydrogens to a cleaned structure using Modeller.

    Args:
        topology: Cleaned topology from :func:`cleanup_structure`.
        positions: Cleaned positions.
        forcefield: Force field to use for hydrogen placement.

    Returns:
        Modeller with hydrogens added.
    """
    modeller = Modeller(topology, positions)
    modeller.addHydrogens(forcefield)
    return modeller


# ---------------------------------------------------------------------------
# Solvation
# ---------------------------------------------------------------------------


def solvate_system(
    modeller: Modeller,
    forcefield: ForceField,
    config: dict[str, Any],
) -> Modeller:
    """Add explicit solvent and ions to the system.

    Reads solvation parameters from *config*:
    - ``box_shape`` (str): ``"cubic"``, ``"dodecahedron"``, ``"truncated_octahedron"``
    - ``box_padding`` (float): Padding in nm (default 1.0)
    - ``ion_type`` (str): ``"NaCl"`` or ``"KCl"`` (default ``"NaCl"``)
    - ``ion_concentration`` (float): Molar concentration (default 0.15)

    Args:
        modeller: Modeller with hydrogens already added.
        forcefield: Force field (must include a water model XML).
        config: Configuration dict with solvation parameters.

    Returns:
        The same Modeller instance, now solvated.
    """
    box_shape = config.get("box_shape", "cubic")
    box_padding = config.get("box_padding", 1.0)
    ion_type = config.get("ion_type", "NaCl")
    ion_concentration = config.get("ion_concentration", 0.15)

    openmm_box_shape = BOX_SHAPE_MAP.get(box_shape, "cube")

    # Determine ion species
    if ion_type == "KCl":
        positive_ion = "K+"
        negative_ion = "Cl-"
    else:
        positive_ion = "Na+"
        negative_ion = "Cl-"

    print(
        f"[openmm] Solvating: {openmm_box_shape} box, "
        f"{box_padding} nm padding, "
        f"{ion_concentration} M {ion_type}"
    )

    # Map config water model name to OpenMM addSolvent model name
    water_model = config.get("water_model", "tip3p")

    modeller.addSolvent(
        forcefield,
        model=water_model,
        padding=box_padding * unit.nanometer,
        boxShape=openmm_box_shape,
        positiveIon=positive_ion,
        negativeIon=negative_ion,
        ionicStrength=ion_concentration * unit.molar,
    )

    return modeller


# ---------------------------------------------------------------------------
# Restraints
# ---------------------------------------------------------------------------


def add_restraints(
    system: openmm.System,
    topology: openmm.app.Topology,
    positions: list[openmm.Vec3],
    restraint_set: str,
    stiffness_kj_per_nm2: float,
    exclusions: set[int] | None = None,
) -> None:
    """Add harmonic positional restraints to the system.

    Args:
        system: OpenMM system to add the force to.
        topology: Molecular topology.
        positions: Reference positions for restraints.
        restraint_set: Which atoms to restrain: ``"ca"``, ``"heavy_atoms"``,
            or ``"none"`` (no-op).
        stiffness_kj_per_nm2: Spring constant in kJ/mol/nm^2.
        exclusions: Residue indices to exclude from restraints.
    """
    if restraint_set == "none":
        return

    if exclusions is None:
        exclusions = set()

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
        if atom.element is None:
            continue  # skip virtual sites (e.g., TIP4P-Ew extra particles)
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


def get_energy_decomposition(
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
    for i, force in enumerate(system.getForces()):
        force.setForceGroup(i)

    simulation.context.reinitialize(preserveState=True)

    breakdown: dict[str, float] = {}
    for i, force in enumerate(system.getForces()):
        state = simulation.context.getState(getEnergy=True, groups={i})
        energy = state.getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        force_name = type(force).__name__
        breakdown[force_name] = energy
    return breakdown


# ---------------------------------------------------------------------------
# Solvent stripping
# ---------------------------------------------------------------------------


def strip_solvent(
    topology: openmm.app.Topology,
    positions: list[openmm.Vec3],
) -> tuple[openmm.app.Topology, list[openmm.Vec3]]:
    """Remove water molecules and ions, returning protein-only topology and positions.

    Args:
        topology: Full topology including solvent.
        positions: Full position list.

    Returns:
        Tuple of (protein_topology, protein_positions).
    """
    modeller = Modeller(topology, positions)
    modeller.deleteWater()

    # Also remove common ions (Na+, Cl-, K+)
    ion_names = {"NA", "CL", "K", "HOH", "WAT"}
    to_delete = [
        r for r in modeller.topology.residues() if r.name in ion_names
    ]
    if to_delete:
        modeller.delete(to_delete)

    return modeller.topology, modeller.positions


def write_pdb(
    topology: openmm.app.Topology,
    positions: list[openmm.Vec3],
    path: str,
) -> None:
    """Write a PDB file.

    Args:
        topology: Molecular topology.
        positions: Atom positions.
        path: Output file path.
    """
    with open(path, "w") as f:
        PDBFile.writeFile(topology, positions, f)
