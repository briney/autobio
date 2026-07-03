"""EvoEF2 tool — structure repair, binding energy, and mutant building.

A single catalog Tool, ``evoef2``, exposing three Modes sharing the
``EvoEF2Runner`` runner class:

- ``repair`` (default) — Rebuild incomplete side chains, optimize hydrogens.
- ``binding`` — Compute protein-protein binding energy (auto-repair by default).
- ``build_mutant`` — Build mutant structures from a mutation specification.

All three modes share a single Docker image since EvoEF2 is one binary with a
``--command=`` flag. EvoEF2 is a physics-based energy function (compiled
C++) — **CPU-only**, no GPU required, no model weights.
"""

from __future__ import annotations

import json
import re
import shutil
from typing import TYPE_CHECKING, Any

from autobio.core.catalog import Mode, Tool, register
from autobio.core.registry import ToolCategory
from autobio.core.result import AutobioError
from autobio.schemas.base import BaseInput  # noqa: TC001 - needed at runtime for isinstance
from autobio.schemas.scoring import (
    EvoEF2BaseInput,
    EvoEF2BindingInput,
    EvoEF2BuildMutantInput,
    EvoEF2RepairInput,
    ScoredStructure,
    ScoringOutput,
)
from autobio.tools.base import ToolRunner

if TYPE_CHECKING:
    from autobio.core.workspace import Workspace

# ---------------------------------------------------------------------------
# Container-internal paths
# ---------------------------------------------------------------------------

_EVOEF2_BIN = "/app/evoef2/EvoEF2"

# ---------------------------------------------------------------------------
# Mode configuration — maps mode name to the EvoEF2 --command= value
# ---------------------------------------------------------------------------

_MODE_COMMAND: dict[str, str] = {
    "repair": "RepairStructure",
    "binding": "ComputeBinding",
    "build_mutant": "BuildMutant",
}

# Mutation format: single-letter WT AA + chain ID + residue number + single-letter new AA
# e.g., "EA63Q" means chain E, Ala-63 -> Gln
_MUTATION_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY][A-Za-z]\d+[ACDEFGHIKLMNPQRSTVWY]$")

_MUTATION_FORMAT_HELP = (
    "Mutations must be strings like 'EA63Q' (WT-chain-resnum-new). "
    "Each character: single-letter amino acid code for wild-type, "
    "chain ID letter, residue number, single-letter amino acid code for mutant."
)

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class EvoEF2Runner(ToolRunner):
    """Runner for the EvoEF2 repair/binding/build_mutant modes.

    ``prepare_workspace`` validates inputs on the host side, copies the input
    structure into the workspace, generates mutation files for build_mutant,
    and writes ``config.json``.

    ``parse_output`` reads the standardised ``result_data.json`` produced by
    the container's ``standardize.py`` script.
    """

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json and copy input structure into the workspace."""
        assert isinstance(input_data, EvoEF2BaseInput)
        assert self.current_mode is not None
        mode = self.current_mode.name

        # -- Host-side validation (fail fast before container launch) --------
        self._validate_inputs(input_data)

        # -- Copy input structure into workspace ----------------------------
        src_path = input_data.structure_path
        dest_name = src_path.name
        shutil.copy2(src_path, workspace.inputs_dir / dest_name)
        container_structure_path = f"/workspace/inputs/{dest_name}"

        # -- Build config.json ----------------------------------------------
        config: dict[str, Any] = {
            "command": _MODE_COMMAND[mode],
            "structure_path": container_structure_path,
            "evoef2_bin": _EVOEF2_BIN,
            "out_dir": "/workspace/outputs/raw",
        }

        # Binding: auto-repair flag and chain split
        if mode == "binding":
            assert isinstance(input_data, EvoEF2BindingInput)
            config["repair"] = input_data.repair
            if input_data.split_chains:
                config["split_chains"] = input_data.split_chains

        # BuildMutant: write mutation file
        if mode == "build_mutant":
            assert isinstance(input_data, EvoEF2BuildMutantInput)
            config["mutations"] = input_data.mutations
            self._write_mutation_file(input_data.mutations, workspace)
            config["mutant_file"] = "/workspace/inputs/individual_list.txt"

        self._apply_extra(config, input_data)

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

    def _validate_inputs(self, input_data: EvoEF2BaseInput) -> None:
        """Host-side validation — catch errors before container launch."""
        if not input_data.structure_path.exists():
            raise AutobioError(f"Input structure file does not exist: {input_data.structure_path}")

        # EvoEF2 only supports PDB format
        suffix = input_data.structure_path.suffix.lower()
        if suffix not in (".pdb",):
            raise AutobioError(
                f"EvoEF2 only supports PDB format, got '{suffix}'. "
                "Convert mmCIF/other formats to PDB before using EvoEF2 tools."
            )

        assert self.current_mode is not None
        mode = self.current_mode.name

        # BuildMutant requires mutations
        if mode == "build_mutant":
            assert isinstance(input_data, EvoEF2BuildMutantInput)
            if not input_data.mutations:
                raise AutobioError(
                    f"EvoEF2 build_mutant requires at least one mutation. {_MUTATION_FORMAT_HELP}"
                )
            for m in input_data.mutations:
                if not _MUTATION_RE.match(m):
                    raise AutobioError(f"Invalid mutation format: {m!r}. {_MUTATION_FORMAT_HELP}")

        # Validate split_chains format for binding if provided
        if mode == "binding":
            assert isinstance(input_data, EvoEF2BindingInput)
            split_chains = input_data.split_chains
            if split_chains is not None and split_chains.count(",") != 1:
                raise AutobioError(
                    "The 'split_chains' parameter must be a string with exactly one "
                    "comma separating two chain groups (e.g., 'A,BC' or 'AB,CD')."
                )

    @staticmethod
    def _write_mutation_file(mutations: list[str], workspace: Workspace) -> None:
        """Write EvoEF2 individual_list.txt mutation file.

        EvoEF2 format: mutations on one line comma-separated, terminated with ``;``.
        Example: ``EA63Q,KB42A;``
        """
        line = ",".join(mutations) + ";"
        (workspace.inputs_dir / "individual_list.txt").write_text(line + "\n")


# ---------------------------------------------------------------------------
# Catalog registration — populated when this module is imported
# ---------------------------------------------------------------------------

_REPAIR_NOTES = (
    "Repairs incomplete side chains and optimizes hydrogen positions in a "
    "PDB structure. Recommended as preprocessing before scoring or binding "
    "energy calculations, especially for crystal structures with missing atoms.",
    "EvoEF2 uses backbone-dependent rotamer libraries (Dunbrack 2010) to rebuild side chains.",
    "Output includes the repaired PDB structure and the total energy of the repaired structure.",
)

_BINDING_NOTES = (
    "Computes the binding energy between protein chains in a complex. "
    "By default, structures are auto-repaired before scoring "
    "(the repair field defaults to True). Set repair = False to skip.",
    "The binding energy is computed as: E(complex) - E(chain_group_1) - "
    "E(chain_group_2). A more negative value indicates stronger binding.",
    "Use the split_chains field to specify which chains form each binding "
    "partner (e.g., 'A,BC' means chain A vs. chains B+C). If omitted, "
    "EvoEF2 uses its default chain grouping.",
    "Output includes total binding energy and a per-term energy breakdown "
    "(van der Waals, electrostatics, desolvation, hydrogen bonds, etc.).",
)

_BUILD_MUTANT_NOTES = (
    "Builds mutant protein structures by introducing specified amino acid "
    "substitutions and optimizing the local environment. Produces a "
    "model PDB file.",
    "Mutations are specified as a list of strings in the mutations field. "
    "Format: 'EA63Q' means chain E, Ala-63 -> Gln. Multiple mutations "
    "are applied simultaneously.",
)

EVOEF2_TOOL = Tool(
    name="evoef2",
    display_name="EvoEF2",
    category=ToolCategory.SCORING,
    description=(
        "EvoEF2 physics-based protein structure repair, binding-energy scoring, and "
        "mutant building. Modes: repair, binding, build_mutant."
    ),
    version="1.0.0",
    image_tag="evoef2:1.0.0",
    requires_gpu=False,
    gpu_count=0,
    default_mode="repair",
    modes={
        "repair": Mode(
            name="repair",
            display_name="Repair",
            description="Rebuild incomplete side chains and optimize hydrogen positions.",
            input_schema=EvoEF2RepairInput,
            output_schema=ScoringOutput,
            default_timeout=600,
            notes=_REPAIR_NOTES,
        ),
        "binding": Mode(
            name="binding",
            display_name="Binding energy",
            description="Compute protein-protein binding energy (auto-repairs by default).",
            input_schema=EvoEF2BindingInput,
            output_schema=ScoringOutput,
            default_timeout=600,
            notes=_BINDING_NOTES,
        ),
        "build_mutant": Mode(
            name="build_mutant",
            display_name="Build mutant",
            description="Introduce amino-acid substitutions and optimize the local environment.",
            input_schema=EvoEF2BuildMutantInput,
            output_schema=ScoringOutput,
            default_timeout=600,
            notes=_BUILD_MUTANT_NOTES,
        ),
    },
    keywords=("evoef2", "scoring", "repair", "binding energy", "mutant", "ddg"),
)
"""Catalog Tool for EvoEF2 (repair/binding/build_mutant modes)."""

register(EVOEF2_TOOL)
