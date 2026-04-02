"""EvoEF2 tool runners — structure repair, binding energy, and mutant building.

All EvoEF2 tools share a single ``EvoEF2Runner`` class that dispatches by
``tool_name`` using the ``_VARIANT_CONFIG`` dict.  All variants share a single
Docker image since EvoEF2 is one binary with a ``--command=`` flag.

EvoEF2 is a physics-based energy function (compiled C++) — **CPU-only**,
no GPU required, no model weights.

Supported tools:

- ``evoef2_repair`` — Repair incomplete side chains and optimize hydrogens.
- ``evoef2_binding`` — Compute protein–protein binding energy (auto-repair by default).
- ``evoef2_build_mutant`` — Build mutant structures from a mutation specification.
"""

from __future__ import annotations

import json
import re
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
# Container-internal paths
# ---------------------------------------------------------------------------

_EVOEF2_BIN = "/app/evoef2/EvoEF2"

# ---------------------------------------------------------------------------
# Variant configuration — maps tool name to protocol-specific settings
# ---------------------------------------------------------------------------

_VARIANT_CONFIG: dict[str, dict[str, Any]] = {
    "evoef2_repair": {
        "command": "RepairStructure",
        "produces_structure": True,
        "requires_mutations": False,
        "has_auto_repair": False,
    },
    "evoef2_binding": {
        "command": "ComputeBinding",
        "produces_structure": False,
        "requires_mutations": False,
        "has_auto_repair": True,
    },
    "evoef2_build_mutant": {
        "command": "BuildMutant",
        "produces_structure": True,
        "requires_mutations": True,
        "has_auto_repair": False,
    },
}

# Keys in ``extra`` that are consumed by the runner and should NOT be
# flat-merged into config.json.
_CONSUMED_EXTRA_KEYS = frozenset({"mutations", "repair", "split_chains"})

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
    """Runner for EvoEF2 computational biology tools.

    ``prepare_workspace`` validates inputs on the host side, copies the input
    structure into the workspace, generates mutation files for BuildMutant, and
    writes ``config.json``.

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
            "command": variant_cfg["command"],
            "structure_path": container_structure_path,
            "evoef2_bin": _EVOEF2_BIN,
            "out_dir": "/workspace/outputs/raw",
        }

        # Binding: auto-repair flag and chain split
        if variant_cfg["has_auto_repair"]:
            config["repair"] = input_data.extra.get("repair", True)
            split_chains = input_data.extra.get("split_chains")
            if split_chains:
                config["split_chains"] = split_chains

        # BuildMutant: write mutation file
        if variant_cfg["requires_mutations"]:
            mutations: list[str] = input_data.extra["mutations"]
            config["mutations"] = mutations
            self._write_mutation_file(mutations, workspace)
            config["mutant_file"] = "/workspace/inputs/individual_list.txt"

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

        # EvoEF2 only supports PDB format
        suffix = input_data.structure_path.suffix.lower()
        if suffix not in (".pdb",):
            raise AutobioError(
                f"EvoEF2 only supports PDB format, got '{suffix}'. "
                "Convert mmCIF/other formats to PDB before using EvoEF2 tools."
            )

        variant_cfg = _VARIANT_CONFIG[self.tool_name]

        # BuildMutant requires mutations
        if variant_cfg["requires_mutations"]:
            mutations = input_data.extra.get("mutations")
            if not mutations:
                raise AutobioError(
                    f"Tool {self.tool_name!r} requires 'mutations' in the extra dict. "
                    f"{_MUTATION_FORMAT_HELP}"
                )
            if not isinstance(mutations, list) or not all(isinstance(m, str) for m in mutations):
                raise AutobioError(
                    f"'mutations' must be a list of strings, got {type(mutations).__name__}. "
                    f"{_MUTATION_FORMAT_HELP}"
                )
            for m in mutations:
                if not _MUTATION_RE.match(m):
                    raise AutobioError(f"Invalid mutation format: {m!r}. {_MUTATION_FORMAT_HELP}")

        # Validate split_chains format for binding if provided
        if self.tool_name == "evoef2_binding":
            split_chains = input_data.extra.get("split_chains")
            if split_chains is not None and (
                not isinstance(split_chains, str) or split_chains.count(",") != 1
            ):
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
# Registry entries — populated when this module is imported
# ---------------------------------------------------------------------------

_REPAIR_NOTES = (
    "Repairs incomplete side chains and optimizes hydrogen positions in a "
    "PDB structure. Recommended as preprocessing before scoring or binding "
    "energy calculations, especially for crystal structures with missing atoms.",
    "EvoEF2 uses backbone-dependent rotamer libraries (Dunbrack 2010) to rebuild side chains.",
    "Output includes the repaired PDB structure and the total energy of the repaired structure.",
)

_REPAIR_INPUT_FORMAT = (
    "Provide a PDB file via structure_path. EvoEF2 only supports PDB "
    "format — mmCIF files must be converted first.",
)

_BINDING_NOTES = (
    "Computes the binding energy between protein chains in a complex. "
    "By default, structures are auto-repaired before scoring "
    "(extra['repair'] = True). Set extra['repair'] = False to skip.",
    "The binding energy is computed as: E(complex) - E(chain_group_1) - "
    "E(chain_group_2). A more negative value indicates stronger binding.",
    "Use extra['split_chains'] to specify which chains form each binding "
    "partner (e.g., 'A,BC' means chain A vs. chains B+C). If omitted, "
    "EvoEF2 uses its default chain grouping.",
    "Output includes total binding energy and a per-term energy breakdown "
    "(van der Waals, electrostatics, desolvation, hydrogen bonds, etc.).",
)

_BINDING_INPUT_FORMAT = (
    "Provide a multi-chain PDB complex via structure_path. Optional "
    "parameters in extra dict: 'repair' (bool, default True), "
    "'split_chains' (str, e.g., 'A,BC').",
)

_BUILD_MUTANT_NOTES = (
    "Builds mutant protein structures by introducing specified amino acid "
    "substitutions and optimizing the local environment. Produces a "
    "model PDB file.",
    "Mutations are specified as a list of strings in extra['mutations']. "
    "Format: 'EA63Q' means chain E, Ala-63 -> Gln. Multiple mutations "
    "are applied simultaneously.",
)

_BUILD_MUTANT_INPUT_FORMAT = (
    "Provide a PDB file via structure_path. Required: extra['mutations'] "
    "as a list of strings (e.g., ['EA63Q', 'KB42A']).",
)

TOOL_REGISTRY["evoef2_repair"] = ToolEntry(
    image_tag="evoef2:1.0.0",
    category=ToolCategory.SCORING,
    requires_gpu=False,
    gpu_count=0,
    input_schema=ScoringInput,
    output_schema=ScoringOutput,
    default_timeout=600,
    supports_batch=False,
    description=(
        "Repair a protein structure by rebuilding incomplete side chains "
        "and optimizing hydrogen positions using EvoEF2's physics-based "
        "rotamer library. Essential preprocessing for crystal structures "
        "with missing atoms before scoring or binding energy calculations."
    ),
    version="1.0.0",
    notes=_REPAIR_NOTES,
    input_format=_REPAIR_INPUT_FORMAT,
)

TOOL_REGISTRY["evoef2_binding"] = ToolEntry(
    image_tag="evoef2:1.0.0",
    category=ToolCategory.SCORING,
    requires_gpu=False,
    gpu_count=0,
    input_schema=ScoringInput,
    output_schema=ScoringOutput,
    default_timeout=600,
    supports_batch=False,
    description=(
        "Compute protein–protein binding energy using EvoEF2's physics-based "
        "energy function. Decomposes binding energy into van der Waals, "
        "electrostatic, desolvation, and hydrogen bond contributions. "
        "Auto-repairs structures by default before scoring."
    ),
    version="1.0.0",
    notes=_BINDING_NOTES,
    input_format=_BINDING_INPUT_FORMAT,
)

TOOL_REGISTRY["evoef2_build_mutant"] = ToolEntry(
    image_tag="evoef2:1.0.0",
    category=ToolCategory.SCORING,
    requires_gpu=False,
    gpu_count=0,
    input_schema=ScoringInput,
    output_schema=ScoringOutput,
    default_timeout=600,
    supports_batch=False,
    description=(
        "Build mutant protein structures by introducing amino acid "
        "substitutions and optimizing the local environment with EvoEF2's "
        "rotamer library. Produces model PDB files suitable for downstream "
        "binding energy or stability calculations."
    ),
    version="1.0.0",
    notes=_BUILD_MUTANT_NOTES,
    input_format=_BUILD_MUTANT_INPUT_FORMAT,
)
