"""LigandMPNN sidechain packing tool — build mutant structures.

Uses LigandMPNN's sidechain packing neural network to introduce amino acid
mutations and repack sidechains.  The packing model predicts chi1–chi4
torsion angles as mixtures of von Mises distributions, producing full-atom
PDB structures with per-residue confidence scores.

This is conceptually an alternative to ``evoef2_build_mutant``, which uses
a physics-based rotamer library.  The LigandMPNN packer uses a learned model
that is also ligand-aware (can consider bound ligands when packing).

The tool runs in a dedicated container built from the original
``dauparas/LigandMPNN`` code (the Rosetta Commons foundry does not expose
sidechain packing).
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from autobio.core.catalog import Mode, Tool, register
from autobio.core.registry import ToolCategory
from autobio.core.result import AutobioError
from autobio.schemas.base import BaseInput  # noqa: TC001 - needed at runtime for isinstance
from autobio.schemas.scoring import LigandMPNNPackerInput, ScoredStructure, ScoringOutput
from autobio.tools.base import ToolRunner

if TYPE_CHECKING:
    from autobio.core.workspace import Workspace

# ---------------------------------------------------------------------------
# Container-internal paths
# ---------------------------------------------------------------------------

_CHECKPOINT_SC = "/app/LigandMPNN/model_params/ligandmpnn_sc_v_32_002_16.pt"
_CHECKPOINT_BB = "/app/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt"

# ---------------------------------------------------------------------------
# Mutation validation — same regex and format as EvoEF2 build_mutant
# ---------------------------------------------------------------------------

_MUTATION_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY][A-Za-z]\d+[ACDEFGHIKLMNPQRSTVWY]$")

_MUTATION_FORMAT_HELP = (
    "Mutations must be strings like 'EA63Q' (WT-chain-resnum-new). "
    "Each character: single-letter amino acid code for wild-type, "
    "chain ID letter, residue number, single-letter amino acid code for mutant."
)

# ---------------------------------------------------------------------------
# Default packing parameters
# ---------------------------------------------------------------------------

_DEFAULT_NUM_PACKS = 4
_DEFAULT_NUM_DENOISING_STEPS = 3
_DEFAULT_NUM_SAMPLES = 16

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class LigandMPNNPackerRunner(ToolRunner):
    """Runner for LigandMPNN sidechain packing (build mutant structures).

    ``prepare_workspace`` validates mutations, copies the input PDB, and
    writes ``config.json`` with packing parameters.

    ``parse_output`` reads the standardised ``result_data.json`` produced by
    the container's ``standardize.py`` script.
    """

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json and copy input structure into the workspace."""
        assert isinstance(input_data, LigandMPNNPackerInput)

        # -- Host-side validation (fail fast before container launch) --------
        self._validate_inputs(input_data)

        # -- Copy input structure into workspace ----------------------------
        src_path = input_data.structure_path
        dest_name = src_path.name
        shutil.copy2(src_path, workspace.inputs_dir / dest_name)
        container_structure_path = f"/workspace/inputs/{dest_name}"

        # -- Build config.json ----------------------------------------------
        config: dict[str, Any] = {
            "structure_path": container_structure_path,
            "mutations": input_data.mutations,
            "checkpoint_sc": _CHECKPOINT_SC,
            "checkpoint_bb": _CHECKPOINT_BB,
            "num_packs": input_data.num_packs,
            "num_denoising_steps": input_data.num_denoising_steps,
            "num_samples": input_data.num_samples,
            "repack_everything": input_data.repack_everything,
            "pack_with_ligand_context": input_data.pack_with_ligand_context,
        }

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
                structure_path = _resolve_container_path(s["structure_path"], workspace)

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

    def _validate_inputs(self, input_data: LigandMPNNPackerInput) -> None:
        """Host-side validation — catch errors before container launch."""
        if not input_data.structure_path.exists():
            raise AutobioError(f"Input structure file does not exist: {input_data.structure_path}")

        suffix = input_data.structure_path.suffix.lower()
        if suffix not in (".pdb",):
            raise AutobioError(
                f"LigandMPNN sidechain packer only supports PDB format, got '{suffix}'. "
                "Convert mmCIF/other formats to PDB before using this tool."
            )

        mutations = input_data.mutations
        if not mutations:
            raise AutobioError(
                f"LigandMPNN packer requires at least one mutation. {_MUTATION_FORMAT_HELP}"
            )
        for m in mutations:
            if not _MUTATION_RE.match(m):
                raise AutobioError(f"Invalid mutation format: {m!r}. {_MUTATION_FORMAT_HELP}")


def _resolve_container_path(container_path_str: str, workspace: Workspace) -> Path:
    """Map a container-internal ``/workspace/...`` path to the host workspace."""
    container_path = Path(container_path_str)
    try:
        relative = container_path.relative_to("/workspace")
    except ValueError:
        return container_path
    return workspace.root / relative


# ---------------------------------------------------------------------------
# Catalog registration — populated when this module is imported
# ---------------------------------------------------------------------------

_NOTES = (
    "Builds mutant protein structures using LigandMPNN's neural network "
    "sidechain packing model, which predicts chi1–chi4 torsion angles as "
    "mixtures of von Mises distributions. Produces full-atom PDB structures.",
    "Mutations are specified as a list of strings. Format: 'EA63Q' means "
    "chain E, Ala-63 -> Gln. Multiple mutations are applied simultaneously.",
    "Scores are chi-angle log-probabilities from the packing model "
    "(higher = more confident). These are NOT energy scores like EvoEF2.",
    "The packer is ligand-aware: if the input PDB contains bound ligands "
    "(HETATM records), they are used as context during sidechain packing. "
    "This is an advantage over physics-based methods for ligand-binding sites.",
    "Proline ring geometry and disulfide bonds are predicted by the model "
    "but may benefit from downstream energy minimization for accuracy.",
)

LIGANDMPNN_PACKER_TOOL = Tool(
    name="ligandmpnn_build_mutant",
    display_name="LigandMPNN Build Mutant",
    category=ToolCategory.SCORING,
    description=(
        "Build mutant protein structures by introducing amino acid substitutions and "
        "repacking sidechains with LigandMPNN's neural-network sidechain packing model. "
        "Predicts chi angles as mixtures of von Mises distributions, producing full-atom "
        "PDB structures with confidence scores."
    ),
    version="1.0.0",
    image_tag="ligandmpnn-packer:1.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="build_mutant",
    modes={
        "build_mutant": Mode(
            name="build_mutant",
            display_name="Build mutant",
            description="Introduce mutations and repack sidechains into full-atom structures.",
            input_schema=LigandMPNNPackerInput,
            output_schema=ScoringOutput,
            default_timeout=600,
            notes=_NOTES,
        )
    },
    keywords=("ligandmpnn", "mutant", "sidechain packing", "repack", "mutation"),
)
"""Catalog Tool for the LigandMPNN sidechain packer."""

register(LIGANDMPNN_PACKER_TOOL)
