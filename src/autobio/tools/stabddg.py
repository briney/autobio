"""StaB-ddG tool runner — ML-based binding ddG prediction.

Predicts binding stability changes (ddG) from mutations in protein-protein
complexes using the StaB-ddG method (ProteinMPNN architecture fine-tuned
on stability and SKEMPI binding data).

Reference:
    Deng et al. "StaB-ddG: Stability-aware Binding Free Energy Change
    Prediction" (ICML 2025). https://arxiv.org/abs/2507.05502
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, Any

from autobio.core.catalog import Mode, Tool, register
from autobio.core.registry import ToolCategory
from autobio.core.result import AutobioError
from autobio.schemas.base import BaseInput  # noqa: TC001 - needed at runtime for isinstance
from autobio.schemas.scoring import ScoredStructure, ScoringOutput, StaBddGInput
from autobio.tools.base import ToolRunner

if TYPE_CHECKING:
    from autobio.core.workspace import Workspace

# ---------------------------------------------------------------------------
# Container-internal paths
# ---------------------------------------------------------------------------

_STABDDG_DIR = "/app/stabddg"
_DEFAULT_CHECKPOINT = f"{_STABDDG_DIR}/model_ckpts/stabddg.pt"

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class StaBddGRunner(ToolRunner):
    """Runner for StaB-ddG binding ddG prediction.

    ``prepare_workspace`` validates inputs on the host side, copies the input
    structure into the workspace, and writes ``config.json`` with mutation
    and chain specifications.

    ``parse_output`` reads the standardised ``result_data.json`` produced by
    the container's ``standardize.py`` script.
    """

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json and copy input structure into the workspace."""
        assert isinstance(input_data, StaBddGInput)

        # -- Host-side validation (fail fast before container launch) --------
        self._validate_inputs(input_data)

        # -- Copy input structure into workspace ----------------------------
        src_path = input_data.structure_path
        dest_name = src_path.name
        shutil.copy2(src_path, workspace.inputs_dir / dest_name)
        container_structure_path = f"/workspace/inputs/{dest_name}"

        # -- Build config.json ----------------------------------------------
        config: dict[str, Any] = {
            "pdb_path": container_structure_path,
            "mutations": ",".join(input_data.mutations),
            "chains": input_data.chains,
            "checkpoint_path": _DEFAULT_CHECKPOINT,
            "output_dir": "/workspace/outputs/raw",
            "mc_samples": input_data.mc_samples,
            "noise_level": input_data.noise_level,
            "batch_size": input_data.batch_size,
            "trials": input_data.trials,
            "seed": input_data.seed,
            "device": input_data.device,
        }

        self._apply_extra(config, input_data)

        workspace.write_config(config)

    def parse_output(self, workspace: Workspace) -> ScoringOutput:
        """Read standardised outputs and return a ``ScoringOutput``."""
        result_data_path = workspace.std_output_dir / "result_data.json"
        data = json.loads(result_data_path.read_text())

        scores = []
        for s in data["scores"]:
            scores.append(
                ScoredStructure(
                    total_score=s["total_score"],
                    per_residue_scores=s.get("per_residue_scores"),
                    score_breakdown=s.get("score_breakdown"),
                    units=s.get("units"),
                    structure_path=None,
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
    def _validate_inputs(input_data: StaBddGInput) -> None:
        """Host-side validation — catch errors before container launch."""
        if not input_data.structure_path.exists():
            raise AutobioError(f"Input structure file does not exist: {input_data.structure_path}")

        # Mutations are required
        if not input_data.mutations:
            raise AutobioError("StaB-ddG requires at least one mutation.")

        # Chains specification is required
        chains = input_data.chains
        if not chains:
            raise AutobioError(
                "StaB-ddG requires 'chains' in the extra dict. "
                "Provide a string in 'binder1_binder2' format (e.g., 'ABC_DE')."
            )
        if chains.count("_") != 1:
            raise AutobioError(
                f"'chains' must be a string with exactly one underscore separator "
                f"(e.g., 'ABC_DE'), got {chains!r}."
            )


# ---------------------------------------------------------------------------
# Catalog registration — populated when this module is imported
# ---------------------------------------------------------------------------

_STABDDG_NOTES = (
    "Predicts binding ddG (change in binding free energy upon mutation) "
    "using StaB-ddG, an ML method based on the ProteinMPNN architecture "
    "fine-tuned on stability data and SKEMPI binding data.",
    "Mutations use StaB-ddG native format: [WT_AA][ChainID][Resnum][Mut_AA]. "
    "For example, 'YH103H' means Tyr at chain H position 103 mutated to His. "
    "Multiple mutations can be provided to predict their combined effect.",
    "The 'chains' parameter specifies the binding interface in 'binder1_binder2' "
    "format. For example, 'ABC_DE' defines the interface between chains A,B,C "
    "and chains D,E.",
    "Key parameters (via extra dict): 'mc_samples' (default 20, controls "
    "variance reduction), 'noise_level' (default 0.1, backbone perturbation), "
    "'trials' (default 1, number of independent predictions), 'seed' (default 0).",
    "Output ddG is in kcal/mol. Positive values indicate destabilization "
    "(weaker binding), negative values indicate stabilization (stronger binding).",
)

STABDDG_TOOL = Tool(
    name="stabddg",
    display_name="StaB-ddG",
    category=ToolCategory.SCORING,
    description=(
        "Predict binding ddG at protein-protein interfaces using StaB-ddG, a "
        "ProteinMPNN-based ML method. Returns ddG in kcal/mol."
    ),
    version="1.0.0",
    image_tag="stabddg:1.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="predict",
    modes={
        "predict": Mode(
            name="predict",
            display_name="Predict ddG",
            description="Predict binding ddG for mutations in a protein complex.",
            input_schema=StaBddGInput,
            output_schema=ScoringOutput,
            default_timeout=600,
            notes=_STABDDG_NOTES,
        )
    },
    keywords=("stabddg", "ddg", "binding affinity", "mutation", "interface"),
)
"""Catalog Tool for StaB-ddG — exposed for tests re-registering after CATALOG-clearing fixtures."""

register(STABDDG_TOOL)
