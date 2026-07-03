"""BA-ddG tool runner — Boltzmann-aligned binding ddG prediction.

Predicts binding stability changes (ddG) from mutations in protein-protein
complexes using BA-ddG, which applies Boltzmann Alignment to a modified
ProteinMPNN inverse folding model via a thermodynamic cycle.

Reference:
    Li et al. "Boltzmann-Aligned Inverse Folding Model as a Predictor of
    Mutational Effects on Protein-Protein Interactions" (ICLR 2025).
    https://arxiv.org/abs/2410.09543
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, Any

from autobio.core.catalog import Mode, Tool, register
from autobio.core.registry import ToolCategory
from autobio.core.result import AutobioError
from autobio.schemas.base import BaseInput  # noqa: TC001 - needed at runtime for isinstance
from autobio.schemas.scoring import BAddGInput, ScoredStructure, ScoringOutput
from autobio.tools.base import ToolRunner

if TYPE_CHECKING:
    from autobio.core.workspace import Workspace

# ---------------------------------------------------------------------------
# Container-internal paths
# ---------------------------------------------------------------------------

_BADDG_DIR = "/app/baddg"
_DEFAULT_MPNN_CHECKPOINT = f"{_BADDG_DIR}/ckpt/soluble_model_weights/v_48_020.pt"
_DEFAULT_DDG_CHECKPOINT = f"{_BADDG_DIR}/ckpt/ddg_model.ckpt"

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class BAddGRunner(ToolRunner):
    """Runner for BA-ddG binding ddG prediction.

    ``prepare_workspace`` validates inputs on the host side, copies the input
    structure into the workspace, and writes ``config.json`` with mutation
    and chain specifications.

    ``parse_output`` reads the standardised ``result_data.json`` produced by
    the container's ``standardize.py`` script.
    """

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json and copy input structure into the workspace."""
        assert isinstance(input_data, BAddGInput)

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
            "mpnn_checkpoint_path": _DEFAULT_MPNN_CHECKPOINT,
            "ddg_checkpoint_path": _DEFAULT_DDG_CHECKPOINT,
            "output_dir": "/workspace/outputs/raw",
            "n_folds": input_data.n_folds,
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
    def _validate_inputs(input_data: BAddGInput) -> None:
        """Host-side validation — catch errors before container launch."""
        if not input_data.structure_path.exists():
            raise AutobioError(f"Input structure file does not exist: {input_data.structure_path}")

        # Mutations are required
        if not input_data.mutations:
            raise AutobioError(
                "BA-ddG requires 'mutations' in the extra dict. "
                "Provide a list of mutation strings in format: "
                "[WT_AA][ChainID][Resnum][Mut_AA] (e.g., ['YH103H', 'QD30V'])."
            )

        # Chains specification is required
        chains = input_data.chains
        if not chains:
            raise AutobioError(
                "BA-ddG requires 'chains' in the extra dict. "
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

_BADDG_NOTES = (
    "Predicts binding ddG (change in binding free energy upon mutation) "
    "using BA-ddG, a Boltzmann-aligned inverse folding model based on "
    "ProteinMPNN. Uses a thermodynamic cycle to score both the complex "
    "and unbound states.",
    "Mutations use format: [WT_AA][ChainID][Resnum][Mut_AA]. "
    "For example, 'YH103H' means Tyr at chain H position 103 mutated to His. "
    "Multiple mutations can be provided to predict their combined effect.",
    "The 'chains' parameter specifies the binding interface in 'binder1_binder2' "
    "format. For example, 'ABC_DE' defines the interface between chains A,B,C "
    "and chains D,E.",
    "Predictions are averaged across 3 cross-validation folds by default. "
    "Set extra['n_folds'] to 1 or 2 for faster (but less robust) predictions.",
    "Output ddG is in kcal/mol. Positive values indicate destabilization "
    "(weaker binding), negative values indicate stabilization (stronger binding).",
)

BADDG_TOOL = Tool(
    name="baddg",
    display_name="BA-ddG",
    category=ToolCategory.SCORING,
    description=(
        "Predict binding ddG at protein-protein interfaces using BA-ddG, a "
        "Boltzmann-aligned inverse folding model. Returns ddG in kcal/mol."
    ),
    version="1.0.0",
    image_tag="baddg:1.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="predict",
    modes={
        "predict": Mode(
            name="predict",
            display_name="Predict ddG",
            description="Predict binding ddG for mutations in a protein complex.",
            input_schema=BAddGInput,
            output_schema=ScoringOutput,
            default_timeout=600,
            notes=_BADDG_NOTES,
        )
    },
    keywords=("baddg", "ddg", "binding affinity", "mutation", "interface"),
)
"""Catalog Tool for BA-ddG — exposed for tests re-registering after CATALOG-clearing fixtures."""

register(BADDG_TOOL)
