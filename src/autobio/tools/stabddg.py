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

_STABDDG_DIR = "/app/stabddg"
_DEFAULT_CHECKPOINT = f"{_STABDDG_DIR}/model_ckpts/stabddg.pt"

# Keys in ``extra`` that are consumed by the runner and should NOT be
# flat-merged into config.json.
_CONSUMED_EXTRA_KEYS = frozenset(
    {
        "mutations",
        "chains",
        "mc_samples",
        "noise_level",
        "batch_size",
        "trials",
        "seed",
        "device",
    }
)

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
        assert isinstance(input_data, ScoringInput)

        # -- Host-side validation (fail fast before container launch) --------
        self._validate_inputs(input_data)

        # -- Copy input structure into workspace ----------------------------
        src_path = input_data.structure_path
        dest_name = src_path.name
        shutil.copy2(src_path, workspace.inputs_dir / dest_name)
        container_structure_path = f"/workspace/inputs/{dest_name}"

        # -- Build config.json ----------------------------------------------
        mutations: list[str] = input_data.extra.get("mutations", [])
        config: dict[str, Any] = {
            "pdb_path": container_structure_path,
            "mutations": ",".join(mutations),
            "chains": input_data.extra["chains"],
            "checkpoint_path": _DEFAULT_CHECKPOINT,
            "output_dir": "/workspace/outputs/raw",
            "mc_samples": input_data.extra.get("mc_samples", 20),
            "noise_level": input_data.extra.get("noise_level", 0.1),
            "batch_size": input_data.extra.get("batch_size", 10000),
            "trials": input_data.extra.get("trials", 1),
            "seed": input_data.extra.get("seed", 0),
            "device": input_data.extra.get("device", "auto"),
        }

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
    def _validate_inputs(input_data: ScoringInput) -> None:
        """Host-side validation — catch errors before container launch."""
        if not input_data.structure_path.exists():
            raise AutobioError(f"Input structure file does not exist: {input_data.structure_path}")

        # Mutations are required
        mutations = input_data.extra.get("mutations")
        if not mutations:
            raise AutobioError(
                "StaB-ddG requires 'mutations' in the extra dict. "
                "Provide a list of mutation strings in StaB-ddG format: "
                "[WT_AA][ChainID][Resnum][Mut_AA] (e.g., ['YH103H', 'QD30V'])."
            )
        if not isinstance(mutations, list) or not all(isinstance(m, str) for m in mutations):
            raise AutobioError(
                f"'mutations' must be a list of strings, got {type(mutations).__name__}. "
                "Each mutation should be in StaB-ddG format: "
                "[WT_AA][ChainID][Resnum][Mut_AA] (e.g., 'YH103H')."
            )

        # Chains specification is required
        chains = input_data.extra.get("chains")
        if not chains:
            raise AutobioError(
                "StaB-ddG requires 'chains' in the extra dict. "
                "Provide a string in 'binder1_binder2' format (e.g., 'ABC_DE')."
            )
        if not isinstance(chains, str) or chains.count("_") != 1:
            raise AutobioError(
                f"'chains' must be a string with exactly one underscore separator "
                f"(e.g., 'ABC_DE'), got {chains!r}."
            )


# ---------------------------------------------------------------------------
# Registry entry — populated when this module is imported
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

_STABDDG_INPUT_FORMAT = (
    "Provide a protein complex PDB via structure_path. Specify "
    "extra['mutations'] as a list of mutation strings in StaB-ddG format "
    "(e.g., ['YH103H', 'QD30V']) and extra['chains'] as a string "
    "defining the binding interface (e.g., 'ABC_DE').",
)

TOOL_REGISTRY["stabddg"] = ToolEntry(
    image_tag="stabddg:1.0.0",
    category=ToolCategory.SCORING,
    requires_gpu=True,
    gpu_count=1,
    input_schema=ScoringInput,
    output_schema=ScoringOutput,
    default_timeout=600,
    supports_batch=False,
    description=(
        "Predict binding ddG at protein-protein interfaces using StaB-ddG, "
        "an ML method based on the ProteinMPNN architecture. Faster than "
        "physics-based methods (seconds vs. hours) with competitive accuracy. "
        "Returns ddG in kcal/mol for mutations in protein-protein complexes."
    ),
    version="1.0.0",
    notes=_STABDDG_NOTES,
    input_format=_STABDDG_INPUT_FORMAT,
)
