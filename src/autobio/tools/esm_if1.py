"""ESM-IF1 inverse folding and sequence scoring tool runner.

ESM-IF1 (esm_if1_gvp4_t16_142M_UR50) is a 142M-parameter inverse folding
model from Facebook Research's ESM team. A single catalog Tool, ``esm_if1``,
exposes two Modes sharing the ``ESMIF1Runner`` runner class:

- ``design`` (default) — sequence design (inverse folding) via ``InverseFoldingInput``.
- ``score`` — sequence scoring via ``ScoringInput``.

The ``mode`` field in config.json (``"design"`` or ``"score"``) tells the
container which path to execute.

Fixed positions are enforced post-hoc by replacing sampled residues at
specified positions with the native residue. ESM-IF1 does not natively
support constrained sampling.

Tool-specific parameters can be passed through the ``extra`` dict:

- ``extra["seed"]``: random seed for reproducibility (design mode)
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

from autobio.core.catalog import Mode, Tool, register
from autobio.core.registry import ToolCategory
from autobio.schemas.base import BaseInput  # noqa: TC001 - needed at runtime for isinstance
from autobio.schemas.inverse_folding import (
    DesignedSequence,
    InverseFoldingInput,
    InverseFoldingOutput,
)
from autobio.schemas.scoring import ScoredStructure, ScoringInput, ScoringOutput
from autobio.tools.base import ToolRunner

if TYPE_CHECKING:
    from autobio.core.workspace import Workspace


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ESMIF1Runner(ToolRunner):
    """Runner for ESM-IF1 inverse folding (design mode) and scoring (score mode).

    ``prepare_workspace`` copies the input structure into the workspace and
    writes ``config.json`` — design mode maps ``InverseFoldingInput`` fields
    with ``mode="design"``; score mode maps ``ScoringInput`` fields with
    ``mode="score"``.

    ``parse_output`` reads the standardised ``result_data.json`` and returns
    an ``InverseFoldingOutput`` (design mode) or a ``ScoringOutput`` (score
    mode).
    """

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json and copy the input structure into the workspace."""
        assert self.current_mode is not None
        mode = self.current_mode.name

        assert isinstance(input_data, (InverseFoldingInput, ScoringInput))

        # Copy structure file into workspace inputs/
        src_path = input_data.structure_path
        dest_name = src_path.name
        shutil.copy2(src_path, workspace.inputs_dir / dest_name)

        if mode == "design":
            assert isinstance(input_data, InverseFoldingInput)

            # Build config.json for the container
            config: dict[str, object] = {
                "mode": "design",
                "structure_path": f"/workspace/inputs/{dest_name}",
                "num_sequences": input_data.num_sequences,
                "temperature": input_data.temperature,
            }

            if input_data.chains_to_design is not None:
                config["chains_to_design"] = input_data.chains_to_design

            if input_data.fixed_positions is not None:
                config["fixed_positions"] = input_data.fixed_positions
        else:
            assert self.current_mode.name == "score", self.current_mode.name
            assert isinstance(input_data, ScoringInput)

            # Build config.json for the container
            config = {
                "mode": "score",
                "structure_path": f"/workspace/inputs/{dest_name}",
                "sequences": input_data.sequences,
            }

        self._apply_extra(config, input_data)

        workspace.write_config(config)

    def parse_output(self, workspace: Workspace) -> InverseFoldingOutput | ScoringOutput:
        """Read standardised outputs and return the mode-appropriate output model."""
        assert self.current_mode is not None
        result_data_path = workspace.std_output_dir / "result_data.json"
        data = json.loads(result_data_path.read_text())

        if self.current_mode.name == "design":
            designed_sequences = [
                DesignedSequence(
                    rank=s["rank"],
                    sequence=s["sequence"],
                    score=s.get("score"),
                    recovery=s.get("recovery"),
                )
                for s in data["designed_sequences"]
            ]

            # Placeholder metadata — overwritten by base class run()
            return InverseFoldingOutput(
                designed_sequences=designed_sequences,
                native_sequence=data.get("native_sequence"),
                metadata=self._build_metadata(workspace, 0.0, [], ""),
                raw_output_path=workspace.raw_output_dir,
            )

        assert self.current_mode.name == "score", self.current_mode.name
        scores = [
            ScoredStructure(
                total_score=s["total_score"],
                per_residue_scores=s.get("per_residue_scores"),
                score_breakdown=s.get("score_breakdown"),
                units=s.get("units"),
                structure_path=s.get("structure_path"),
                ddg=s.get("ddg"),
                mutations=s.get("mutations"),
            )
            for s in data["scores"]
        ]

        # Placeholder metadata — overwritten by base class run()
        return ScoringOutput(
            scores=scores,
            metadata=self._build_metadata(workspace, 0.0, [], ""),
            raw_output_path=workspace.raw_output_dir,
        )


# ---------------------------------------------------------------------------
# Catalog registration — populated when this module is imported
# ---------------------------------------------------------------------------

_ESM_IF1_NOTES = (
    "ESM-IF1 does NOT natively support fixed_positions. When fixed_positions "
    "are specified, the container enforces them post-hoc by replacing sampled "
    "residues at those positions with the native residue. The model does not "
    "condition on the fixed residues during sampling. For native fixed-position "
    "support, use proteinmpnn or ligandmpnn.",
    "ESM-IF1 accepts PDB input only (not mmCIF). Convert mmCIF files to PDB format before use.",
    "The score field in design output is null. ESM-IF1 sampling does not return "
    "per-sequence log-likelihoods. Use the score mode to score designed sequences.",
    "Multi-chain inverse folding uses ESM-IF1's complex-aware sampling, which "
    "conditions on the full complex structure when designing each chain.",
)

_ESM_IF1_SCORE_NOTES = (
    "Computes conditional log-likelihood of a sequence given a backbone structure "
    "using ESM-IF1. Returns average negative log-likelihood (lower is better). "
    "The score_breakdown includes ll_fullseq (full sequence NLL) and "
    "ll_withcoord (NLL of residues with coordinates only).",
    "ESM-IF1 accepts PDB input only (not mmCIF). Convert mmCIF files to PDB format before use.",
    "The sequences field in ScoringInput is required — it must map chain IDs "
    "to amino acid sequences to score against the structure.",
)

ESM_IF1_TOOL = Tool(
    name="esm_if1",
    display_name="ESM-IF1",
    category=ToolCategory.INVERSE_FOLDING,
    description=(
        "ESM-IF1 (142M) inverse folding: design sequences for a backbone (design mode) "
        "or score sequences against a backbone by conditional log-likelihood (score mode)."
    ),
    version="1.0.0",
    image_tag="esm-if1:1.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="design",
    modes={
        "design": Mode(
            name="design",
            display_name="Design sequences",
            description="Design protein sequences for a backbone structure (inverse folding).",
            input_schema=InverseFoldingInput,
            output_schema=InverseFoldingOutput,
            default_timeout=600,
            notes=_ESM_IF1_NOTES,
        ),
        "score": Mode(
            name="score",
            display_name="Score sequences",
            description="Score sequences against a backbone (conditional log-likelihood).",
            input_schema=ScoringInput,
            output_schema=ScoringOutput,
            default_timeout=300,
            category=ToolCategory.SCORING,
            notes=_ESM_IF1_SCORE_NOTES,
        ),
    },
    keywords=("esm-if1", "inverse folding", "sequence design", "scoring", "log-likelihood"),
)
"""Catalog Tool for ESM-IF1 (design + score modes)."""

register(ESM_IF1_TOOL)
