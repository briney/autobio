"""AntiFold antibody inverse folding and sequence scoring tool runner.

AntiFold is an antibody-specific inverse folding model fine-tuned from
ESM-IF1, trained on solved and predicted antibody structures from SAbDab
and OAS. A single catalog Tool, ``antifold``, exposes two Modes sharing the
``AntiFoldRunner`` runner class:

- ``design`` (default) — antibody sequence design (inverse folding) via
  ``InverseFoldingInput``.
- ``score`` — sequence scoring via ``ScoringInput``.

The ``mode`` field in config.json (``"design"`` or ``"score"``) tells the
container which path to execute.

Unlike ESM-IF1, AntiFold uses antibody-specific parameters passed through
the ``extra`` dict:

- ``extra["heavy_chain"]``: heavy chain ID in the PDB (e.g. ``"H"``)
- ``extra["light_chain"]``: light chain ID in the PDB (e.g. ``"L"``)
- ``extra["antigen_chain"]``: optional antigen chain ID(s)
- ``extra["regions"]``: CDR/framework regions to target (e.g.
  ``["CDRH1", "CDRH3"]``); ``None`` means all regions

ANARCI renumbering to IMGT scheme is handled automatically inside the
container, so input PDBs do not need to be pre-numbered.
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING

from autobio.core.catalog import Mode, Tool, register
from autobio.core.registry import ToolCategory
from autobio.core.result import AutobioError
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


def _validate_chain_ids(input_data: BaseInput) -> None:
    """Validate that at least one antibody chain ID is provided in extra."""
    heavy = input_data.extra.get("heavy_chain")
    light = input_data.extra.get("light_chain")
    if not heavy and not light:
        raise AutobioError(
            "AntiFold requires at least one of extra['heavy_chain'] or "
            "extra['light_chain'] to identify antibody chains in the input PDB. "
            "Example: extra={'heavy_chain': 'H', 'light_chain': 'L'}"
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class AntiFoldRunner(ToolRunner):
    """Runner for AntiFold antibody inverse folding (design mode) and scoring (score mode).

    ``prepare_workspace`` validates antibody chain IDs (both modes), copies
    the input structure into the workspace, and writes ``config.json`` —
    design mode maps ``InverseFoldingInput`` fields with ``mode="design"``;
    score mode maps ``ScoringInput`` fields with ``mode="score"``.

    AntiFold uses antibody-specific parameters (``heavy_chain``,
    ``light_chain``, ``antigen_chain``, ``regions``) passed via the ``extra``
    dict rather than ``chains_to_design`` or ``fixed_positions``.

    ``parse_output`` reads the standardised ``result_data.json`` and returns
    an ``InverseFoldingOutput`` (design mode) or a ``ScoringOutput`` (score
    mode).
    """

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json and copy the input structure into the workspace."""
        assert self.current_mode is not None
        mode = self.current_mode.name

        _validate_chain_ids(input_data)

        if mode == "design":
            assert isinstance(input_data, InverseFoldingInput)

            # Copy structure file into workspace inputs/
            src_path = input_data.structure_path
            dest_name = src_path.name
            shutil.copy2(src_path, workspace.inputs_dir / dest_name)

            # Build config.json for the container
            # Note: chains_to_design and fixed_positions are intentionally NOT
            # mapped — AntiFold uses heavy_chain/light_chain and regions instead.
            config: dict[str, object] = {
                "mode": "design",
                "structure_path": f"/workspace/inputs/{dest_name}",
                "num_sequences": input_data.num_sequences,
                "temperature": input_data.temperature,
            }
        else:  # "score"
            assert isinstance(input_data, ScoringInput)

            # Copy structure file into workspace inputs/
            src_path = input_data.structure_path
            dest_name = src_path.name
            shutil.copy2(src_path, workspace.inputs_dir / dest_name)

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

_ANTIFOLD_NOTES = (
    "AntiFold requires antibody chain IDs via extra['heavy_chain'] and/or "
    "extra['light_chain']; at least one must be provided. Optional: "
    "extra['antigen_chain'] for antigen context.",
    "chains_to_design and fixed_positions from InverseFoldingInput are not "
    "used by AntiFold. Use extra['regions'] to target specific CDR/framework "
    "regions for redesign (e.g. ['CDRH1', 'CDRH3']). Valid region names: "
    "CDRH1, CDRH2, CDRH3, CDRL1, CDRL2, CDRL3, FWH1, FWH2, FWH3, FWH4, "
    "FWL1, FWL2, FWL3, FWL4. None (default) targets all regions.",
    "ANARCI renumbering to IMGT scheme is applied automatically inside the "
    "container. Input PDBs do not need to be pre-numbered.",
    "AntiFold accepts PDB input only (not mmCIF). Convert mmCIF files to PDB format before use.",
    "The score field in design output contains per-sequence log-likelihood "
    "(unlike ESM-IF1 where it is null). Lower scores indicate better fit.",
)

_ANTIFOLD_SCORE_NOTES = (
    "When sequences is None, scores the native sequence from the PDB. When "
    "provided, computes sequence-specific scores from the full logit matrix.",
    "Requires extra['heavy_chain'] and/or extra['light_chain'] to identify "
    "antibody chains. Optional: extra['antigen_chain'] for antigen context.",
    "Returns per-residue log-likelihoods and perplexity in score_breakdown. "
    "total_score is the mean negative log-likelihood (lower is better).",
    "AntiFold accepts PDB input only (not mmCIF). Convert mmCIF files to PDB format before use.",
)

ANTIFOLD_TOOL = Tool(
    name="antifold",
    display_name="AntiFold",
    category=ToolCategory.INVERSE_FOLDING,
    description=(
        "AntiFold antibody-specific inverse folding (fine-tuned from ESM-IF1): design "
        "antibody sequences for a backbone (design mode) or score sequences by "
        "conditional log-likelihood (score mode). Requires heavy/light chain IDs via extra."
    ),
    version="1.0.0",
    image_tag="antifold:1.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="design",
    modes={
        "design": Mode(
            name="design",
            display_name="Design sequences",
            description="Design antibody sequences for a backbone (targeting CDR/FW regions).",
            input_schema=InverseFoldingInput,
            output_schema=InverseFoldingOutput,
            default_timeout=600,
            notes=_ANTIFOLD_NOTES,
        ),
        "score": Mode(
            name="score",
            display_name="Score sequences",
            description="Score antibody sequences against a backbone (log-likelihood/perplexity).",
            input_schema=ScoringInput,
            output_schema=ScoringOutput,
            default_timeout=300,
            category=ToolCategory.SCORING,
            notes=_ANTIFOLD_SCORE_NOTES,
        ),
    },
    keywords=("antifold", "antibody", "inverse folding", "sequence design", "scoring"),
)
"""Catalog Tool for AntiFold (design + score modes)."""

register(ANTIFOLD_TOOL)
