"""Rosetta tool — structure scoring, refinement, and interface DDG prediction.

A single catalog Tool, ``rosetta``, exposing four Modes sharing the
``RosettaRunner`` runner class:

- ``score`` (default) — Score a structure with Rosetta's energy function.
- ``relax`` — FastRelax a structure (cycles of minimization + repacking).
- ``minimize`` — Local energy minimization of a structure.
- ``flexddg`` — Ensemble-based interface DDG prediction using backrub sampling.

Unlike most catalog Tools, each Rosetta mode ships as its own Docker image
(a thin layer on top of the shared ``autobio-rosetta-base`` image) — see the
per-mode ``image_tag`` overrides on the ``ROSETTA_TOOL`` modes below.

Rosetta tools are **CPU-only** — no GPU is required.
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, Any

from autobio.core.catalog import Mode, Tool, register
from autobio.core.registry import ToolCategory
from autobio.core.result import AutobioError
from autobio.schemas.base import BaseInput  # noqa: TC001 - needed at runtime for isinstance
from autobio.schemas.scoring import (
    RosettaBaseInput,
    RosettaFlexDdgInput,
    RosettaRelaxInput,
    ScoredStructure,
    ScoringOutput,
)
from autobio.tools.base import ToolRunner

if TYPE_CHECKING:
    from autobio.core.workspace import Workspace

# ---------------------------------------------------------------------------
# Container-internal paths
# ---------------------------------------------------------------------------

_ROSETTA_BIN = "/app/rosetta/bin"
_ROSETTA_DB = "/usr/local/lib/python3.8/dist-packages/pyrosetta/database"

# ---------------------------------------------------------------------------
# Mode configuration — maps mode name to protocol-specific settings
# ---------------------------------------------------------------------------

_MODE_CONFIG: dict[str, dict[str, str]] = {
    "score": {"binary": "score_jd2", "protocol": "score"},
    "relax": {
        "binary": "rosetta_scripts",
        "protocol": "relax",
        "xml_path": "/opt/tool/xml/relax.xml",
    },
    "minimize": {
        "binary": "rosetta_scripts",
        "protocol": "minimize",
        "xml_path": "/opt/tool/xml/minimize.xml",
    },
    "flexddg": {"binary": "rosetta_scripts", "protocol": "flexddg"},
}

# Mutation format: single-letter original + residue number + single-letter new
# e.g., "A42F" means Ala-42 -> Phe
_MUTATION_PATTERN_HELP = (
    "Mutations must be strings like 'A42F' (original-residue-number-new). "
    "Multi-letter chains can use 'A:42:F' format."
)

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class RosettaRunner(ToolRunner):
    """Runner for the Rosetta score/relax/minimize/flexddg modes.

    ``prepare_workspace`` validates inputs on the host side, copies the input
    structure into the workspace, generates mutation files for flex-ddG, and
    writes ``config.json``.

    ``parse_output`` reads the standardised ``result_data.json`` produced by
    the container's ``standardize.py`` script.
    """

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json and copy input structure into the workspace."""
        assert isinstance(input_data, RosettaBaseInput)
        assert self.current_mode is not None
        mode = self.current_mode.name
        mode_cfg = _MODE_CONFIG[mode]

        # -- Host-side validation (fail fast before container launch) --------
        self._validate_inputs(input_data)

        # -- Copy input structure into workspace ----------------------------
        src_path = input_data.structure_path
        dest_name = src_path.name
        shutil.copy2(src_path, workspace.inputs_dir / dest_name)
        container_structure_path = f"/workspace/inputs/{dest_name}"

        # -- Build config.json ----------------------------------------------
        config: dict[str, Any] = {
            "binary": mode_cfg["binary"],
            "protocol": mode_cfg["protocol"],
            "structure_path": container_structure_path,
            "database_path": _ROSETTA_DB,
            "score_function": input_data.score_function,
            "out_dir": "/workspace/outputs/raw",
        }
        config["nstruct"] = input_data.nstruct

        # XML protocol path (for rosetta_scripts-based modes)
        if "xml_path" in mode_cfg:
            config["xml_path"] = mode_cfg["xml_path"]

        # flex-ddG: mutation handling
        if mode == "flexddg":
            assert isinstance(input_data, RosettaFlexDdgInput)
            self._prepare_mutations(input_data, workspace, config)

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

    def _validate_inputs(self, input_data: RosettaBaseInput) -> None:
        """Host-side validation — catch errors before container launch."""
        if not input_data.structure_path.exists():
            raise AutobioError(f"Input structure file does not exist: {input_data.structure_path}")

        assert self.current_mode is not None
        if self.current_mode.name == "flexddg":
            assert isinstance(input_data, RosettaFlexDdgInput)
            if not input_data.mutations:
                raise AutobioError(
                    f"Rosetta flex-ddG requires at least one mutation. {_MUTATION_PATTERN_HELP}"
                )
            if not input_data.chains_to_move:
                raise AutobioError(
                    "Tool 'rosetta' (mode 'flexddg') requires 'chains_to_move' "
                    "(e.g., 'B' for the partner chain at the interface)."
                )

    @staticmethod
    def _prepare_mutations(
        input_data: RosettaFlexDdgInput,
        workspace: Workspace,
        config: dict[str, Any],
    ) -> None:
        """Generate mutation specification files for flex-ddG."""
        config["mutations"] = input_data.mutations
        config["chains_to_move"] = input_data.chains_to_move

        # Write a resfile for flex-ddG, or fall back to the raw mutation list.
        # The container's run.sh reads these from inputs/.
        if input_data.resfile:
            # Power-user escape hatch: raw resfile content
            (workspace.inputs_dir / "mutations.resfile").write_text(input_data.resfile)
            config["resfile_path"] = "/workspace/inputs/mutations.resfile"
        else:
            # Generate from mutations list — container handles format conversion
            config["mutation_list"] = input_data.mutations


# ---------------------------------------------------------------------------
# Catalog registration — populated when this module is imported
# ---------------------------------------------------------------------------

_SCORE_NOTES = (
    "Scores a PDB structure using Rosetta's energy function. The default "
    "score function is ref2015 (the recommended modern score function). "
    "Override via score_function. Supported: ref2015, ref2015_cart, "
    "beta_nov16, score12, talaris2014, franklin2019.",
    "Output includes total_score in REU (Rosetta Energy Units) and a "
    "score_breakdown with all individual energy terms (fa_atr, fa_rep, "
    "fa_sol, fa_elec, hbond terms, etc.).",
    "For expanded rotamer sampling (more accurate but slower), set "
    "extra['ex1'] = True and/or extra['ex2'] = True.",
)

_RELAX_NOTES = (
    "FastRelax cycles through minimization and side-chain repacking to "
    "find a low-energy conformation near the input structure. Produces "
    "both scored and refined PDB output.",
    "Set nstruct to generate multiple relaxed structures (default 5). "
    "The lowest-energy structure is typically the best starting point for "
    "downstream analysis.",
    "Default score function is ref2015. Override with score_function.",
)

_MINIMIZE_NOTES = (
    "Gradient-based energy minimization of a structure. Faster than relax "
    "but explores less conformational space — useful for resolving clashes "
    "or refining after mutations without full repacking.",
    "Default score function is ref2015. Override with score_function.",
)

_FLEXDDG_NOTES = (
    "Ensemble-based DDG prediction using backrub conformational sampling. "
    "More accurate than ddg_monomer for interface mutations because it "
    "accounts for backbone flexibility.",
    "Requires chains_to_move — the chain ID(s) of the binding partner "
    "(e.g., 'B'). Mutations are specified in the mutations field as a "
    "list of strings like ['A42F'].",
    "Key parameters (via extra dict): 'backrub_trials' (default 35000), "
    "'max_minimization_iter' (default 5000). nstruct is the number of "
    "independent backrub samples (default 35 — use 3 for quick tests).",
    "The protocol runs multiple independent backrub trajectories and "
    "averages the DDG predictions across the ensemble for robust "
    "estimates. Runtime scales linearly with nstruct.",
    "Output includes per-sample DDG values and ensemble statistics "
    "(mean, standard deviation) in the score_breakdown.",
)

ROSETTA_TOOL = Tool(
    name="rosetta",
    display_name="Rosetta",
    category=ToolCategory.SCORING,
    description=(
        "Rosetta structure scoring, refinement, and interface DDG prediction. Modes: "
        "score (energy), relax (FastRelax), minimize (gradient minimization), and "
        "flexddg (flex-ddG interface mutation DDG)."
    ),
    version="1.0.0",
    image_tag="rosetta-score:1.0.0",
    requires_gpu=False,
    gpu_count=0,
    default_mode="score",
    modes={
        "score": Mode(
            name="score",
            display_name="Score",
            description="Score a structure with Rosetta's energy function.",
            input_schema=RosettaBaseInput,
            output_schema=ScoringOutput,
            default_timeout=600,
            image_tag="rosetta-score:1.0.0",
            notes=_SCORE_NOTES,
        ),
        "relax": Mode(
            name="relax",
            display_name="Relax",
            description="FastRelax a structure to a low-energy conformation.",
            input_schema=RosettaRelaxInput,
            output_schema=ScoringOutput,
            default_timeout=3600,
            image_tag="rosetta-relax:1.0.0",
            notes=_RELAX_NOTES,
        ),
        "minimize": Mode(
            name="minimize",
            display_name="Minimize",
            description="Gradient-based energy minimization of a structure.",
            input_schema=RosettaBaseInput,
            output_schema=ScoringOutput,
            default_timeout=1800,
            image_tag="rosetta-minimize:1.0.0",
            notes=_MINIMIZE_NOTES,
        ),
        "flexddg": Mode(
            name="flexddg",
            display_name="Flex-ddG",
            description="Predict interface binding DDG with backrub sampling.",
            input_schema=RosettaFlexDdgInput,
            output_schema=ScoringOutput,
            default_timeout=14400,
            image_tag="rosetta-flexddg:1.0.0",
            notes=_FLEXDDG_NOTES,
        ),
    },
    keywords=("rosetta", "scoring", "relax", "minimize", "flexddg", "ddg", "energy"),
)
"""Catalog Tool for Rosetta (score/relax/minimize/flexddg modes)."""

register(ROSETTA_TOOL)
