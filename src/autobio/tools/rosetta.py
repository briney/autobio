"""Rosetta tool runners — scoring, relaxation, minimization, and DDG prediction.

All Rosetta tools share a single ``RosettaRunner`` class that dispatches by
``tool_name`` using the ``_VARIANT_CONFIG`` dict.  Each tool maps to a
different Docker image (thin layer on top of the shared
``autobio-rosetta-base`` image).

Rosetta tools are **CPU-only** — no GPU is required.

Supported tools:

- ``rosetta_score`` — Score a structure with Rosetta energy function.
- ``rosetta_relax`` — FastRelax a structure (cycles of minimization + repacking).
- ``rosetta_minimize`` — Local energy minimization of a structure.
- ``rosetta_ddg_monomer`` — Predict stability change (DDG) upon point mutation.
- ``rosetta_flexddg`` — Ensemble-based DDG prediction using backrub sampling.
"""

from __future__ import annotations

import json
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

_ROSETTA_BIN = "/app/rosetta/bin"
_ROSETTA_DB = "/usr/local/lib/python3.8/dist-packages/pyrosetta/database"

# ---------------------------------------------------------------------------
# Variant configuration — maps tool name to protocol-specific settings
# ---------------------------------------------------------------------------

_VARIANT_CONFIG: dict[str, dict[str, Any]] = {
    "rosetta_score": {
        "binary": "score_jd2",
        "protocol": "score",
        "uses_xml": False,
        "produces_structure": False,
        "requires_mutations": False,
        "default_nstruct": 1,
    },
    "rosetta_relax": {
        "binary": "rosetta_scripts",
        "protocol": "relax",
        "uses_xml": True,
        "xml_path": "/opt/tool/xml/relax.xml",
        "produces_structure": True,
        "requires_mutations": False,
        "default_nstruct": 5,
    },
    "rosetta_minimize": {
        "binary": "rosetta_scripts",
        "protocol": "minimize",
        "uses_xml": True,
        "xml_path": "/opt/tool/xml/minimize.xml",
        "produces_structure": True,
        "requires_mutations": False,
        "default_nstruct": 1,
    },
    "rosetta_flexddg": {
        "binary": "rosetta_scripts",
        "protocol": "flexddg",
        "uses_xml": False,
        "produces_structure": False,
        "requires_mutations": True,
        "default_nstruct": 35,
    },
}

# Keys in ``extra`` that are consumed by the runner and should NOT be
# flat-merged into config.json.
_CONSUMED_EXTRA_KEYS = frozenset(
    {"mutations", "chains", "chains_to_move", "resfile", "nstruct", "score_function"}
)

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
    """Runner for Rosetta computational biology tools.

    ``prepare_workspace`` validates inputs on the host side, copies the input
    structure into the workspace, generates mutation files for DDG tools, and
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
            "binary": variant_cfg["binary"],
            "protocol": variant_cfg["protocol"],
            "structure_path": container_structure_path,
            "database_path": _ROSETTA_DB,
            "score_function": input_data.extra.get("score_function", "ref2015"),
            "out_dir": "/workspace/outputs/raw",
        }

        # nstruct: from extra or variant default
        config["nstruct"] = input_data.extra.get("nstruct", variant_cfg["default_nstruct"])

        # XML protocol path (for rosetta_scripts-based tools)
        if variant_cfg["uses_xml"]:
            config["xml_path"] = variant_cfg["xml_path"]

        # DDG tools: mutation handling
        if variant_cfg["requires_mutations"]:
            self._prepare_mutations(input_data, workspace, config)

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

        variant_cfg = _VARIANT_CONFIG[self.tool_name]

        # DDG tools require mutations
        if variant_cfg["requires_mutations"]:
            mutations = input_data.extra.get("mutations")
            if not mutations:
                raise AutobioError(
                    f"Tool {self.tool_name!r} requires 'mutations' in the extra dict. "
                    f"{_MUTATION_PATTERN_HELP}"
                )
            if not isinstance(mutations, list) or not all(isinstance(m, str) for m in mutations):
                raise AutobioError(
                    f"'mutations' must be a list of strings, got {type(mutations).__name__}. "
                    f"{_MUTATION_PATTERN_HELP}"
                )

            # flexddg requires chains_to_move
            if self.tool_name == "rosetta_flexddg":
                chains = input_data.extra.get("chains_to_move") or input_data.extra.get("chains")
                if not chains:
                    raise AutobioError(
                        "Tool 'rosetta_flexddg' requires 'chains_to_move' in the extra "
                        "dict (e.g., 'B' for the partner chain at the interface)."
                    )

    @staticmethod
    def _prepare_mutations(
        input_data: ScoringInput,
        workspace: Workspace,
        config: dict[str, Any],
    ) -> None:
        """Generate mutation specification files for DDG tools."""
        mutations: list[str] = input_data.extra.get("mutations", [])
        config["mutations"] = mutations

        # chains_to_move for flexddg
        chains = input_data.extra.get("chains_to_move") or input_data.extra.get("chains")
        if chains:
            config["chains_to_move"] = chains

        # Write a resfile for flexddg or a .mut file for ddg_monomer
        # The container's run.sh reads these from inputs/
        resfile_content = input_data.extra.get("resfile")
        if resfile_content:
            # Power-user escape hatch: raw resfile content
            (workspace.inputs_dir / "mutations.resfile").write_text(resfile_content)
            config["resfile_path"] = "/workspace/inputs/mutations.resfile"
        else:
            # Generate from mutations list — container handles format conversion
            config["mutation_list"] = mutations


# ---------------------------------------------------------------------------
# Registry entries — populated when this module is imported
# ---------------------------------------------------------------------------

_SCORE_NOTES = (
    "Scores a PDB structure using Rosetta's energy function. The default "
    "score function is ref2015 (the recommended modern score function). "
    "Override via extra['score_function']. Supported: ref2015, ref2015_cart, "
    "beta_nov16, score12, talaris2014, franklin2019.",
    "Output includes total_score in REU (Rosetta Energy Units) and a "
    "score_breakdown with all individual energy terms (fa_atr, fa_rep, "
    "fa_sol, fa_elec, hbond terms, etc.).",
    "For expanded rotamer sampling (more accurate but slower), set "
    "extra['ex1'] = True and/or extra['ex2'] = True.",
)

_SCORE_INPUT_FORMAT = (
    "Provide a PDB or mmCIF file via structure_path. The structure should "
    "be clean (no missing backbone atoms). Rosetta handles hydrogens "
    "internally — pre-protonated structures are fine but not required.",
)

_RELAX_NOTES = (
    "FastRelax cycles through minimization and side-chain repacking to "
    "find a low-energy conformation near the input structure. Produces "
    "both scored and refined PDB output.",
    "Set extra['nstruct'] to generate multiple relaxed structures (default 5). "
    "The lowest-energy structure is typically the best starting point for "
    "downstream analysis.",
    "Default score function is ref2015. Override with extra['score_function'].",
)

_RELAX_INPUT_FORMAT = (
    "Provide a PDB or mmCIF file via structure_path. The structure will "
    "be relaxed using Rosetta's FastRelax protocol. Output includes the "
    "refined structure(s) and their scores.",
)

_MINIMIZE_NOTES = (
    "Gradient-based energy minimization of a structure. Faster than relax "
    "but explores less conformational space — useful for resolving clashes "
    "or refining after mutations without full repacking.",
    "Default score function is ref2015. Override with extra['score_function'].",
)

_MINIMIZE_INPUT_FORMAT = (
    "Provide a PDB or mmCIF file via structure_path. The structure will "
    "be minimized. Output includes the minimized structure and its score.",
)

_FLEXDDG_NOTES = (
    "Ensemble-based DDG prediction using backrub conformational sampling. "
    "More accurate than ddg_monomer for interface mutations because it "
    "accounts for backbone flexibility.",
    "Requires extra['chains_to_move'] — the chain ID(s) of the binding "
    "partner (e.g., 'B'). Mutations are specified in extra['mutations'] "
    "as a list of strings like ['A42F'].",
    "Key parameters (via extra dict): 'backrub_trials' (default 35000), "
    "'max_minimization_iter' (default 5000), 'nstruct' (number of "
    "independent backrub samples, default 35 — use 3 for quick tests).",
    "The protocol runs multiple independent backrub trajectories and "
    "averages the DDG predictions across the ensemble for robust "
    "estimates. Runtime scales linearly with nstruct.",
    "Output includes per-sample DDG values and ensemble statistics "
    "(mean, standard deviation) in the score_breakdown.",
)

_FLEXDDG_INPUT_FORMAT = (
    "Provide a protein complex PDB via structure_path. Specify "
    "extra['mutations'] (list of strings like ['A42F']) and "
    "extra['chains_to_move'] (partner chain ID, e.g., 'B'). "
    "The structure should be pre-relaxed with rosetta_relax.",
)

TOOL_REGISTRY["rosetta_score"] = ToolEntry(
    image_tag="rosetta-score:1.0.0",
    category=ToolCategory.SCORING,
    requires_gpu=False,
    gpu_count=0,
    input_schema=ScoringInput,
    output_schema=ScoringOutput,
    default_timeout=600,
    supports_batch=False,
    description=(
        "Score a protein structure using Rosetta's energy function. "
        "Returns total energy in REU (Rosetta Energy Units) with a "
        "per-term breakdown (van der Waals, electrostatics, hydrogen "
        "bonds, solvation, etc.). Useful for evaluating structure "
        "quality or comparing conformations."
    ),
    version="1.0.0",
    notes=_SCORE_NOTES,
    input_format=_SCORE_INPUT_FORMAT,
)

TOOL_REGISTRY["rosetta_relax"] = ToolEntry(
    image_tag="rosetta-relax:1.0.0",
    category=ToolCategory.SCORING,
    requires_gpu=False,
    gpu_count=0,
    input_schema=ScoringInput,
    output_schema=ScoringOutput,
    default_timeout=3600,
    supports_batch=False,
    description=(
        "Relax a protein structure using Rosetta's FastRelax protocol. "
        "Iteratively minimizes energy and repacks side chains, producing "
        "a low-energy conformation close to the input. Returns the "
        "refined structure and its score. Essential preprocessing step "
        "before DDG calculations or design."
    ),
    version="1.0.0",
    notes=_RELAX_NOTES,
    input_format=_RELAX_INPUT_FORMAT,
)

TOOL_REGISTRY["rosetta_minimize"] = ToolEntry(
    image_tag="rosetta-minimize:1.0.0",
    category=ToolCategory.SCORING,
    requires_gpu=False,
    gpu_count=0,
    input_schema=ScoringInput,
    output_schema=ScoringOutput,
    default_timeout=1800,
    supports_batch=False,
    description=(
        "Minimize a protein structure using gradient-based energy "
        "minimization. Faster than full relax — resolves clashes and "
        "refines geometry without side-chain repacking. Returns the "
        "minimized structure and its score."
    ),
    version="1.0.0",
    notes=_MINIMIZE_NOTES,
    input_format=_MINIMIZE_INPUT_FORMAT,
)

TOOL_REGISTRY["rosetta_flexddg"] = ToolEntry(
    image_tag="rosetta-flexddg:1.0.0",
    category=ToolCategory.SCORING,
    requires_gpu=False,
    gpu_count=0,
    input_schema=ScoringInput,
    output_schema=ScoringOutput,
    default_timeout=14400,
    supports_batch=False,
    description=(
        "Predict binding DDG at protein-protein interfaces using the "
        "flex-ddG protocol with backrub conformational sampling. More "
        "accurate than ddg_monomer for interface mutations. Runs an "
        "ensemble of backrub trajectories and averages DDG predictions. "
        "Requires a complex structure and chains_to_move specification."
    ),
    version="1.0.0",
    notes=_FLEXDDG_NOTES,
    input_format=_FLEXDDG_INPUT_FORMAT,
)
