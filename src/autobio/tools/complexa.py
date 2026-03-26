"""Proteina-Complexa binder design tool runners.

All three Proteina-Complexa variants — protein binder, ligand binder, and AME
(motif scaffolding) — share a single Docker image and runner class.  The
``tool_name`` (``"complexa"``, ``"complexa_ligand"``, or ``"complexa_ame"``)
determines which checkpoint and pipeline config are used.

Design specifications are provided via the ``design_specs`` dict on
``StructureDesignInput``.  Each entry describes one target: a PDB file, chain/
residue specification, hotspot residues, and binder length constraints.  Input
structure files are listed in ``input_structures`` so the runner can copy them
into the workspace and rewrite paths.

Generation-level parameters (``batch_size``, ``search_algorithm``, ``seed``,
etc.) are passed through the ``extra`` dict.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from autobio.core.registry import TOOL_REGISTRY, ToolCategory, ToolEntry
from autobio.core.result import AutobioError
from autobio.schemas.base import BaseInput  # noqa: TC001 - needed at runtime for isinstance
from autobio.schemas.structure_design import (
    DesignedStructure,
    StructureDesignInput,
    StructureDesignOutput,
)
from autobio.tools.base import ToolRunner

if TYPE_CHECKING:
    from autobio.core.workspace import Workspace

# ---------------------------------------------------------------------------
# Model configuration — maps tool name to variant-specific settings
# ---------------------------------------------------------------------------

_WEIGHTS_DIR = "/app/proteina-complexa/ckpts"

_VARIANT_CONFIG: dict[str, dict[str, str]] = {
    "complexa": {
        "variant": "protein_binder",
        "pipeline_config": "search_binder_local_pipeline",
        "ckpt_name": "complexa.ckpt",
        "ae_ckpt_name": "complexa_ae.ckpt",
    },
    "complexa_ligand": {
        "variant": "ligand_binder",
        "pipeline_config": "search_ligand_binder_local_pipeline",
        "ckpt_name": "complexa_ligand.ckpt",
        "ae_ckpt_name": "complexa_ligand_ae.ckpt",
    },
    "complexa_ame": {
        "variant": "ame",
        "pipeline_config": "search_ame_local_pipeline",
        "ckpt_name": "complexa_ame.ckpt",
        "ae_ckpt_name": "complexa_ame_ae.ckpt",
    },
}

# Keys in ``extra`` that are consumed by the runner and should NOT be
# flat-merged into config.json.
_CONSUMED_EXTRA_KEYS = frozenset[str]()

# Per-spec keys that have known semantics in the runner.
_VALID_SPEC_KEYS = frozenset(
    {
        "input",
        "target_input",
        "hotspot_residues",
        "binder_length",
        "binder_center",
        "pdb_id",
        # Ligand binder fields
        "ligand_chain",  # backward compat alias for 'ligand'
        "ligand",  # 3-letter residue name of ligand (e.g., "BEN", "OQO")
        "ligand_only",  # design around ligand only (default True)
        "smiles",  # SMILES string for ligand
        "use_bonds_from_file",  # use bonds from PDB file (default True)
        # AME fields
        "motif_residues",
        "contig_atoms",  # per-residue atom spec for motif scaffolding
    }
)

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ComplexaRunner(ToolRunner):
    """Runner for Proteina-Complexa binder design.

    ``prepare_workspace`` validates inputs on the host side, copies target
    structure files into the workspace, rewrites ``"input"`` references in
    ``design_specs`` to container-internal paths, and writes ``config.json``.

    ``parse_output`` reads the standardised ``result_data.json`` produced by
    the container's ``standardize.py`` script.
    """

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json and copy input structures into the workspace."""
        assert isinstance(input_data, StructureDesignInput)

        # -- Host-side validation (fail fast before container launch) --------
        self._validate_inputs(input_data)

        # -- Copy input structure files into workspace -----------------------
        filename_map: dict[str, str] = {}  # original filename -> container path
        for src_path in input_data.input_structures:
            dest_name = src_path.name
            shutil.copy2(src_path, workspace.inputs_dir / dest_name)
            filename_map[dest_name] = f"/workspace/inputs/{dest_name}"

        # -- Rewrite "input" paths in design_specs to container paths --------
        specs = copy.deepcopy(input_data.design_specs)
        for spec_name, spec in specs.items():
            if "input" in spec:
                original = Path(spec["input"])
                fname = original.name
                if fname not in filename_map:
                    raise AutobioError(
                        f"Spec {spec_name!r} references input file {spec['input']!r} "
                        f"(filename: {fname!r}), but no matching file was found in "
                        f"input_structures. Provided files: "
                        f"{[p.name for p in input_data.input_structures]}"
                    )
                spec["input"] = filename_map[fname]

        # -- Resolve variant config from tool name ---------------------------
        variant_cfg = _VARIANT_CONFIG[self.tool_name]

        # -- Build config.json -----------------------------------------------
        config: dict[str, Any] = {
            "variant": variant_cfg["variant"],
            "pipeline_config": variant_cfg["pipeline_config"],
            "ckpt_name": variant_cfg["ckpt_name"],
            "ae_ckpt_name": variant_cfg["ae_ckpt_name"],
            "weights_dir": _WEIGHTS_DIR,
            "design_specs": specs,
            "n_batches": input_data.n_batches,
            "out_dir": "/workspace/outputs/raw",
        }

        # Flat-merge extra dict for generation-level args
        for key, value in input_data.extra.items():
            if key not in _CONSUMED_EXTRA_KEYS:
                config[key] = value

        workspace.write_config(config)

    def parse_output(self, workspace: Workspace) -> StructureDesignOutput:
        """Read standardised outputs and return a ``StructureDesignOutput``."""
        result_data_path = workspace.std_output_dir / "result_data.json"
        data = json.loads(result_data_path.read_text())

        designs = [
            DesignedStructure(
                spec_name=d["spec_name"],
                batch_index=d["batch_index"],
                design_index=d["design_index"],
                structure_path=self._resolve_container_path(d["structure_path"], workspace),
                diffusion_metadata=d.get("diffusion_metadata"),
                evaluation_metrics=d.get("evaluation_metrics"),
            )
            for d in data["designs"]
        ]

        spec_summary: dict[str, int] = data.get("spec_summary", {})
        if not spec_summary:
            spec_summary = {}
            for d in designs:
                spec_summary[d.spec_name] = spec_summary.get(d.spec_name, 0) + 1

        # Placeholder metadata — overwritten by base class run()
        return StructureDesignOutput(
            designs=designs,
            spec_summary=spec_summary,
            metadata=self._build_metadata(workspace, 0.0, [], ""),
            raw_output_path=workspace.raw_output_dir,
        )

    @staticmethod
    def _resolve_container_path(container_path_str: str, workspace: Workspace) -> Path:
        """Map a container-internal ``/workspace/...`` path to the host workspace."""
        container_path = Path(container_path_str)
        try:
            relative = container_path.relative_to("/workspace")
        except ValueError:
            return container_path
        return workspace.root / relative

    @staticmethod
    def _validate_inputs(input_data: StructureDesignInput) -> None:
        """Host-side validation — catch errors before container launch."""
        if not input_data.design_specs:
            raise AutobioError("design_specs must contain at least one specification.")

        for spec_name, spec in input_data.design_specs.items():
            if not isinstance(spec, dict):
                raise AutobioError(
                    f"design_specs[{spec_name!r}] must be a dict, got {type(spec).__name__}."
                )

        if input_data.n_batches < 1:
            raise AutobioError(f"n_batches must be at least 1, got {input_data.n_batches}.")

        for src_path in input_data.input_structures:
            if not src_path.exists():
                raise AutobioError(f"Input structure file does not exist: {src_path}")

        # Check that every "input" reference in specs maps to a provided file
        provided_names = {p.name for p in input_data.input_structures}
        for spec_name, spec in input_data.design_specs.items():
            if "input" in spec:
                fname = Path(spec["input"]).name
                if fname not in provided_names:
                    raise AutobioError(
                        f"Spec {spec_name!r} references input file {spec['input']!r} "
                        f"(filename: {fname!r}), but no matching file was found in "
                        f"input_structures. Provided files: {sorted(provided_names)}"
                    )


# ---------------------------------------------------------------------------
# Registry entries — populated when this module is imported
# ---------------------------------------------------------------------------

# --- Shared notes and input format documentation ---

_COMPLEXA_INPUT_FORMAT = (
    # Per-spec fields
    "Each entry in design_specs describes one binder design target. Required "
    "fields: 'input' (path to target PDB, must be listed in input_structures), "
    "'target_input' (chain and residue range, e.g., 'A1-115'). Recommended: "
    "'hotspot_residues' (list of target residues to contact, e.g., "
    "['A37', 'A39', 'A49', 'A98']), 'binder_length' ([min, max] residue "
    "count, e.g., [64, 155]). Pass 'mode': 'design' in the extra dict to "
    "run the full pipeline with evaluation (AF2, RF3, MPNN). Default mode "
    "is 'generate' (generation only).",
    # Hotspot format
    "Hotspot residues are specified as a list of strings, each being a chain "
    "letter followed by a residue number: ['A37', 'A39', 'A49']. These guide "
    "the model to design binders that contact specific target residues. "
    "Providing hotspots significantly improves binder quality.",
    # Target input format
    "The 'target_input' field specifies which chain and residues of the target "
    "PDB to use. Format: '<chain><start>-<end>' (e.g., 'A1-115'). This "
    "defines the binding surface that the designed binder should target.",
    # Binder length
    "The 'binder_length' field is a [min, max] list specifying the range of "
    "designed binder lengths in residues. The model samples uniformly from "
    "this range. Typical range: [50, 150] for mini-protein binders, "
    "[20, 40] for peptide binders.",
)

_COMPLEXA_NOTES = (
    # Design mode
    "Set 'mode' to 'design' in the extra dict to run the full pipeline: "
    "generate -> filter -> evaluate -> analyze. This uses AF2 and RF3 for "
    "structure prediction evaluation, MPNN for sequence design, and returns "
    "evaluation_metrics on each DesignedStructure. Requires the full "
    "container image (with community model weights). Default mode is "
    "'generate' (generation only, no evaluation).",
    # Search algorithms
    "By default, generation uses 'single-pass' search (no reward model "
    "needed). For higher quality at the cost of more compute, set "
    "'search_algorithm' to 'best-of-n' or 'beam-search' in the extra dict. "
    "In generate mode, 'best-of-n' and 'beam-search' will fall back to the "
    "built-in reward if no external reward model is available. In design "
    "mode, the full reward model (AF2/RF3) is automatically used.",
    # Generation parameters
    "Key generation parameters (pass via extra dict): 'batch_size' (int, "
    "default 16, samples per batch), 'n_samples_per_length' (int, default 4, "
    "samples at each length), 'binder_length_samples' (int, default 4, "
    "number of lengths to sample from the range), 'seed' (int, default 42). "
    "For design mode: 'eval_njobs' (int, default 1, parallel evaluation "
    "jobs), 'gen_njobs' (int, default 1, parallel generation jobs).",
    # GPU memory
    "Proteina-Complexa requires a GPU with at least 16 GB VRAM. For longer "
    "binders (>100 residues), 24+ GB is recommended. Reduce batch_size to "
    "avoid OOM errors. Design mode requires additional GPU memory for "
    "evaluation models (AF2, RF3).",
    # Output format
    "Output structures are PDB files containing the designed binder chain "
    "alongside the target. The diffusion_metadata dict on each "
    "DesignedStructure includes generation metrics (total_reward, sample_type) "
    "when available. In design mode, evaluation_metrics contains AF2 "
    "iPTM/pTM/pLDDT, RF3 scores, MPNN recovery metrics, and scRMSD.",
    # Multi-spec efficiency
    "Multiple design specifications run sequentially within one container. "
    "Each spec triggers a separate pass. For parallel execution "
    "across specs, submit separate autobio runs.",
    # Downstream workflow (generate mode)
    "In generate mode, binders should be validated downstream: use a "
    "structure prediction tool (e.g., boltz2, chai1, openfold3) to refold "
    "the binder-target complex, then score with proteinmpnn for sequence "
    "optimization. In design mode, this evaluation is done automatically.",
)

_COMPLEXA_LIGAND_INPUT_FORMAT = (
    # Per-spec fields
    "Each entry in design_specs describes one ligand-binding protein design "
    "target. Required fields: 'input' (path to protein-ligand complex PDB, "
    "must be listed in input_structures), 'target_input' (chain and residue "
    "range of the protein, e.g., 'A1-200'). Recommended: 'hotspot_residues' "
    "(residues near the ligand binding site), 'binder_length' ([min, max]), "
    "'ligand_chain' (chain ID of the ligand in the PDB).",
    # Ligand handling
    "The target PDB must contain both the protein and ligand. The model "
    "designs a binder that contacts the protein near the ligand binding site. "
    "Specify 'ligand_chain' to indicate which chain is the ligand (e.g., 'B').",
    # Same hotspot and length format as protein binder
    _COMPLEXA_INPUT_FORMAT[1],  # hotspot format
    _COMPLEXA_INPUT_FORMAT[3],  # binder length
)

_COMPLEXA_LIGAND_NOTES = (
    _COMPLEXA_NOTES[0],  # design mode
    _COMPLEXA_NOTES[1],  # search algorithms
    _COMPLEXA_NOTES[2],  # generation parameters
    _COMPLEXA_NOTES[3],  # GPU memory
    # Ligand-specific output note
    "Output structures are PDB files containing the designed binder, target "
    "protein, and ligand. The model is trained on protein-ligand complexes "
    "from the PLINDER database.",
    _COMPLEXA_NOTES[5],  # multi-spec efficiency
    _COMPLEXA_NOTES[6],  # downstream workflow
)

_COMPLEXA_AME_INPUT_FORMAT = (
    # Per-spec fields
    "Each entry in design_specs describes one motif scaffolding target. "
    "Required fields: 'input' (path to motif PDB, must be listed in "
    "input_structures), 'target_input' (chain and residue range of the motif "
    "to scaffold). Recommended: 'binder_length' ([min, max] for the scaffold "
    "protein), 'motif_residues' (specific residues to preserve).",
    # AME description
    "AME (Atomistic Motif Extension) designs a scaffold protein around a "
    "functional motif. The motif structure is provided as a PDB, and the "
    "model generates a complete protein that incorporates the motif with "
    "correct geometry. Useful for enzyme active site design and functional "
    "domain scaffolding.",
    _COMPLEXA_INPUT_FORMAT[2],  # target input format
    _COMPLEXA_INPUT_FORMAT[3],  # binder length
)

_COMPLEXA_AME_NOTES = (
    _COMPLEXA_NOTES[0],  # design mode
    _COMPLEXA_NOTES[1],  # search algorithms
    _COMPLEXA_NOTES[2],  # generation parameters
    _COMPLEXA_NOTES[3],  # GPU memory
    # AME-specific output note
    "Output structures are PDB files containing the designed scaffold with "
    "the motif embedded. The diffusion_metadata includes motif RMSD when "
    "available, indicating how well the designed scaffold preserves the "
    "original motif geometry.",
    _COMPLEXA_NOTES[5],  # multi-spec efficiency
    _COMPLEXA_NOTES[6],  # downstream workflow
)

TOOL_REGISTRY["complexa"] = ToolEntry(
    image_tag="complexa:2.0.0",
    category=ToolCategory.STRUCTURE_DESIGN,
    requires_gpu=True,
    gpu_count=1,
    input_schema=StructureDesignInput,
    output_schema=StructureDesignOutput,
    default_timeout=43200,
    supports_batch=True,
    description=(
        "Design novel protein binders for protein targets using "
        "Proteina-Complexa. Generates binder sequences and all-atom 3D "
        "structures simultaneously via flow-matching generative modeling. "
        "Supports generate-only mode (default) and full design pipeline "
        "mode with AF2/RF3/MPNN evaluation. Provide target structure, "
        "hotspot residues, and binder length constraints via design_specs."
    ),
    version="2.0.0",
    notes=_COMPLEXA_NOTES,
    input_format=_COMPLEXA_INPUT_FORMAT,
)

TOOL_REGISTRY["complexa_ligand"] = ToolEntry(
    image_tag="complexa:2.0.0",
    category=ToolCategory.STRUCTURE_DESIGN,
    requires_gpu=True,
    gpu_count=1,
    input_schema=StructureDesignInput,
    output_schema=StructureDesignOutput,
    default_timeout=43200,
    supports_batch=True,
    description=(
        "Design novel protein binders for small-molecule ligand targets using "
        "Proteina-Complexa. Generates binder sequences and all-atom 3D "
        "structures for proteins that bind near a specified ligand. Supports "
        "generate-only mode (default) and full design pipeline mode with "
        "evaluation. Provide a protein-ligand complex PDB with hotspot "
        "residues and binder length constraints via design_specs."
    ),
    version="2.0.0",
    notes=_COMPLEXA_LIGAND_NOTES,
    input_format=_COMPLEXA_LIGAND_INPUT_FORMAT,
)

TOOL_REGISTRY["complexa_ame"] = ToolEntry(
    image_tag="complexa:2.0.0",
    category=ToolCategory.STRUCTURE_DESIGN,
    requires_gpu=True,
    gpu_count=1,
    input_schema=StructureDesignInput,
    output_schema=StructureDesignOutput,
    default_timeout=43200,
    supports_batch=True,
    description=(
        "Scaffold functional motifs into complete proteins using "
        "Proteina-Complexa AME (Atomistic Motif Extension). Designs a "
        "scaffold protein around a provided structural motif, preserving "
        "the motif geometry. Supports generate-only mode (default) and "
        "full design pipeline mode with evaluation. Useful for enzyme "
        "active site design and functional domain scaffolding."
    ),
    version="2.0.0",
    notes=_COMPLEXA_AME_NOTES,
    input_format=_COMPLEXA_AME_INPUT_FORMAT,
)
