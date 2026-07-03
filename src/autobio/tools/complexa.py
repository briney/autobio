"""Proteina-Complexa binder design tool runner.

A single catalog Tool, ``complexa``, exposing three Modes sharing the
``ComplexaRunner`` runner class:

- ``protein_binder`` (default) — Design binders for a protein target.
- ``ligand_binder`` — Design binders for a small-molecule ligand target.
- ``ame`` — Scaffold a functional motif into a complete protein (Atomistic
  Motif Extension).

All three modes share a single Docker image; the mode determines which
checkpoint and pipeline config are used (``_MODE_CONFIG``).

Design specifications are provided via the ``design_specs`` dict on
``ComplexaInput``.  Each entry describes one target: a PDB file, chain/
residue specification, hotspot residues, and binder length constraints.  Input
structure files are listed in ``input_structures`` so the runner can copy them
into the workspace and rewrite paths.

Generation-level parameters (``batch_size``, ``search_algorithm``, ``seed``,
etc.) are passed through the ``extra`` dict.

Naming note: the catalog ``Mode`` concept (``protein_binder``/``ligand_binder``/
``ame`` — selected via ``ToolRunner.run(..., mode=...)`` -> ``self.current_mode``)
is unrelated to Complexa's own pipeline-depth switch, ``config["mode"]``
(``generate``/``design``). The latter flows through ``input_data.extra["mode"]``
and is consumed only container-side; no host code branches on it.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from autobio.core.catalog import Mode, Tool, register
from autobio.core.registry import ToolCategory
from autobio.core.result import AutobioError
from autobio.schemas.base import BaseInput  # noqa: TC001 - needed at runtime for isinstance
from autobio.schemas.structure_design import (
    ComplexaInput,
    DesignedStructure,
    StructureDesignOutput,
)
from autobio.tools.base import ToolRunner

if TYPE_CHECKING:
    from autobio.core.workspace import Workspace

# ---------------------------------------------------------------------------
# Model configuration — maps mode name to checkpoint/pipeline settings
# ---------------------------------------------------------------------------

_WEIGHTS_DIR = "/app/proteina-complexa/ckpts"

_MODE_CONFIG: dict[str, dict[str, str]] = {
    "protein_binder": {
        "pipeline_config": "search_binder_local_pipeline",
        "ckpt_name": "complexa.ckpt",
        "ae_ckpt_name": "complexa_ae.ckpt",
    },
    "ligand_binder": {
        "pipeline_config": "search_ligand_binder_local_pipeline",
        "ckpt_name": "complexa_ligand.ckpt",
        "ae_ckpt_name": "complexa_ligand_ae.ckpt",
    },
    "ame": {
        "pipeline_config": "search_ame_local_pipeline",
        "ckpt_name": "complexa_ame.ckpt",
        "ae_ckpt_name": "complexa_ame_ae.ckpt",
    },
}

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
        assert isinstance(input_data, ComplexaInput)
        assert self.current_mode is not None
        mode = self.current_mode.name
        mode_cfg = _MODE_CONFIG[mode]

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

        # -- Build config.json -----------------------------------------------
        config: dict[str, Any] = {
            "variant": mode,
            "pipeline_config": mode_cfg["pipeline_config"],
            "ckpt_name": mode_cfg["ckpt_name"],
            "ae_ckpt_name": mode_cfg["ae_ckpt_name"],
            "weights_dir": _WEIGHTS_DIR,
            "design_specs": specs,
            "n_batches": input_data.n_batches,
            "out_dir": "/workspace/outputs/raw",
        }

        self._apply_extra(config, input_data)

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
    def _validate_inputs(input_data: ComplexaInput) -> None:
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
# Catalog registration — populated when this module is imported
# ---------------------------------------------------------------------------

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

COMPLEXA_TOOL = Tool(
    name="complexa",
    display_name="Proteina-Complexa",
    category=ToolCategory.STRUCTURE_DESIGN,
    description=(
        "Design novel protein binders and scaffolds with Proteina-Complexa (flow-matching "
        "sequence+structure generation). Modes: protein_binder, ligand_binder, ame (atomistic "
        "motif extension). Provide targets/hotspots/length constraints via design_specs."
    ),
    version="2.0.0",
    image_tag="complexa:2.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="protein_binder",
    modes={
        "protein_binder": Mode(
            name="protein_binder",
            display_name="Protein binder",
            description="Design binders for a protein target.",
            input_schema=ComplexaInput,
            output_schema=StructureDesignOutput,
            default_timeout=43200,
            supports_batch=True,
            notes=_COMPLEXA_NOTES,
        ),
        "ligand_binder": Mode(
            name="ligand_binder",
            display_name="Ligand binder",
            description="Design binders for a small-molecule ligand target.",
            input_schema=ComplexaInput,
            output_schema=StructureDesignOutput,
            default_timeout=43200,
            supports_batch=True,
            notes=_COMPLEXA_LIGAND_NOTES,
        ),
        "ame": Mode(
            name="ame",
            display_name="AME (motif scaffolding)",
            description="Scaffold a functional motif into a complete protein.",
            input_schema=ComplexaInput,
            output_schema=StructureDesignOutput,
            default_timeout=43200,
            supports_batch=True,
            notes=_COMPLEXA_AME_NOTES,
        ),
    },
    keywords=("complexa", "proteina", "binder design", "scaffold", "ligand", "motif", "ame"),
)
"""Catalog Tool for Proteina-Complexa (protein_binder/ligand_binder/ame modes)."""

register(COMPLEXA_TOOL)
