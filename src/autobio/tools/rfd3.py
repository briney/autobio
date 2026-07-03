"""RFDiffusion3 tool runner.

Exposes RFDiffusion3 as a single-mode (``generate``) catalog Tool with the
full set of design options. The ``design_specs`` dict is passed through to
the container essentially verbatim — agents construct RFD3-native config
dicts directly, guided by the comprehensive ``notes`` on the ``generate`` mode.

CLI-level args (``diffusion_batch_size``, ``step_scale``, etc.) are passed
through the ``extra`` dict on ``RFD3Input``.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from autobio.core.catalog import Mode, Tool, register
from autobio.core.registry import ToolCategory
from autobio.core.result import AutobioError
from autobio.schemas.base import BaseInput  # noqa: TC001 - needed at runtime for isinstance
from autobio.schemas.structure_design import (
    DesignedStructure,
    RFD3Input,
    StructureDesignOutput,
)
from autobio.tools.base import ToolRunner

if TYPE_CHECKING:
    from autobio.core.workspace import Workspace

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class RFD3Runner(ToolRunner):
    """Runner for RFDiffusion3 generative structure design.

    ``prepare_workspace`` validates inputs on the host side, copies structure
    files into the workspace, rewrites ``"input"`` references in
    ``design_specs`` to container-internal paths, and writes ``config.json``.

    ``parse_output`` reads the standardised ``result_data.json`` produced by
    the container's ``standardize.py`` script.
    """

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json and copy input structures into the workspace."""
        assert isinstance(input_data, RFD3Input)

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
        config: dict[str, object] = {
            "design_specs": specs,
            "n_batches": input_data.n_batches,
            "out_dir": "/workspace/outputs/raw",
        }

        # Flat-merge extra dict for CLI-level args.
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
    def _validate_inputs(input_data: RFD3Input) -> None:
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

_RFD3_NOTES = (
    # Designability vs diversity
    "Key parameters for controlling designability vs diversity (pass via "
    "extra dict): 'step_scale' (default 1.5; higher=more designable, less "
    "diverse), 'gamma_0' (default 0.6; lower=more designable), "
    "'is_non_loopy' (per-spec; true biases against loops, improves "
    "designability). For maximum designability: step_scale=3.0, gamma_0=0.2, "
    "is_non_loopy=true. For maximum diversity: step_scale=1.0, gamma_0=1.0+.",
    # Multi-spec efficiency
    "Multiple design specifications in a single design_specs dict share one "
    "model load, which is substantially faster than separate runs. Use this "
    "for parameter sweeps or related design tasks.",
    # GPU memory
    "RFD3 requires substantial GPU memory. For designs >300 residues, consider "
    "setting 'low_memory_mode': true in the extra dict. 'diffusion_batch_size' "
    "(default 8) controls per-GPU parallelism — reduce for large designs to "
    "avoid OOM.",
    # Output format
    "Output structures are gzipped mmCIF files. The diffusion_metadata dict on "
    "each DesignedStructure contains residue_mapping (maps designed residue "
    "indices back to input positions) and timing information. Use "
    "residue_mapping to identify which residues were designed vs. kept from "
    "input when feeding designs into downstream tools like ProteinMPNN.",
    # CLI-level args via extra
    "CLI-level args passed via extra dict: 'diffusion_batch_size' (int, "
    "designs per batch), 'num_timesteps' (int, default 200), 'step_scale' "
    "(float, default 1.5), 'gamma_0' (float, default 0.6), "
    "'low_memory_mode' (bool), 'dump_trajectories' (bool), 'ckpt_path' "
    "(str, custom checkpoint), 'skip_existing' (bool, default true). "
    "Advanced sampler args: 'use_classifier_free_guidance' (bool), "
    "'cfg_scale' (float), 'cfg_features' (list), 'noise_scale' (float), "
    "'s_jitter_origin' (float), 'allow_realignment' (bool).",
    # Use-case recipes
    "Common use case patterns for design_specs entries: "
    "(1) Protein binder: set 'input' (target PDB), 'contig' (designed + target "
    "residues), 'select_hotspots' (target interface atoms), "
    "'infer_ori_strategy': 'hotspots', 'is_non_loopy': true. "
    "(2) Unconditioned design: set 'length' (e.g., '100' or '80-120'), no "
    "'input' needed. "
    "(3) Enzyme design: set 'input', 'ligand' (component codes), 'unindex' "
    "(catalytic residues to float), 'select_fixed_atoms' (key atoms). "
    "(4) Nucleic acid binder: set 'input', 'contig', 'select_hbond_donor' "
    "and 'select_hbond_acceptor' on nucleic acid residues. "
    "(5) Partial diffusion: set 'input', 'partial_t' (noise level in Å, "
    "recommended 5.0-15.0). "
    "(6) Symmetric design: add 'symmetry': {'id': 'C3'} (C or D groups).",
)

RFD3_TOOL = Tool(
    name="rfd3",
    display_name="RFdiffusion3",
    category=ToolCategory.STRUCTURE_DESIGN,
    description=(
        "Generate novel protein backbone structures using RFDiffusion3. "
        "Supports unconditioned design, protein binder design, enzyme active "
        "site design, nucleic acid binder design, partial diffusion, and "
        "symmetric design. Provide design specifications via the design_specs "
        "dict — each entry is a named design job with tool-native parameters."
    ),
    version="1.0.0",
    image_tag="rfd3:1.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="generate",
    modes={
        "generate": Mode(
            name="generate",
            display_name="Generate designs",
            description="Generate novel backbone designs from design specifications.",
            input_schema=RFD3Input,
            output_schema=StructureDesignOutput,
            default_timeout=3600,
            supports_batch=True,
            notes=_RFD3_NOTES,
        )
    },
    keywords=("rfd3", "rfdiffusion", "structure design", "protein design", "binder", "diffusion"),
)
"""Catalog Tool for RFDiffusion3 — exposed for tests re-registering after
CATALOG-clearing fixtures.
"""

register(RFD3_TOOL)
