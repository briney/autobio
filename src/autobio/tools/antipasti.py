"""ANTIPASTI tool runner — antibody binding affinity prediction.

Predicts antibody-antigen binding affinity (log10 Kd) from a 3D PDB
complex structure using ANTIPASTI. Computes Normal Mode Correlation Maps
(DCCM) via bio3d and runs a lightweight CNN to predict binding affinity.

Reference:
    Michalewicz et al. "ANTIPASTI: Interpretable prediction of antibody
    binding affinity exploiting normal modes and deep learning"
    Structure 32(12):2422-2434.e5 (2024). DOI: 10.1016/j.str.2024.10.001
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, Any

from autobio.core.catalog import Mode, Tool, register
from autobio.core.registry import ToolCategory
from autobio.core.result import AutobioError
from autobio.schemas.base import BaseInput  # noqa: TC001 - needed at runtime for isinstance
from autobio.schemas.binding_affinity import (
    AntipastiInput,
    BindingAffinityOutput,
    BindingAffinityPrediction,
)
from autobio.tools.base import ToolRunner

if TYPE_CHECKING:
    from autobio.core.workspace import Workspace

# ---------------------------------------------------------------------------
# Container-internal paths
# ---------------------------------------------------------------------------

_ANTIPASTI_DIR = "/app/antipasti"
_DEFAULT_CHECKPOINT = (
    f"{_ANTIPASTI_DIR}/checkpoints/full_ags_all_modes/"
    "model_epochs_1044_modes_all_pool_1_filters_4_size_4.pt"
)

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class AntipastiRunner(ToolRunner):
    """Runner for ANTIPASTI antibody binding affinity prediction.

    ``prepare_workspace`` validates inputs on the host side, copies the input
    structure into the workspace, and writes ``config.json`` with chain
    identifiers and model configuration.

    ``parse_output`` reads the standardised ``result_data.json`` produced by
    the container's ``standardize.py`` script.
    """

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json and copy input structure into the workspace."""
        assert isinstance(input_data, AntipastiInput)

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
            "heavy_chain": input_data.heavy_chain,
            "light_chain": input_data.light_chain,
            "antigen_chains": input_data.antigen_chains,
            "checkpoint_path": _DEFAULT_CHECKPOINT,
            "output_dir": "/workspace/outputs/raw",
            "antipasti_dir": _ANTIPASTI_DIR,
            "modes": input_data.modes,
        }

        self._apply_extra(config, input_data)

        workspace.write_config(config)

    def parse_output(self, workspace: Workspace) -> BindingAffinityOutput:
        """Read standardised outputs and return a ``BindingAffinityOutput``."""
        result_data_path = workspace.std_output_dir / "result_data.json"
        data = json.loads(result_data_path.read_text())

        predictions = []
        for p in data["predictions"]:
            predictions.append(
                BindingAffinityPrediction(
                    log10_kd=p["log10_kd"],
                    kd_molar=p.get("kd_molar"),
                    units=p.get("units"),
                    score_breakdown=p.get("score_breakdown"),
                )
            )

        # Placeholder metadata — overwritten by base class run()
        return BindingAffinityOutput(
            predictions=predictions,
            metadata=self._build_metadata(workspace, 0.0, [], ""),
            raw_output_path=workspace.raw_output_dir,
        )

    # -- Private helpers ----------------------------------------------------

    @staticmethod
    def _validate_inputs(input_data: AntipastiInput) -> None:
        """Host-side validation — catch errors before container launch."""
        if not input_data.structure_path.exists():
            raise AutobioError(f"Input structure file does not exist: {input_data.structure_path}")

        if not input_data.heavy_chain or not input_data.heavy_chain.strip():
            raise AutobioError("heavy_chain must be a non-empty string.")

        if not input_data.light_chain or not input_data.light_chain.strip():
            raise AutobioError("light_chain must be a non-empty string.")

        if not input_data.antigen_chains:
            raise AutobioError(
                "antigen_chains must be a non-empty list of chain IDs (e.g., ['A'] or ['A', 'B'])."
            )
        for chain_id in input_data.antigen_chains:
            if not chain_id or not chain_id.strip():
                raise AutobioError("Each antigen chain ID must be a non-empty string.")

        # Check for duplicate chain IDs
        all_chains = [
            input_data.heavy_chain,
            input_data.light_chain,
            *input_data.antigen_chains,
        ]
        if len(all_chains) != len(set(all_chains)):
            raise AutobioError(
                f"Duplicate chain IDs detected: {all_chains}. "
                "Heavy chain, light chain, and antigen chain(s) must all be distinct."
            )


# ---------------------------------------------------------------------------
# Catalog registration — populated when this module is imported
# ---------------------------------------------------------------------------

_ANTIPASTI_NOTES = (
    "Predicts antibody-antigen binding affinity as log10(Kd) in molar using "
    "ANTIPASTI, a CNN trained on Normal Mode Correlation Maps (DCCM) derived "
    "from antibody-antigen complex structures.",
    "Input is a PDB structure of an antibody-antigen complex with explicit "
    "chain assignments. The structure should ideally have Chothia numbering "
    "for best results, though the tool will attempt to process any PDB.",
    "The pipeline computes Normal Mode Analysis (NMA) via the R bio3d "
    "package, extracts DCCM maps, and runs a lightweight CNN to predict "
    "binding affinity. No GPU required.",
    "Output log10(Kd) values: more negative = tighter binding. For example, "
    "-9.0 corresponds to Kd ~1 nM (nanomolar), -6.0 to Kd ~1 µM (micromolar).",
    "Key parameter: modes (default 'all', or an integer "
    "for the number of normal modes to use in the DCCM calculation).",
)

ANTIPASTI_TOOL = Tool(
    name="antipasti",
    display_name="ANTIPASTI",
    category=ToolCategory.SCORING,
    description=(
        "Predict antibody-antigen binding affinity (log10 Kd) from a 3D PDB complex "
        "using ANTIPASTI (Normal Mode Correlation Maps + CNN). CPU-only."
    ),
    version="1.0.0",
    image_tag="antipasti:1.0.0",
    requires_gpu=False,
    gpu_count=0,
    default_mode="predict",
    modes={
        "predict": Mode(
            name="predict",
            display_name="Predict affinity",
            description="Predict antibody-antigen binding affinity (log10 Kd).",
            input_schema=AntipastiInput,
            output_schema=BindingAffinityOutput,
            default_timeout=1800,
            notes=_ANTIPASTI_NOTES,
        )
    },
    keywords=("antipasti", "binding affinity", "antibody", "antigen", "kd"),
)
"""Catalog Tool for ANTIPASTI — exposed for tests re-registering after CATALOG-clearing fixtures."""

register(ANTIPASTI_TOOL)
