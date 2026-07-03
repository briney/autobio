"""PRODIGY tool runner — protein-protein binding affinity prediction.

Predicts binding affinity (delta-G in kcal/mol, Kd in M) for protein-protein
complexes using PRODIGY, a contact-based predictor that counts interatomic
contacts at the interface and uses a linear model to estimate binding energy.

Reference:
    Xue, Rodrigues, Kastritis, Bonvin & Vangone. "PRODIGY: a web server
    for predicting the binding affinity of protein-protein complexes"
    Bioinformatics 32(23):3676-3678 (2016). DOI: 10.1093/bioinformatics/btw514
"""

from __future__ import annotations

import json
import shutil
from typing import TYPE_CHECKING, Any

from autobio.core.catalog import Mode, Tool, register
from autobio.core.registry import ToolCategory
from autobio.core.result import AutobioError
from autobio.schemas.base import BaseInput  # noqa: TC001 - needed at runtime for isinstance
from autobio.schemas.protein_binding_affinity import (
    ProdigyInput,
    ProteinBindingAffinityOutput,
    ProteinBindingAffinityPrediction,
)
from autobio.tools.base import ToolRunner

if TYPE_CHECKING:
    from autobio.core.workspace import Workspace

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class ProdigyRunner(ToolRunner):
    """Runner for PRODIGY protein-protein binding affinity prediction.

    ``prepare_workspace`` validates inputs on the host side, copies the input
    structure into the workspace, and writes ``config.json`` with chain
    selection, temperature, and analysis parameters.

    ``parse_output`` reads the standardised ``result_data.json`` produced by
    the container's ``standardize.py`` script.
    """

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json and copy input structure into the workspace."""
        assert isinstance(input_data, ProdigyInput)

        # -- Host-side validation (fail fast before container launch) --------
        self._validate_inputs(input_data)

        # -- Copy input structure into workspace ----------------------------
        src_path = input_data.structure_path
        dest_name = src_path.name
        shutil.copy2(src_path, workspace.inputs_dir / dest_name)
        container_structure_path = f"/workspace/inputs/{dest_name}"

        # -- Build config.json ----------------------------------------------
        config: dict[str, Any] = {
            "structure_path": container_structure_path,
            "selection": input_data.chain_selection,
            "temperature": input_data.temperature,
            "distance_cutoff": input_data.distance_cutoff,
            "contact_list": input_data.contact_list,
            "output_dir": "/workspace/outputs/raw",
        }

        self._apply_extra(config, input_data)

        workspace.write_config(config)

    def parse_output(self, workspace: Workspace) -> ProteinBindingAffinityOutput:
        """Read standardised outputs and return a ``ProteinBindingAffinityOutput``."""
        result_data_path = workspace.std_output_dir / "result_data.json"
        data = json.loads(result_data_path.read_text())

        predictions = []
        for p in data["predictions"]:
            predictions.append(
                ProteinBindingAffinityPrediction(
                    delta_g_kcal_mol=p["delta_g_kcal_mol"],
                    kd_molar=p.get("kd_molar"),
                    units=p.get("units"),
                    score_breakdown=p.get("score_breakdown"),
                )
            )

        # Placeholder metadata — overwritten by base class run()
        return ProteinBindingAffinityOutput(
            predictions=predictions,
            metadata=self._build_metadata(workspace, 0.0, [], ""),
            raw_output_path=workspace.raw_output_dir,
        )

    # -- Private helpers ----------------------------------------------------

    @staticmethod
    def _validate_inputs(input_data: ProdigyInput) -> None:
        """Host-side validation — catch errors before container launch."""
        if not input_data.structure_path.exists():
            raise AutobioError(f"Input structure file does not exist: {input_data.structure_path}")

        if input_data.temperature <= -273.15:
            raise AutobioError(
                f"Temperature must be above absolute zero (-273.15 °C), "
                f"got {input_data.temperature}."
            )

        if input_data.chain_selection is not None and not input_data.chain_selection.strip():
            raise AutobioError("chain_selection must be a non-empty string or None.")


# ---------------------------------------------------------------------------
# Catalog registration — populated when this module is imported
# ---------------------------------------------------------------------------

_PRODIGY_NOTES = (
    "Predicts protein-protein binding affinity (delta-G in kcal/mol and Kd in "
    "molar) using PRODIGY, a contact-based predictor. Counts interatomic "
    "contacts at the protein-protein interface and uses a linear model trained "
    "on experimental binding affinity data.",
    "Input is a PDB structure of a protein-protein complex. Chain selection "
    "specifies which chains form each binding partner (e.g., 'A B' for chain A "
    "vs chain B, or 'A,B C' to treat A+B as one partner against C). If omitted, "
    "all inter-chain contacts are used.",
    "PRODIGY classifies contacts by polar/apolar/charged character and "
    "incorporates Non-Interacting Surface (NIS) properties. The output includes "
    "predicted delta-G, Kd (temperature-dependent), and a full breakdown of "
    "contact counts and surface properties.",
    "CPU-only — no GPU required. Typical runtime is seconds for a single complex.",
    "Key parameters: distance_cutoff (default 5.5 angstrom), "
    "contact_list (boolean, default False — include detailed contact list).",
)

PRODIGY_TOOL = Tool(
    name="prodigy",
    display_name="PRODIGY",
    category=ToolCategory.SCORING,
    description=(
        "Predict protein-protein binding affinity (delta-G and Kd) from a 3D complex "
        "using PRODIGY (interatomic contact counting). CPU-only."
    ),
    version="2.4.0",
    image_tag="prodigy:2.4.0",
    requires_gpu=False,
    gpu_count=0,
    default_mode="predict",
    modes={
        "predict": Mode(
            name="predict",
            display_name="Predict affinity",
            description="Predict protein-protein binding affinity (delta-G, Kd).",
            input_schema=ProdigyInput,
            output_schema=ProteinBindingAffinityOutput,
            default_timeout=300,
            notes=_PRODIGY_NOTES,
        )
    },
    keywords=("prodigy", "binding affinity", "protein-protein", "delta-g", "kd"),
)
"""Catalog Tool for PRODIGY — exposed for tests re-registering after CATALOG-clearing fixtures."""

register(PRODIGY_TOOL)
