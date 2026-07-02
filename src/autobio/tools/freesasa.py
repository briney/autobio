"""FreeSASA tool — SASA and buried surface area calculation.

A single catalog Tool, ``freesasa``, exposing two Modes sharing the
``FreeSASARunner`` runner class:

- ``sasa`` (default) — Solvent-accessible surface area (SASA) of a structure.
- ``bsa``  — Buried surface area (BSA) at a protein–protein interface.

Both are CPU-only, no GPU or model weights required.  BSA is computed as:

    BSA = SASA(partner1 alone) + SASA(partner2 alone) − SASA(complex)

Reference:
    Mitternacht. "FreeSASA: An open source C library for solvent accessible
    surface area calculations" F1000Research 5:189 (2016).
    DOI: 10.12688/f1000research.7931.1
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
    FreeSASABaseInput,
    FreeSASABSAInput,
    FreeSASASASAInput,
    ScoredStructure,
    ScoringOutput,
)
from autobio.tools.base import ToolRunner

if TYPE_CHECKING:
    from autobio.core.workspace import Workspace

# All formerly-consumed keys are now typed fields; nothing is stripped from extra.
_CONSUMED_EXTRA_KEYS: frozenset[str] = frozenset()

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class FreeSASARunner(ToolRunner):
    """Runner for FreeSASA SASA and BSA calculations.

    ``prepare_workspace`` validates inputs on the host side, copies the input
    structure into the workspace, and writes ``config.json`` with chain
    partner specifications (BSA only), algorithm choice, and probe radius.

    ``parse_output`` reads the standardised ``result_data.json`` produced by
    the container's ``standardize.py`` script.
    """

    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json and copy input structure into the workspace."""
        assert isinstance(input_data, FreeSASABaseInput)
        assert self.current_mode is not None
        is_bsa = self.current_mode.name == "bsa"

        self._validate_inputs(input_data, is_bsa=is_bsa)

        src_path = input_data.structure_path
        dest_name = src_path.name
        shutil.copy2(src_path, workspace.inputs_dir / dest_name)
        container_structure_path = f"/workspace/inputs/{dest_name}"

        config: dict[str, Any] = {
            "mode": "bsa" if is_bsa else "sasa",
            "structure_path": container_structure_path,
            "algorithm": input_data.algorithm,
            "probe_radius": input_data.probe_radius,
            "per_residue": input_data.per_residue,
            "output_dir": "/workspace/outputs/raw",
        }

        if is_bsa:
            assert isinstance(input_data, FreeSASABSAInput)
            config["partner1"] = input_data.partner1
            config["partner2"] = input_data.partner2

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
    def _validate_inputs(input_data: FreeSASABaseInput, *, is_bsa: bool) -> None:
        """Host-side validation — catch errors before container launch."""
        if not input_data.structure_path.exists():
            raise AutobioError(f"Input structure file does not exist: {input_data.structure_path}")

        suffix = input_data.structure_path.suffix.lower()
        if suffix != ".pdb":
            raise AutobioError(
                f"FreeSASA only supports PDB format, got '{suffix}'. "
                "Convert mmCIF/other formats to PDB before using FreeSASA."
            )

        if is_bsa:
            assert isinstance(input_data, FreeSASABSAInput)
            p1_chains = {c.strip() for c in input_data.partner1.split(",")}
            p2_chains = {c.strip() for c in input_data.partner2.split(",")}
            if not p1_chains or any(c == "" for c in p1_chains):
                raise AutobioError("partner1 contains empty chain IDs.")
            if not p2_chains or any(c == "" for c in p2_chains):
                raise AutobioError("partner2 contains empty chain IDs.")
            overlap = p1_chains & p2_chains
            if overlap:
                raise AutobioError(
                    f"partner1 and partner2 chains must not overlap. "
                    f"Overlapping chains: {', '.join(sorted(overlap))}."
                )


# ---------------------------------------------------------------------------
# Catalog registration — populated when this module is imported
# ---------------------------------------------------------------------------

_BSA_NOTES = (
    "Calculates the buried surface area (BSA) at a protein-protein interface "
    "using FreeSASA. BSA is the surface area that becomes inaccessible to "
    "solvent upon complex formation: BSA = SASA(partner1) + SASA(partner2) "
    "- SASA(complex).",
    "Requires explicit chain partner specification via 'partner1' and "
    "'partner2' as comma-separated chain IDs (e.g., partner1='A,B', "
    "partner2='C').",
    "Output includes total BSA, polar/apolar BSA breakdown, per-chain SASA, "
    "and optionally per-residue BSA (per_residue=True).",
    "Supports two algorithms: 'LeeRichards' (default) and 'ShrakeRupley'. "
    "Probe radius defaults to 1.4 angstrom (water molecule).",
    "CPU-only — no GPU required. Typical runtime is seconds.",
)

_SASA_NOTES = (
    "Calculates the solvent-accessible surface area (SASA) of a protein "
    "structure using FreeSASA. Returns total SASA with polar/apolar breakdown "
    "and per-chain SASA values.",
    "Optionally returns per-residue SASA via per_residue=True.",
    "Supports two algorithms: 'LeeRichards' (default) and 'ShrakeRupley'. "
    "Probe radius defaults to 1.4 angstrom (water molecule).",
    "CPU-only — no GPU required. Typical runtime is seconds.",
)

FREESASA_TOOL = Tool(
    name="freesasa",
    display_name="FreeSASA",
    category=ToolCategory.SCORING,
    description=(
        "Solvent-accessible surface area (SASA) and buried surface area (BSA) via "
        "FreeSASA. CPU-only, no GPU required."
    ),
    version="2.2.1",
    image_tag="freesasa:2.2.1",
    requires_gpu=False,
    gpu_count=0,
    default_mode="sasa",
    modes={
        "sasa": Mode(
            name="sasa",
            display_name="SASA",
            description="Solvent-accessible surface area of a structure.",
            input_schema=FreeSASASASAInput,
            output_schema=ScoringOutput,
            default_timeout=300,
            notes=_SASA_NOTES,
        ),
        "bsa": Mode(
            name="bsa",
            display_name="BSA",
            description="Buried surface area at a protein-protein interface.",
            input_schema=FreeSASABSAInput,
            output_schema=ScoringOutput,
            default_timeout=300,
            notes=_BSA_NOTES,
        ),
    },
    keywords=("sasa", "bsa", "surface area", "interface", "freesasa"),
)
"""The catalog Tool object for FreeSASA — exposed for tests that need to
re-register it after CATALOG-clearing fixtures (e.g. CLI isolation tests)."""

register(FREESASA_TOOL)
