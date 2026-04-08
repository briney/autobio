"""FreeSASA tool runners — SASA and buried surface area calculation.

Provides two tools sharing a single ``FreeSASARunner`` class:

- ``freesasa_sasa`` — Solvent-accessible surface area (SASA) of a structure.
- ``freesasa_bsa``  — Buried surface area (BSA) at a protein–protein interface.

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

from autobio.core.registry import TOOL_REGISTRY, ToolCategory, ToolEntry
from autobio.core.result import AutobioError
from autobio.schemas.base import BaseInput  # noqa: TC001 - needed at runtime for isinstance
from autobio.schemas.scoring import ScoredStructure, ScoringInput, ScoringOutput
from autobio.tools.base import ToolRunner

if TYPE_CHECKING:
    from autobio.core.workspace import Workspace

# Keys in ``extra`` that are consumed by the runner and should NOT be
# flat-merged into config.json.
_CONSUMED_EXTRA_KEYS = frozenset(
    {
        "partner1",
        "partner2",
        "algorithm",
        "probe_radius",
        "per_residue",
    }
)

_VALID_ALGORITHMS = frozenset({"LeeRichards", "ShrakeRupley"})

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
        assert isinstance(input_data, ScoringInput)

        is_bsa = self.tool_name == "freesasa_bsa"

        # -- Host-side validation (fail fast before container launch) --------
        self._validate_inputs(input_data, is_bsa=is_bsa)

        # -- Copy input structure into workspace ----------------------------
        src_path = input_data.structure_path
        dest_name = src_path.name
        shutil.copy2(src_path, workspace.inputs_dir / dest_name)
        container_structure_path = f"/workspace/inputs/{dest_name}"

        # -- Build config.json ----------------------------------------------
        config: dict[str, Any] = {
            "mode": "bsa" if is_bsa else "sasa",
            "structure_path": container_structure_path,
            "algorithm": input_data.extra.get("algorithm", "LeeRichards"),
            "probe_radius": input_data.extra.get("probe_radius", 1.4),
            "per_residue": input_data.extra.get("per_residue", False),
            "output_dir": "/workspace/outputs/raw",
        }

        if is_bsa:
            config["partner1"] = input_data.extra["partner1"]
            config["partner2"] = input_data.extra["partner2"]

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
    def _validate_inputs(input_data: ScoringInput, *, is_bsa: bool) -> None:
        """Host-side validation — catch errors before container launch."""
        if not input_data.structure_path.exists():
            raise AutobioError(f"Input structure file does not exist: {input_data.structure_path}")

        suffix = input_data.structure_path.suffix.lower()
        if suffix not in (".pdb",):
            raise AutobioError(
                f"FreeSASA only supports PDB format, got '{suffix}'. "
                "Convert mmCIF/other formats to PDB before using FreeSASA tools."
            )

        # Validate algorithm
        algorithm = input_data.extra.get("algorithm", "LeeRichards")
        if algorithm not in _VALID_ALGORITHMS:
            raise AutobioError(
                f"Invalid algorithm '{algorithm}'. "
                f"Must be one of: {', '.join(sorted(_VALID_ALGORITHMS))}."
            )

        # Validate probe_radius
        probe_radius = input_data.extra.get("probe_radius", 1.4)
        if not isinstance(probe_radius, (int, float)) or probe_radius <= 0:
            raise AutobioError(f"probe_radius must be a positive number, got {probe_radius!r}.")

        # BSA-specific: require partner chain specifications
        if is_bsa:
            partner1 = input_data.extra.get("partner1")
            partner2 = input_data.extra.get("partner2")

            if not partner1 or not isinstance(partner1, str):
                raise AutobioError(
                    "BSA calculation requires 'partner1' in the extra dict "
                    "(comma-separated chain IDs, e.g., 'A,B')."
                )
            if not partner2 or not isinstance(partner2, str):
                raise AutobioError(
                    "BSA calculation requires 'partner2' in the extra dict "
                    "(comma-separated chain IDs, e.g., 'C')."
                )

            p1_chains = {c.strip() for c in partner1.split(",")}
            p2_chains = {c.strip() for c in partner2.split(",")}

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
# Registry entries — populated when this module is imported
# ---------------------------------------------------------------------------

_BSA_NOTES = (
    "Calculates the buried surface area (BSA) at a protein-protein interface "
    "using FreeSASA. BSA is the surface area that becomes inaccessible to "
    "solvent upon complex formation: BSA = SASA(partner1) + SASA(partner2) "
    "- SASA(complex).",
    "Requires explicit chain partner specification via extra dict: "
    "extra['partner1'] and extra['partner2'] as comma-separated chain IDs "
    "(e.g., partner1='A,B', partner2='C').",
    "Output includes total BSA, polar/apolar BSA breakdown, per-chain SASA, "
    "and optionally per-residue BSA (extra['per_residue'] = True).",
    "Supports two algorithms: 'LeeRichards' (default) and 'ShrakeRupley'. "
    "Probe radius defaults to 1.4 angstrom (water molecule).",
    "CPU-only — no GPU required. Typical runtime is seconds.",
)

_BSA_INPUT_FORMAT = (
    "Provide a multi-chain PDB complex via structure_path. Required: "
    "extra['partner1'] and extra['partner2'] as comma-separated chain IDs "
    "(e.g., 'A,B' and 'C'). Optional: extra['algorithm'] ('LeeRichards' or "
    "'ShrakeRupley', default 'LeeRichards'), extra['probe_radius'] (default "
    "1.4), extra['per_residue'] (bool, default False).",
)

_SASA_NOTES = (
    "Calculates the solvent-accessible surface area (SASA) of a protein "
    "structure using FreeSASA. Returns total SASA with polar/apolar breakdown "
    "and per-chain SASA values.",
    "Optionally returns per-residue SASA via extra['per_residue'] = True.",
    "Supports two algorithms: 'LeeRichards' (default) and 'ShrakeRupley'. "
    "Probe radius defaults to 1.4 angstrom (water molecule).",
    "CPU-only — no GPU required. Typical runtime is seconds.",
)

_SASA_INPUT_FORMAT = (
    "Provide a PDB structure via structure_path. Optional: "
    "extra['algorithm'] ('LeeRichards' or 'ShrakeRupley', default "
    "'LeeRichards'), extra['probe_radius'] (default 1.4), "
    "extra['per_residue'] (bool, default False).",
)

TOOL_REGISTRY["freesasa_bsa"] = ToolEntry(
    image_tag="freesasa:2.2.1",
    category=ToolCategory.SCORING,
    requires_gpu=False,
    gpu_count=0,
    input_schema=ScoringInput,
    output_schema=ScoringOutput,
    default_timeout=300,
    supports_batch=False,
    description=(
        "Calculate buried surface area (BSA) at a protein-protein interface "
        "using FreeSASA. Measures the surface area buried upon complex "
        "formation, with polar/apolar breakdown and per-chain SASA. "
        "CPU-only, no GPU required."
    ),
    version="2.2.1",
    notes=_BSA_NOTES,
    input_format=_BSA_INPUT_FORMAT,
)

TOOL_REGISTRY["freesasa_sasa"] = ToolEntry(
    image_tag="freesasa:2.2.1",
    category=ToolCategory.SCORING,
    requires_gpu=False,
    gpu_count=0,
    input_schema=ScoringInput,
    output_schema=ScoringOutput,
    default_timeout=300,
    supports_batch=False,
    description=(
        "Calculate solvent-accessible surface area (SASA) of a protein "
        "structure using FreeSASA. Returns total SASA with polar/apolar "
        "breakdown and per-chain values. CPU-only, no GPU required."
    ),
    version="2.2.1",
    notes=_SASA_NOTES,
    input_format=_SASA_INPUT_FORMAT,
)
