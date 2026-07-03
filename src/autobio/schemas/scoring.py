"""Input/output schemas for structure scoring tools."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - Pydantic needs at runtime
from typing import Any, Literal

from pydantic import BaseModel, Field

from autobio.schemas.base import BaseInput, BaseOutput
from autobio.schemas.hints import Tier, Widget, ui


class ScoringInput(BaseInput):
    """Input schema for structure scoring tools (e.g., Rosetta, OpenMM energy)."""

    structure_path: Path = Field(description="Path to the structure to score (PDB or mmCIF).")
    sequences: dict[str, str] | None = Field(
        default=None,
        description=(
            "Optional mapping of chain ID to amino acid sequence. "
            "If provided, scores the structure with these sequences threaded onto the backbone."
        ),
    )


class FreeSASABaseInput(BaseInput):
    """Shared input for FreeSASA modes (SASA and BSA)."""

    structure_path: Path = Field(
        description="Path to the input PDB structure.",
        json_schema_extra=ui(widget=Widget.FILE, tier=Tier.PRIMARY, order=0),
    )
    algorithm: Literal["LeeRichards", "ShrakeRupley"] = Field(
        default="LeeRichards",
        description="SASA computation algorithm.",
        json_schema_extra=ui(widget=Widget.SELECT, tier=Tier.ADVANCED, order=10),
    )
    probe_radius: float = Field(
        default=1.4,
        gt=0,
        description="Solvent probe radius.",
        json_schema_extra=ui(
            widget=Widget.NUMBER, tier=Tier.ADVANCED, unit="Å", step=0.1, order=11
        ),
    )
    per_residue: bool = Field(
        default=False,
        description="Return per-residue values in addition to totals.",
        json_schema_extra=ui(widget=Widget.TOGGLE, tier=Tier.ADVANCED, order=12),
    )


class FreeSASASASAInput(FreeSASABaseInput):
    """Input for the FreeSASA ``sasa`` mode (solvent-accessible surface area)."""


class FreeSASABSAInput(FreeSASABaseInput):
    """Input for the FreeSASA ``bsa`` mode (buried surface area at an interface)."""

    partner1: str = Field(
        description="Comma-separated chain IDs for interface partner 1 (e.g. 'A,B').",
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=1),
    )
    partner2: str = Field(
        description="Comma-separated chain IDs for interface partner 2 (e.g. 'C').",
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=2),
    )


_ROSETTA_SCORE_FUNCTIONS = Literal[
    "ref2015", "ref2015_cart", "beta_nov16", "score12", "talaris2014", "franklin2019"
]


class RosettaBaseInput(BaseInput):
    """Shared input for Rosetta score/minimize modes."""

    structure_path: Path = Field(
        description="Path to the input structure (PDB or mmCIF).",
        json_schema_extra=ui(widget=Widget.FILE, tier=Tier.PRIMARY, order=0),
    )
    score_function: _ROSETTA_SCORE_FUNCTIONS = Field(
        default="ref2015",
        description="Rosetta energy score function.",
        json_schema_extra=ui(widget=Widget.SELECT, tier=Tier.ADVANCED, order=10),
    )
    nstruct: int = Field(
        default=1,
        ge=1,
        description="Number of output structures to generate.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=11),
    )


class RosettaRelaxInput(RosettaBaseInput):
    """Input for Rosetta relax mode (FastRelax; higher nstruct default)."""

    nstruct: int = Field(
        default=5,
        ge=1,
        description="Number of relaxed structures to generate.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=11),
    )


class RosettaFlexDdgInput(RosettaBaseInput):
    """Input for Rosetta flex-ddG interface-mutation DDG mode."""

    mutations: list[str] = Field(
        description=(
            "Mutations to score, e.g. ['A42F'] (original-residue-number-new; "
            "'A:42:F' for multi-letter chains)."
        ),
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=1),
    )
    chains_to_move: str = Field(
        description="Chain ID(s) of the binding partner to perturb at the interface (e.g. 'B').",
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=2),
    )
    nstruct: int = Field(
        default=35,
        ge=1,
        description="Number of independent backrub samples (use 3 for quick tests).",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=11),
    )
    resfile: str | None = Field(
        default=None,
        description=(
            "Raw Rosetta resfile content (power-user override of the generated mutation list)."
        ),
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.ADVANCED, order=12),
    )


class BAddGInput(BaseInput):
    """Input for BA-ddG binding-ddG prediction (single ``predict`` mode)."""

    structure_path: Path = Field(
        description="Path to the protein complex PDB structure.",
        json_schema_extra=ui(widget=Widget.FILE, tier=Tier.PRIMARY, order=0),
    )
    mutations: list[str] = Field(
        description=(
            "Mutations to score, format [WT_AA][ChainID][Resnum][Mut_AA] "
            "(e.g. ['YH103H', 'QD30V']); combined effect is predicted."
        ),
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=1),
    )
    chains: str = Field(
        description="Binding interface as 'binder1_binder2' (e.g. 'ABC_DE' = A,B,C vs D,E).",
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=2),
    )
    n_folds: int = Field(
        default=3,
        ge=1,
        le=3,
        description="Cross-validation folds to average (1-3).",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=10),
    )
    seed: int = Field(
        default=0,
        description="Random seed.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=11),
    )
    device: str = Field(
        default="auto",
        description="Compute device ('auto', 'cpu', or 'cuda').",
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.ADVANCED, order=12),
    )


class StaBddGInput(BaseInput):
    """Input for StaB-ddG binding-ddG prediction (single ``predict`` mode)."""

    structure_path: Path = Field(
        description="Path to the protein complex PDB structure.",
        json_schema_extra=ui(widget=Widget.FILE, tier=Tier.PRIMARY, order=0),
    )
    mutations: list[str] = Field(
        description=(
            "Mutations to score, StaB-ddG format [WT_AA][ChainID][Resnum][Mut_AA] "
            "(e.g. ['YH103H', 'QD30V'])."
        ),
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=1),
    )
    chains: str = Field(
        description="Binding interface as 'binder1_binder2' (e.g. 'ABC_DE').",
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=2),
    )
    mc_samples: int = Field(
        default=20,
        ge=1,
        description="Monte-Carlo samples for variance reduction.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=10),
    )
    noise_level: float = Field(
        default=0.1,
        description="Backbone perturbation noise level.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, step=0.05, order=11),
    )
    batch_size: int = Field(
        default=10000,
        ge=1,
        description="Batch size for scoring.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=12),
    )
    trials: int = Field(
        default=1,
        ge=1,
        description="Number of independent prediction trials.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=13),
    )
    seed: int = Field(
        default=0,
        description="Random seed.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=14),
    )
    device: str = Field(
        default="auto",
        description="Compute device ('auto', 'cpu', or 'cuda').",
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.ADVANCED, order=15),
    )


class LigandMPNNPackerInput(BaseInput):
    """Input for LigandMPNN sidechain-packing mutant building (single ``build_mutant`` mode)."""

    structure_path: Path = Field(
        description="Path to the input PDB structure.",
        json_schema_extra=ui(widget=Widget.FILE, tier=Tier.PRIMARY, order=0),
    )
    mutations: list[str] = Field(
        description=(
            "Mutations to introduce, format [WT_AA][ChainID][Resnum][Mut_AA] "
            "(e.g. ['EA63Q', 'KB42A']); applied simultaneously."
        ),
        json_schema_extra=ui(widget=Widget.TEXT, tier=Tier.PRIMARY, order=1),
    )
    num_packs: int = Field(
        default=4,
        ge=1,
        description="Number of packed structures to produce.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=10),
    )
    num_denoising_steps: int = Field(
        default=3,
        ge=1,
        description="Denoising steps during sidechain packing.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=11),
    )
    num_samples: int = Field(
        default=16,
        ge=1,
        description="Samples drawn per pack.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=12),
    )
    repack_everything: bool = Field(
        default=True,
        description="Repack all sidechains (not only mutated residues).",
        json_schema_extra=ui(widget=Widget.TOGGLE, tier=Tier.ADVANCED, order=13),
    )
    pack_with_ligand_context: bool = Field(
        default=True,
        description="Use bound ligands (HETATM) as context during packing.",
        json_schema_extra=ui(widget=Widget.TOGGLE, tier=Tier.ADVANCED, order=14),
    )


class ScoredStructure(BaseModel):
    """Scoring results for a single structure."""

    total_score: float = Field(description="Total energy score for the structure.")
    per_residue_scores: list[float] | None = Field(
        default=None,
        description="Per-residue energy scores in residue order.",
    )
    score_breakdown: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Breakdown of the total score by energy term "
            "(e.g., {'van_der_waals': -120.5, 'electrostatics': -45.2})."
        ),
    )
    units: str | None = Field(
        default=None,
        description="Units of the score (e.g., 'REU' for Rosetta energy units, 'kcal/mol').",
    )
    structure_path: Path | None = Field(
        default=None,
        description="Path to the scored/refined structure file, if the tool produces one.",
    )
    ddg: float | None = Field(
        default=None,
        description=(
            "Delta-delta-G: change in binding free energy upon mutation "
            "(kcal/mol or REU). Positive values indicate destabilization."
        ),
    )
    mutations: list[str] | None = Field(
        default=None,
        description="Mutations scored, e.g., ['A42F', 'L55W'].",
    )


class ScoringOutput(BaseOutput):
    """Output schema for structure scoring tools."""

    scores: list[ScoredStructure] = Field(
        description="Scoring results for each evaluated structure."
    )
