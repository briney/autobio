"""Input/output schemas for structure prediction tools."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - Pydantic needs at runtime
from typing import Any

from pydantic import BaseModel, Field

from autobio.schemas.base import BaseInput, BaseOutput
from autobio.schemas.hints import Tier, Widget, ui
from autobio.schemas.sequences import GenericSequenceSet  # noqa: TC001 - needed at runtime


class PredictedStructure(BaseModel):
    """A single predicted structure model with confidence metrics."""

    model_rank: int = Field(description="Rank by predicted quality (1 = best).")
    structure_path: Path = Field(
        description="Path to the predicted structure in outputs/standardized/."
    )
    plddt_per_residue: list[float] | None = Field(
        default=None,
        description="Per-residue pLDDT confidence scores (0-100 scale).",
    )
    plddt_mean: float | None = Field(
        default=None,
        description="Mean pLDDT score across all residues (0-100 scale).",
    )
    ptm: float | None = Field(
        default=None,
        description="Predicted TM-score (0-1 scale, global structure confidence).",
    )
    iptm: float | None = Field(
        default=None,
        description="Interface predicted TM-score (0-1 scale, multimer interface confidence).",
    )
    chain_mapping: dict[str, str] | None = Field(
        default=None,
        description="Mapping from input chain IDs to output chain IDs.",
    )
    affinity_probability: float | None = Field(
        default=None,
        description="Predicted probability of binding (0-1 scale).",
    )
    affinity_value: float | None = Field(
        default=None,
        description="Predicted binding affinity as log10(IC50) in uM.",
    )


class ConfidenceMetrics(BaseModel):
    """Summary confidence metrics across all predicted models."""

    best_plddt_mean: float | None = Field(
        default=None,
        description="Highest mean pLDDT across all models (0-100 scale).",
    )
    best_ptm: float | None = Field(
        default=None,
        description="Highest predicted TM-score across all models (0-1 scale).",
    )
    best_iptm: float | None = Field(
        default=None,
        description="Highest interface predicted TM-score across all models (0-1 scale).",
    )


class StructurePredictionOutput(BaseOutput):
    """Output schema for structure prediction tools."""

    structures: list[PredictedStructure] = Field(
        description="Predicted structures ranked by quality."
    )
    confidence: ConfidenceMetrics = Field(
        description="Summary confidence metrics across all predicted models."
    )


class ESMFoldInput(BaseInput):
    """Input for ESMFold single-sequence structure prediction (single ``predict`` mode)."""

    sequences: GenericSequenceSet = Field(
        description=(
            "A single protein sequence: a dict of id→sequence (one chain), "
            "FASTA text, or a FASTA file path."
        ),
        json_schema_extra=ui(widget=Widget.SEQUENCE, flavor="generic", tier=Tier.PRIMARY, order=0),
    )
    num_models: int = Field(
        default=1,
        description="Number of models. ESMFold is deterministic; must be 1.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=10),
    )
    templates: list[Path] | None = Field(
        default=None,
        description="Template structures. ESMFold does not use templates; must be None/empty.",
        json_schema_extra=ui(widget=Widget.FILE, tier=Tier.ADVANCED, order=11),
    )


class BoltzInput(BaseInput):
    """Input for Boltz-1 / Boltz-2 structure prediction (shared by both Tools)."""

    sequences: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Mapping of chain ID to sequence. Values may be protein/DNA/RNA; for "
            "ligand chains the value is ignored when SMILES/CCD is given via "
            "entity_types. May be empty only when boltz_yaml is provided."
        ),
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.PRIMARY, order=0),
    )
    num_models: int = Field(
        default=1,
        ge=1,
        description="Number of structures to generate (maps to diffusion_samples).",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=10),
    )
    templates: list[Path] | None = Field(
        default=None,
        description="Template structures (PDB/mmCIF) copied into the workspace.",
        json_schema_extra=ui(widget=Widget.FILE, tier=Tier.ADVANCED, order=11),
    )
    entity_types: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Per-chain entity type: 'protein'/'dna'/'rna', or a dict "
            "{'smiles': 'CC...'} / {'ccd': 'ATP'} for ligands. Default 'protein'. "
            "In the generated Boltz YAML, a chain entity may use a list for its "
            "id to specify multiple identical chains, e.g. "
            "{protein: {id: [A, B], sequence: MKLL...}} — use boltz_yaml to take "
            "advantage of this directly."
        ),
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.ADVANCED, order=12),
    )
    use_msa_server: bool = Field(
        default=True,
        description="Use ColabFold MMseqs2 MSA server (needs network). Set False to disable.",
        json_schema_extra=ui(widget=Widget.TOGGLE, tier=Tier.ADVANCED, order=13),
    )
    msa_paths: list[str] | None = Field(
        default=None,
        description=(
            "Pre-computed MSA file paths (filename starts with the chain ID, e.g. 'A.a3m')."
        ),
        json_schema_extra=ui(widget=Widget.FILE, tier=Tier.ADVANCED, order=14),
    )
    constraints: list[Any] | None = Field(
        default=None,
        description=(
            "Boltz YAML 'constraints' section. Three constraint types: bond — "
            "covalent bond between atoms: {bond: {atom1: [A, 437, N], atom2: "
            "[B, 1, C1]}}. pocket — binding site conditioning: {pocket: {binder: "
            "B, contacts: [[A, 100], [A, 101]]}}. contact — distance restraint "
            "between residues: {contact: {atoms: [[A, 100], [B, 50]], "
            "max_distance: 5.5}}."
        ),
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.ADVANCED, order=15),
    )
    properties: list[Any] | None = Field(
        default=None,
        description="Boltz YAML 'properties' section (Boltz-2).",
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.ADVANCED, order=16),
    )
    modifications: list[Any] | None = Field(
        default=None,
        description="Boltz YAML 'modifications' section.",
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.ADVANCED, order=17),
    )
    boltz_yaml: dict[str, Any] | str | None = Field(
        default=None,
        description=(
            "Raw Boltz YAML (dict or string) — bypasses automatic YAML generation "
            "for full control over sequences/constraints/modifications/properties. "
            "See https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md "
            "for the full native YAML schema."
        ),
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.ADVANCED, order=18),
    )


class Chai1Input(BaseInput):
    """Input for Chai-1 biomolecular structure prediction (single ``predict`` mode)."""

    sequences: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Mapping of chain ID to sequence (protein/DNA/RNA). For ligand chains "
            "(entity_types = 'ligand' or {'smiles': ...}) the value is a SMILES string. "
            "May be empty only when chai_fasta is provided."
        ),
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.PRIMARY, order=0),
    )
    num_models: int = Field(
        default=1,
        ge=1,
        description="Number of structures to generate (maps to num_diffn_samples).",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=10),
    )
    entity_types: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Per-chain entity type: 'protein'/'dna'/'rna'/'ligand', or a dict "
            "{'smiles': 'CC...'} for ligands. Default 'protein'."
        ),
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.ADVANCED, order=12),
    )
    constraints: str | None = Field(
        default=None,
        description=(
            "Restraints/covalent bonds as CSV content (or a file path). Columns: "
            "chainA,res_idxA,chainB,res_idxB,connection_type,confidence,"
            "min_distance_angstrom,max_distance_angstrom,comment,restraint_id. "
            "connection_type is 'contact', 'pocket', or 'covalent'."
        ),
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.ADVANCED, order=13),
    )
    msa_directory: str | None = Field(
        default=None,
        description="Path to a directory of pre-computed MSA .aligned.pqt files.",
        json_schema_extra=ui(widget=Widget.FILE, tier=Tier.ADVANCED, order=14),
    )
    chai_fasta: str | None = Field(
        default=None,
        description=(
            "Raw Chai-1 FASTA content (headers '>entity_type|name=chain_id'), "
            "bypassing automatic FASTA generation from sequences."
        ),
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.ADVANCED, order=15),
    )
    use_msa_server: bool = Field(
        default=True,
        description="Use ColabFold MMseqs2 MSA server (needs network). Set False to disable.",
        json_schema_extra=ui(widget=Widget.TOGGLE, tier=Tier.ADVANCED, order=16),
    )
    use_esm_embeddings: bool = Field(
        default=False,
        description="Enable ESM protein language model embeddings (extra compute).",
        json_schema_extra=ui(widget=Widget.TOGGLE, tier=Tier.ADVANCED, order=17),
    )


class OpenFold3Input(BaseInput):
    """Input for OpenFold3 biomolecular structure prediction (single ``predict`` mode)."""

    sequences: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Mapping of chain ID to sequence (protein/DNA/RNA). For ligand chains "
            "(entity_types = 'ligand'/{'smiles': ...}/{'ccd': ...}) the value is a "
            "SMILES string. May be empty only when query_json is provided."
        ),
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.PRIMARY, order=0),
    )
    num_models: int = Field(
        default=1,
        ge=1,
        description="Number of structures to generate (maps to num_diffusion_samples).",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=10),
    )
    entity_types: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Per-chain molecule type: 'protein'/'dna'/'rna'/'ligand', or a dict "
            "{'smiles': 'CC...'} / {'ccd': 'ATP'} for ligands. Default 'protein'."
        ),
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.ADVANCED, order=12),
    )
    non_canonical_residues: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Per-chain non-canonical residues as {chain_id: {position: CCD_code}}, "
            "e.g. {'A': {'3': 'MHO', '5': 'SEP'}}."
        ),
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.ADVANCED, order=13),
    )
    msa_paths: list[str] | None = Field(
        default=None,
        description=(
            "Pre-computed MSA file paths (filename stem identifies the target "
            "chain ID, e.g. 'A.a3m' maps to chain 'A', mirroring boltz's "
            "convention). Requires use_msa_server=False."
        ),
        json_schema_extra=ui(widget=Widget.FILE, tier=Tier.ADVANCED, order=14),
    )
    query_json: dict[str, Any] | str | None = Field(
        default=None,
        description=(
            "Raw OpenFold3 query JSON (dict or string) — bypasses automatic query "
            "generation. See https://openfold-3.readthedocs.io/en/latest/input_format_reference.html."
        ),
        json_schema_extra=ui(widget=Widget.TEXTAREA, tier=Tier.ADVANCED, order=15),
    )
    use_msa_server: bool = Field(
        default=True,
        description="Use ColabFold MMseqs2 MSA server (needs network). Set False to disable.",
        json_schema_extra=ui(widget=Widget.TOGGLE, tier=Tier.ADVANCED, order=16),
    )
    use_templates: bool = Field(
        default=True,
        description="Enable template-based prediction.",
        json_schema_extra=ui(widget=Widget.TOGGLE, tier=Tier.ADVANCED, order=17),
    )
    pae_enabled: bool = Field(
        default=True,
        description="Enable the PAE head (produces pTM/ipTM; higher memory).",
        json_schema_extra=ui(widget=Widget.TOGGLE, tier=Tier.ADVANCED, order=18),
    )
