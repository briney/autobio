# Schema Specification

This document defines how to create and maintain standardized input/output schemas for autobio tool categories. Schemas are the contract between host-side runners, containers, and agents — they determine what data flows in and out of every tool invocation.

---

## 1. Purpose

Each tool category (structure prediction, sequence embedding, inverse folding, etc.) defines a pair of Pydantic models: an **input schema** and an **output schema**. These schemas serve three roles:

1. **Host-side validation.** The runner validates user/agent input before launching a container.
2. **Container output contract.** The container's `standardize.sh` writes files conforming to the output schema.
3. **Agent discoverability.** The `autobio info --format json` command exposes schemas so agents can construct valid inputs.

---

## 2. File Location and Naming

All schemas live in `src/autobio/schemas/`. Each tool category gets its own module:

```
src/autobio/schemas/
├── __init__.py                    # re-exports all public types
├── SCHEMA_SPEC.md                 # this document
├── base.py                        # base types (never edited per-category)
├── structure_prediction.py        # StructurePrediction I/O
├── embedding.py                   # SequenceEmbedding I/O
├── inverse_folding.py             # InverseFolding I/O
├── scoring.py                     # EnergyScoring I/O
└── <new_category>.py              # new categories follow same pattern
```

Module names use `snake_case` matching the category's canonical identifier.

---

## 3. Base Types

All category schemas inherit from the base types defined in `base.py`. These must not be modified when adding a new category.

```python
from pydantic import BaseModel, Field
from pathlib import Path
from datetime import datetime
from typing import Any


class RunMetadata(BaseModel):
    """Attached to every tool output. Populated by the host runner, not the container."""
    tool_name: str
    tool_version: str
    image_uri: str
    wall_time_seconds: float
    gpu_ids: list[int] | None = None
    workspace_path: Path
    timestamp: datetime


class BaseInput(BaseModel):
    """
    Base class for all tool input schemas.
    All category inputs MUST inherit from this class.
    """
    extra: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Tool-specific parameters passed through to the container. "
            "Keys and values are forwarded as-is into config.json."
        ),
    )


class BaseOutput(BaseModel):
    """
    Base class for all tool output schemas.
    All category outputs MUST inherit from this class.
    """
    metadata: RunMetadata
    raw_output_path: Path = Field(
        description="Path to the outputs/raw/ directory containing unmodified tool outputs."
    )
```

---

## 4. Authoring a New Category Schema

### 4.1 Define the Input Model

Create a new module (e.g., `src/autobio/schemas/docking.py`). Define an input model that inherits from `BaseInput`. Use `Field(description=...)` for all fields — these descriptions are exposed to agents via `autobio info --format json`:

```python
from pathlib import Path
from pydantic import Field
from autobio.schemas.base import BaseInput


class DockingInput(BaseInput):
    """Input schema for molecular docking tools."""

    receptor_path: Path = Field(description="Path to the receptor structure (PDB or mmCIF).")
    ligand_smiles: str | None = Field(
        default=None,
        description="SMILES string for the ligand. Provide this or ligand_path.",
    )
    ligand_path: Path | None = Field(
        default=None,
        description="Path to the ligand structure (SDF or MOL2). Provide this or ligand_smiles.",
    )
    num_poses: int = Field(default=10, description="Number of docking poses to generate.")
    exhaustiveness: int = Field(
        default=8,
        description="Search exhaustiveness (higher = slower but more thorough).",
    )
```

**Rules for input fields:**

| Rule | Rationale |
|---|---|
| All fields MUST have type annotations. | Validation and serialization depend on types. |
| Required fields have no default value. | Pydantic enforces this at construction time. |
| Optional fields use `X \| None = None` or have a default value. | Agents can omit optional fields safely. |
| Every field MUST use `Field(description=...)`. | The `autobio info` command exposes these descriptions to agents. |
| Use `Path` for file references, not `str`. | Enables validation and consistent handling. |
| Use `dict[str, str]` for per-chain data (e.g., chain ID to sequence). | Multi-chain structures are common; standardize the mapping pattern. |
| Do NOT include fields that are purely execution-level concerns (GPU, timeout). | Those are handled by the runner's `run()` method, not the schema. |
| Do NOT duplicate `extra` — it is inherited from `BaseInput`. | Tool-specific overrides always go through `extra`. |
| Prefer semantic field names over tool-specific jargon. | Agents should be able to understand fields without knowing the underlying tool. |

### 4.2 Define Supporting Types

Complex output fields should be their own models. Keep them in the same module as the category schema.

```python
class DockingPose(BaseModel):
    """A single predicted docking pose."""

    rank: int
    """Rank by predicted affinity (1 = best)."""

    structure_path: Path
    """Path to the docked complex structure in outputs/standardized/."""

    affinity_kcal_mol: float | None = None
    """Predicted binding affinity in kcal/mol."""

    rmsd_from_best: float | None = None
    """RMSD relative to the top-ranked pose, in Ångströms."""

    confidence: float | None = None
    """Model confidence score, if provided by the tool (scale varies)."""
```

**Rules for supporting types:**

| Rule | Rationale |
|---|---|
| Use `Optional` (`X \| None = None`) for fields not all tools can provide. | Different tools in the same category have different output capabilities. |
| Include units in field names or docstrings. | `affinity_kcal_mol` is unambiguous. `affinity` is not. |
| Use standard scientific units. | Distances in Ångströms, energies in kcal/mol, etc. |
| File path fields MUST point to `outputs/standardized/`, not `outputs/raw/`. | Agents should never need to parse raw tool outputs. |

### 4.3 Define the Output Model

```python
from autobio.schemas.base import BaseOutput


class DockingOutput(BaseOutput):
    """Output schema for molecular docking tools."""

    poses: list[DockingPose]
    """Docking poses ranked by predicted affinity."""

    best_affinity_kcal_mol: float | None = None
    """Binding affinity of the top-ranked pose."""

    receptor_path: Path
    """Path to the prepared receptor in outputs/standardized/ (may differ from input)."""
```

**Rules for output fields:**

| Rule | Rationale |
|---|---|
| MUST inherit from `BaseOutput`. | Ensures `metadata` and `raw_output_path` are always present. |
| Include a convenience summary of the "best" result. | Agents making decisions often just need the top result, not the full list. |
| List fields should be ordered by rank/quality where applicable. | Consistent ordering reduces agent-side logic. |
| Structure file paths MUST reference files in `outputs/standardized/`. | The container's `standardize.sh` copies or converts native outputs into this directory. |

### 4.4 Register the Schema

After defining the schema, it must be connected in two places:

1. Export the input and output models from `schemas/__init__.py`:
   ```python
   from autobio.schemas.docking import DockingInput, DockingOutput, DockingPose
   ```

2. Reference them in the tool's `ToolEntry` (registered at the bottom of the tool module, not in `registry.py`):
   ```python
   # In src/autobio/tools/diffdock.py
   TOOL_REGISTRY["diffdock"] = ToolEntry(
       input_schema=DockingInput,
       output_schema=DockingOutput,
       ...
   )
   ```

---

## 5. Schema Serialization Format

Schemas are serialized to and from JSON. This is the format used in two places:

1. **`outputs/standardized/result_data.json`** — the container writes this file containing the serialized output model (minus `metadata` and `raw_output_path`, which the host populates).
2. **`autobio info --format json`** — the host serializes the input schema into a JSON description for agent consumption.

### 5.1 Container-Side Output Serialization

The container's `standardize.sh` produces a JSON file at `outputs/standardized/result_data.json`. This file contains the output schema fields that the container is responsible for — everything except `metadata` and `raw_output_path`, which the host runner populates.

Example for structure prediction:

```json
{
    "structures": [
        {
            "model_rank": 1,
            "structure_path": "outputs/standardized/model_1.pdb",
            "plddt_per_residue": [92.1, 88.4, 91.7, "..."],
            "plddt_mean": 90.7,
            "ptm": 0.89,
            "iptm": 0.85,
            "chain_mapping": {"A": "A", "B": "B"}
        }
    ],
    "confidence": {
        "best_plddt_mean": 90.7,
        "best_ptm": 0.89,
        "best_iptm": 0.85
    }
}
```

Example for inverse folding (produced by `containers/mpnn/standardize.py`):

```json
{
    "designed_sequences": [
        {
            "rank": 1,
            "sequence": {"H": "GVKLTESG...", "L": "ASVLTQPP..."},
            "score": null,
            "recovery": 0.5652
        },
        {
            "rank": 2,
            "sequence": {"H": "EVQLVESG...", "L": "DIVMTQSP..."},
            "score": null,
            "recovery": 0.4891
        }
    ],
    "native_sequence": {"H": "GVKLTESG...", "L": "ASVLTQPP..."}
}
```

**Rules:**
- All `Path` fields are serialized as strings relative to the workspace root.
- Null/None fields MAY be omitted from the JSON.
- Lists MUST preserve rank ordering.
- Floating-point values use full precision (no rounding).

### 5.2 Schema Documentation for Agents

The `autobio info` command generates a JSON description from the Pydantic model's schema. This is automatic — you do not need to write it manually. However, the quality of the generated description depends entirely on the quality of field annotations and docstrings. Every field must be annotated and documented.

---

## 6. Schema Evolution

Schemas will change over time as new tools are added and existing tools gain capabilities.

**Backwards-compatible changes** (safe to make):
- Adding new `Optional` fields to an existing schema.
- Adding new supporting types.
- Expanding a field's description.

**Breaking changes** (require major version bump):
- Removing a field.
- Renaming a field.
- Changing a field's type.
- Making an optional field required.

When a new tool in an existing category provides a metric or output that no existing tool provides, add it as an `Optional` field. If it later becomes common across most tools, it can remain optional (for backwards compatibility with tools that don't provide it) but should be documented as "provided by most tools in this category."

---

## 7. Checklist for New Schemas

- [ ] Module created at `schemas/<category>.py`
- [ ] Input model inherits from `BaseInput`
- [ ] Output model inherits from `BaseOutput`
- [ ] All fields have type annotations
- [ ] All fields have docstrings or `Field(description=...)`
- [ ] File path fields use `Path` type
- [ ] Units are explicit in field names or docstrings
- [ ] Supporting types defined for complex nested structures
- [ ] Optional fields use `X | None = None` pattern
- [ ] No execution-level concerns (GPU, timeout) in schema fields
- [ ] Models exported from `schemas/__init__.py`
- [ ] At least one tool's `ToolEntry` references the new schemas
- [ ] Unit tests cover serialization round-trip (model → JSON → model)
- [ ] Container's `standardize.sh` documentation updated to reference the output schema
