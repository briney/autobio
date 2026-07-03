# Tool→Modes Conformance Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the four as-built conformance gaps from `docs/CLEANUP.md` so every multi-mode Tool ships distinct per-mode schemas, LigandMPNN is one Tool with two modes, the embedding mode name is uniform, and the CLI JSON contract is self-consistent.

**Architecture:** These are four independent, host-side-only changes to the `autobio` package (`src/autobio/`) and its unit/integration tests. No container scripts (`containers/`) change; the container `config.json` contract is deliberately preserved. Each task is atomic — it leaves the package importable and the unit suite green — and can be committed and reviewed on its own.

**Tech Stack:** Python 3.11+, Pydantic v2 (schemas), Typer (CLI), pytest (tests), ruff (lint/format), mypy (types).

## Global Constraints

- **Python 3.11+**; modern syntax (`X | Y` unions, `match`), type hints on all signatures, Google-style docstrings on public classes/functions. Max line length **100**.
- **Run tests with `python -m pytest`** (bare `pytest` uses the wrong environment). If `src/` edits are not picked up, reinstall editable: `pip install -e ".[dev]"`.
- **No backward-compat shims / deprecation paths.** Every change is a clean break (single-owner package); removed names simply disappear.
- **Do not touch execution machinery** (`core/container.py`, `core/gpu.py`, `core/workspace.py`), output schemas, or any file under `containers/`. The container `config.json` shape is frozen.
- **Runner idiom:** one runner class per tool *name* (`TOOL_RUNNERS: dict[str, type[ToolRunner]]`); a single Tool's multiple modes are served by one runner that branches on `self.current_mode.name`.
- **Do not run** Docker/GPU integration tests locally (marked `docker`/`gpu`/`slow`); edit them for API consistency only. Run unit tests with `-m "not docker and not gpu"` when in doubt.
- Before finishing each task: `ruff check --fix src/ tests/` and `ruff format src/ tests/`.

---

## File Structure

| File | Task | Responsibility |
|------|------|----------------|
| `src/autobio/schemas/antibody.py` | 1 | Split `AntibodyInput` → `AntibodyBaseInput` + `AntibodyEmbeddingInput` + `AntibodyPLLInput` |
| `src/autobio/schemas/__init__.py` | 1 | Re-export the new antibody input classes |
| `src/autobio/tools/antibody_lm.py` | 1 | Per-mode `input_schema` wiring; mode-aware config + validation |
| `src/autobio/tools/ligandmpnn_packer.py` | 2 | Convert runner class → module-level helpers (`prepare_build_mutant`, `parse_build_mutant_output`) |
| `src/autobio/tools/mpnn.py` | 2 | Add `build_mutant` mode to the `ligandmpnn` Tool; delegate in `MPNNRunner` |
| `src/autobio/tools/__init__.py` | 2 | Drop `ligandmpnn_build_mutant` from `TOOL_RUNNERS` |
| `src/autobio/tools/esm.py` | 3 | Rename mode `embed` → `embedding` (esm1b, esm2) |
| `src/autobio/cli/formatters.py` | 4 | Unify `list` GPU fields; emit per-mode `image_tag` in `info` |
| Corresponding `tests/unit/*` and `tests/integration/*` | 1–4 | Assert the new contracts |

---

## Task 1: Split antibody-LM schema into per-mode inputs

**Why atomic:** `tools/antibody_lm.py` imports `AntibodyInput` at module load. Removing that name without simultaneously rewiring the runner (and the tests that import it) would make the whole `autobio` package fail to import, breaking every test. Schema + runner + their unit tests land together.

**Files:**
- Modify: `src/autobio/schemas/antibody.py`
- Modify: `src/autobio/schemas/__init__.py:6,51`
- Modify: `src/autobio/tools/antibody_lm.py`
- Test: `tests/unit/test_schemas.py`, `tests/unit/test_antibody_lm.py`
- Edit-only (not run locally): `tests/integration/test_currab_integration.py`, `tests/integration/test_ablang2_integration.py`, `tests/integration/test_antiberta2_integration.py`

**Interfaces:**
- Produces: `AntibodyBaseInput` (field `sequences`), `AntibodyEmbeddingInput(AntibodyBaseInput)` (adds `layer: int | None`, `pooling: str | None`), `AntibodyPLLInput(AntibodyBaseInput)` (adds `per_position: bool`). `AntibodyInput` is **removed**.
- Consumes (unchanged): `AntibodySequenceSet`, `ui`, `Tier`, `Widget`, `BaseInput`, `EmbeddingOutput`, `AntibodyPLLOutput`.

- [ ] **Step 1: Update the schema unit tests to the new API (write the failing tests)**

In `tests/unit/test_schemas.py`, change the antibody import block (lines 12-17) to:

```python
from autobio.schemas.antibody import (
    AntibodyEmbeddingInput,
    AntibodyPLLInput,
    AntibodyPLLOutput,
    AntibodySequence,
    SequencePLL,
)
```

Replace the entire `class TestAntibodyInput:` block (lines 218-269) with two classes:

```python
class TestAntibodyEmbeddingInput:
    def test_required_sequences(self) -> None:
        inp = AntibodyEmbeddingInput(sequences=[AntibodySequence(id="ab1", heavy_chain="EVQLV")])
        assert len(inp.sequences) == 1
        assert inp.layer is None
        assert inp.pooling is None

    def test_optional_fields(self) -> None:
        inp = AntibodyEmbeddingInput(
            sequences=[AntibodySequence(id="ab1", heavy_chain="EVQLV")],
            layer=10,
            pooling="mean",
        )
        assert inp.layer == 10
        assert inp.pooling == "mean"

    def test_no_per_position_field(self) -> None:
        assert "per_position" not in AntibodyEmbeddingInput.model_fields

    def test_extra_passthrough(self) -> None:
        inp = AntibodyEmbeddingInput(
            sequences=[AntibodySequence(id="ab1", heavy_chain="EVQLV")],
            extra={"seed": 42, "batch_size": 8},
        )
        assert inp.extra["seed"] == 42
        assert inp.extra["batch_size"] == 8

    def test_accepts_fasta_text(self) -> None:
        fasta = ">ab1|heavy\nEVQLVESGG\n>ab1|light\nDIQMTQSPS\n"
        inp = AntibodyEmbeddingInput(sequences=fasta)
        assert len(inp.sequences) == 1
        assert inp.sequences[0].id == "ab1"
        assert inp.sequences[0].heavy_chain == "EVQLVESGG"
        assert inp.sequences[0].light_chain == "DIQMTQSPS"


class TestAntibodyPLLInput:
    def test_required_sequences(self) -> None:
        inp = AntibodyPLLInput(sequences=[AntibodySequence(id="ab1", heavy_chain="EVQLV")])
        assert len(inp.sequences) == 1
        assert inp.per_position is False

    def test_per_position_typed_field(self) -> None:
        inp = AntibodyPLLInput(
            sequences=[AntibodySequence(id="ab1", heavy_chain="EVQLV")],
            per_position=True,
        )
        assert inp.per_position is True

    def test_no_layer_or_pooling_fields(self) -> None:
        assert "layer" not in AntibodyPLLInput.model_fields
        assert "pooling" not in AntibodyPLLInput.model_fields
```

In the `TestInputInheritance` parametrize (line ~589-592), change `AntibodyInput` to `AntibodyEmbeddingInput`:

```python
        (
            AntibodyEmbeddingInput,
            {"sequences": [AntibodySequence(id="ab1", heavy_chain="EVQLV")]},
        ),
```

- [ ] **Step 2: Run the schema tests to verify they fail**

Run: `python -m pytest tests/unit/test_schemas.py -q`
Expected: FAIL at collection — `ImportError: cannot import name 'AntibodyEmbeddingInput'`.

- [ ] **Step 3: Split the schema**

In `src/autobio/schemas/antibody.py`, replace the module docstring's second sentence and the `__all__` list, then replace the whole `class AntibodyInput(BaseInput):` block (lines 27-67) with the three classes below. Update `__all__` (lines 19-24) to:

```python
__all__ = [
    "AntibodyBaseInput",
    "AntibodyEmbeddingInput",
    "AntibodyPLLInput",
    "AntibodyPLLOutput",
    "AntibodySequence",
    "SequencePLL",
]
```

Replace the `AntibodyInput` class with:

```python
class AntibodyBaseInput(BaseInput):
    """Shared antibody-LM input: the sequence set (plus inherited ``extra``)."""

    sequences: AntibodySequenceSet = Field(
        description=(
            "One or more antibody sequences: a list of AntibodySequence/dicts, "
            "FASTA text, or a path to a .fasta/.fa file."
        ),
        json_schema_extra=ui(widget=Widget.SEQUENCE, flavor="antibody", tier=Tier.PRIMARY, order=0),
    )


class AntibodyEmbeddingInput(AntibodyBaseInput):
    """Input for the ``embedding`` mode."""

    layer: int | None = Field(
        default=None,
        description="Model layer from which to extract embeddings. None uses the final layer.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=10),
    )
    pooling: str | None = Field(
        default=None,
        description=(
            "Pooling strategy for per-residue embeddings ('mean', 'cls', 'per_residue')."
        ),
        json_schema_extra=ui(
            widget=Widget.SELECT,
            tier=Tier.PRIMARY,
            order=1,
            enum_labels={"mean": "Mean pool", "cls": "CLS token", "per_residue": "Per-residue"},
        ),
    )


class AntibodyPLLInput(AntibodyBaseInput):
    """Input for the ``pll`` (pseudo log-likelihood) mode."""

    per_position: bool = Field(
        default=False,
        description="Return per-position PLL scores. Slower.",
        json_schema_extra=ui(widget=Widget.TOGGLE, tier=Tier.ADVANCED, order=11),
    )
```

Also update the module docstring line that reads `pseudo log-likelihood), all using ``AntibodyInput`` as input.` to:
`pseudo log-likelihood).  The embedding mode uses ``AntibodyEmbeddingInput```
`and the PLL mode uses ``AntibodyPLLInput`` (both extend ``AntibodyBaseInput``).`

- [ ] **Step 4: Update the schema package re-exports**

In `src/autobio/schemas/__init__.py`, replace the single `AntibodyInput,` import line (line 6) with:

```python
    AntibodyBaseInput,
    AntibodyEmbeddingInput,
    AntibodyPLLInput,
```

and replace the `"AntibodyInput",` entry in `__all__` (line 51) with:

```python
    "AntibodyBaseInput",
    "AntibodyEmbeddingInput",
    "AntibodyPLLInput",
```

- [ ] **Step 5: Run the schema tests to verify they pass**

Run: `python -m pytest tests/unit/test_schemas.py -q`
Expected: PASS.

- [ ] **Step 6: Rewire the antibody runner (write failing runner tests first)**

In `tests/unit/test_antibody_lm.py`:

1. Change the import (line 17) to:

```python
from autobio.schemas.antibody import (
    AntibodyEmbeddingInput,
    AntibodyPLLInput,
    AntibodyPLLOutput,
    AntibodySequence,
)
```

2. Mechanical substitution of the input constructor by mode context. Replace `AntibodyInput(` with `AntibodyEmbeddingInput(` **everywhere**, EXCEPT in these PLL-context tests, where it becomes `AntibodyPLLInput(`:
   - `test_mode_pll`, `test_per_position_default_false`, `test_per_position_opt_in`,
   - `test_pll_full_config_defaults`, `test_pll_full_config_with_per_position`,
   - `test_extra_per_position_shadow_raises`,
   - `TestAntibodyLMRunMetadataMode.test_run_metadata_mode_pll`.

   (`test_run_metadata_mode_embedding` and every other test use `AntibodyEmbeddingInput`.)

3. Replace `test_pll_full_config_with_per_position` (lines 387-413) in full — the PLL input no longer carries `layer`/`pooling`, so the config reflects the defaults:

```python
    def test_pll_full_config_with_per_position(
        self, currab_pll_runner: AntibodyLMRunner, tmp_path: Path
    ) -> None:
        workspace = Workspace.create(tmp_path / "ws")
        input_data = AntibodyPLLInput(
            sequences=[AntibodySequence(id="ab1", heavy_chain=_VH)],
            per_position=True,
        )
        currab_pll_runner.prepare_workspace(input_data, workspace)

        cfg = json.loads(workspace.config_path.read_text())
        expected = {
            "model_name": "brineylab/CurrAb",
            "model_family": "esm",
            "chain_separator": "single_cls",
            "input_file": "/workspace/inputs/sequences.json",
            "output_dir": "/workspace/outputs/raw",
            "mode": "pll",
            "layer": None,
            "pooling": "mean",
            "per_position": True,
            "hf_cache": "/app/antibody-lm/hf_cache",
        }
        assert cfg == expected
        assert list(cfg.keys()) == list(expected.keys())
```

4. Replace `test_mode_schemas_and_timeouts` (lines 773-794) — the two modes now point at different schemas:

```python
    @pytest.mark.parametrize(
        ("mode_name", "input_schema", "output_schema", "timeout"),
        [
            ("embedding", AntibodyEmbeddingInput, EmbeddingOutput, 600),
            ("pll", AntibodyPLLInput, AntibodyPLLOutput, 1800),
        ],
    )
    @pytest.mark.parametrize("model_name", _MODEL_NAMES)
    def test_mode_schemas_and_timeouts(
        self,
        model_name: str,
        mode_name: str,
        input_schema: type,
        output_schema: type,
        timeout: int,
    ) -> None:
        mode = get_tool(model_name).modes[mode_name]
        assert mode.input_schema is input_schema
        assert mode.output_schema is output_schema
        assert mode.default_timeout == timeout
        assert mode.supports_batch is True
        # Modes don't override the Tool's image — both modes share one image.
        assert mode.image_tag is None
```

5. Extend `TestAntibodyLMInfoSnapshot.test_info_snapshot` (lines 848-862) to assert the per-mode schemas no longer leak the other mode's fields — append before the final line of that method:

```python
        embedding_props = embedding_mode["input_schema"]["properties"]
        pll_props = pll_mode["input_schema"]["properties"]
        assert "per_position" not in embedding_props
        assert "layer" not in pll_props
        assert "pooling" not in pll_props
```

- [ ] **Step 7: Run the antibody runner tests to verify they fail**

Run: `python -m pytest tests/unit/test_antibody_lm.py -q`
Expected: FAIL at collection — `ImportError: cannot import name 'AntibodyEmbeddingInput' from ... antibody` is already resolved (Task 1 shipped the schema), so instead expect FAIL in `AntibodyLMRunner.prepare_workspace` / `test_mode_schemas_and_timeouts` because the runner still imports/asserts `AntibodyInput` and still wires both modes to it.

- [ ] **Step 8: Rewire the runner source**

In `src/autobio/tools/antibody_lm.py`:

1. Change the schema import (line 23) to:

```python
from autobio.schemas.antibody import (
    AntibodyBaseInput,
    AntibodyEmbeddingInput,
    AntibodyPLLInput,
    AntibodyPLLOutput,
    SequencePLL,
)
```

2. In `prepare_workspace` (line 142), change the assert to the base class and build the config with `getattr` so the container `config.json` keeps all four keys regardless of mode:

```python
        assert isinstance(input_data, AntibodyBaseInput)
```

and replace the `config` dict (lines 164-175) with:

```python
        mode = self.current_mode.name
        config: dict[str, object] = {
            "model_name": spec.model_name,
            "model_family": spec.model_family,
            "chain_separator": spec.chain_separator,
            "input_file": "/workspace/inputs/sequences.json",
            "output_dir": "/workspace/outputs/raw",
            "mode": mode,
            "layer": getattr(input_data, "layer", None),
            "pooling": getattr(input_data, "pooling", None) or "mean",
            "per_position": getattr(input_data, "per_position", False),
            "hf_cache": spec.cache_path,
        }
```

3. Change `_validate_inputs` (line 244) signature and scope the embedding-only checks behind the embedding input type:

```python
    @staticmethod
    def _validate_inputs(input_data: AntibodyBaseInput, spec: _ModelSpec) -> None:
```

Replace the trailing layer/pooling validation (lines 285-296) with:

```python
        # Embedding-only fields (absent on the PLL input schema).
        if isinstance(input_data, AntibodyEmbeddingInput):
            if input_data.layer is not None and not (0 <= input_data.layer <= spec.num_layers):
                raise AutobioError(
                    f"layer must be between 0 and {spec.num_layers} for "
                    f"{spec.model_name}, got {input_data.layer}."
                )
            if input_data.pooling is not None and input_data.pooling not in _VALID_POOLING:
                raise AutobioError(
                    f"pooling must be one of {sorted(_VALID_POOLING)}, got {input_data.pooling!r}."
                )
```

4. In `_register_antibody_lm`, point each mode at its own schema (lines 357 and 367):

```python
                "embedding": Mode(
                    name="embedding",
                    display_name="Embed sequences",
                    description=embed_description,
                    input_schema=AntibodyEmbeddingInput,
                    output_schema=EmbeddingOutput,
                    default_timeout=600,
                    supports_batch=True,
                    notes=_ANTIBODY_NOTES,
                ),
                "pll": Mode(
                    name="pll",
                    display_name="Pseudo log-likelihood",
                    description=pll_description,
                    input_schema=AntibodyPLLInput,
                    output_schema=AntibodyPLLOutput,
                    default_timeout=1800,
                    supports_batch=True,
                    notes=_ANTIBODY_NOTES + _PLL_NOTES,
                ),
```

5. Update the module docstring line 11 (`passed through the ``extra`` dict on ``AntibodyInput``.`) to `passed through the ``extra`` dict on the mode's input schema.`

- [ ] **Step 9: Run the antibody runner tests to verify they pass**

Run: `python -m pytest tests/unit/test_antibody_lm.py tests/unit/test_schemas.py -q`
Expected: PASS.

- [ ] **Step 10: Update the antibody integration tests (edit-only, not run locally)**

In each of `tests/integration/test_currab_integration.py`, `test_ablang2_integration.py`, `test_antiberta2_integration.py`:

1. Change the `from autobio.schemas.antibody import (...)` block to import `AntibodyEmbeddingInput, AntibodyPLLInput, AntibodyPLLOutput, AntibodySequence` (drop `AntibodyInput`).
2. Replace `AntibodyInput(` with `AntibodyEmbeddingInput(` in every embedding-mode test, and with `AntibodyPLLInput(` in every test whose `runner.run(...)` call passes `mode="pll"` (in `test_currab_integration.py` these are the `TestCurrAbPLL`, `TestFtEsmPLL`, `TestBalmPairedPLL`, `TestBalmUnpairedPLL` methods; apply the analogous rule in the other two files).

Verify no references remain: `grep -rn "AntibodyInput" tests/integration/` returns nothing.

- [ ] **Step 11: Lint, format, and commit**

Run:
```bash
ruff check --fix src/ tests/ && ruff format src/ tests/
python -m pytest tests/unit/test_antibody_lm.py tests/unit/test_schemas.py -q
git add src/autobio/schemas/antibody.py src/autobio/schemas/__init__.py \
        src/autobio/tools/antibody_lm.py tests/unit/test_antibody_lm.py \
        tests/unit/test_schemas.py tests/integration/test_currab_integration.py \
        tests/integration/test_ablang2_integration.py tests/integration/test_antiberta2_integration.py
git commit -m "antibody_lm: split AntibodyInput into per-mode schemas"
```

---

## Task 2: Consolidate LigandMPNN into one Tool with modes {design, build_mutant}

**Why atomic:** Removing `LigandMPNNPackerRunner` and its Tool registration, adding the `build_mutant` mode to the `ligandmpnn` Tool, and rewiring `MPNNRunner` all depend on each other; any partial state leaves an import error or a dangling `TOOL_RUNNERS` entry. Source + tests land together.

**Files:**
- Modify: `src/autobio/tools/ligandmpnn_packer.py` (runner class → module helpers)
- Modify: `src/autobio/tools/mpnn.py` (add mode; delegate)
- Modify: `src/autobio/tools/__init__.py:19,60` (drop the packer runner)
- Test: `tests/unit/test_ligandmpnn_packer_e2e.py`, `tests/unit/test_mpnn.py`
- Edit-only (not run locally): `tests/integration/test_ligandmpnn_packer_integration.py`

**Interfaces:**
- Produces (from `ligandmpnn_packer.py`): `prepare_build_mutant(runner: ToolRunner, input_data: BaseInput, workspace: Workspace) -> None`, `parse_build_mutant_output(runner: ToolRunner, workspace: Workspace) -> ScoringOutput`, `BUILD_MUTANT_NOTES: tuple[str, ...]`. Removed: `LigandMPNNPackerRunner`, `LIGANDMPNN_PACKER_TOOL`.
- Consumes: `MPNNRunner` (in `mpnn.py`) calls the two helpers with `self`. The `ligandmpnn` Tool gains a `build_mutant` `Mode` (`input_schema=LigandMPNNPackerInput`, `output_schema=ScoringOutput`, `image_tag="ligandmpnn-packer:1.0.0"`, `category=ToolCategory.SCORING`).

- [ ] **Step 1: Update the packer e2e tests to drive the consolidated tool (write failing tests)**

In `tests/unit/test_ligandmpnn_packer_e2e.py`:

1. Replace the packer-runner import (line 27) with:

```python
from autobio.tools.mpnn import MPNNRunner
```

2. Replace `_make_runner` (lines 131-139) with:

```python
def _make_runner(config: AutobioConfig) -> MPNNRunner:
    """Create an MPNNRunner (ligandmpnn tool) with mocked deps, mode=build_mutant."""
    with (
        patch("autobio.tools.base.ContainerManager"),
        patch("autobio.tools.base.GPUManager"),
    ):
        runner = MPNNRunner("ligandmpnn", config)
    runner.current_mode = get_tool("ligandmpnn").modes["build_mutant"]
    return runner
```

3. Change the two `LigandMPNNPackerRunner` type hints (in `_written_config` line 161 and `_run_e2e` where it constructs `runner`) to `MPNNRunner`.

4. Replace the entire `class TestRegistration:` block (lines 565-603) with:

```python
class TestRegistration:
    """LigandMPNN is one Tool with design + build_mutant modes."""

    def test_build_mutant_is_a_mode_of_ligandmpnn(self) -> None:
        tool = get_tool("ligandmpnn")
        assert "build_mutant" in tool.modes
        mode = tool.modes["build_mutant"]
        assert mode.input_schema is LigandMPNNPackerInput
        assert mode.output_schema is ScoringOutput
        assert mode.image_tag == "ligandmpnn-packer:1.0.0"
        assert mode.default_timeout == 600

    def test_old_tool_name_removed(self) -> None:
        from autobio.core.catalog import CATALOG

        assert "ligandmpnn_build_mutant" not in CATALOG
        assert "ligandmpnn_build_mutant" not in TOOL_RUNNERS

    def test_has_notes(self) -> None:
        notes = get_tool("ligandmpnn").modes["build_mutant"].notes
        assert len(notes) > 0
        assert any("chi" in note.lower() or "sidechain" in note.lower() for note in notes)
```

5. Replace `TestInfoSnapshot.test_info_snapshot` (lines 614-627) to target the `build_mutant` mode (index 1; `design` is index 0):

```python
    def test_info_snapshot(self) -> None:
        from autobio.cli.formatters import OutputFormat, format_tool_info_catalog

        parsed = json.loads(
            format_tool_info_catalog(get_tool("ligandmpnn"), OutputFormat.JSON)
        )
        mode = next(m for m in parsed["modes"] if m["name"] == "build_mutant")
        props = mode["input_schema"]["properties"]
        assert props["structure_path"]["x-autobio"]["widget"] == "file"
        assert props["structure_path"]["x-autobio"]["tier"] == "primary"
        assert props["num_packs"]["x-autobio"]["tier"] == "advanced"
        assert mode["image_tag"] == "ligandmpnn-packer:1.0.0"
        assert "output_schema" in mode
        assert mode["notes"]
```

- [ ] **Step 2: Run the packer e2e tests to verify they fail**

Run: `python -m pytest tests/unit/test_ligandmpnn_packer_e2e.py -q`
Expected: FAIL at collection — `ImportError` on `from autobio.tools.ligandmpnn_packer import LIGANDMPNN_PACKER_TOOL, LigandMPNNPackerRunner` is gone, but `get_tool("ligandmpnn").modes["build_mutant"]` raises `KeyError` because the mode does not exist yet.

- [ ] **Step 3: Convert the packer runner class to module-level helpers**

Rewrite `src/autobio/tools/ligandmpnn_packer.py` in full:

```python
"""LigandMPNN sidechain packing — build mutant structures (``build_mutant`` mode).

Uses LigandMPNN's sidechain packing neural network to introduce amino acid
mutations and repack sidechains.  The packing model predicts chi1–chi4
torsion angles as mixtures of von Mises distributions, producing full-atom
PDB structures with per-residue confidence scores.

This is conceptually an alternative to ``evoef2_build_mutant``, which uses
a physics-based rotamer library.  The LigandMPNN packer uses a learned model
that is also ligand-aware (can consider bound ligands when packing).

These helpers are the ``build_mutant`` mode of the ``ligandmpnn`` Tool; they are
invoked by ``MPNNRunner`` (see :mod:`autobio.tools.mpnn`), which owns the Tool
registration and dispatches on ``self.current_mode.name``.  The task runs in a
dedicated container built from the original ``dauparas/LigandMPNN`` code (the
Rosetta Commons foundry does not expose sidechain packing).
"""

from __future__ import annotations

import json
import re
import shutil
from typing import TYPE_CHECKING, Any

from autobio.core.result import AutobioError
from autobio.schemas.scoring import LigandMPNNPackerInput, ScoredStructure, ScoringOutput

if TYPE_CHECKING:
    from autobio.core.workspace import Workspace
    from autobio.schemas.base import BaseInput
    from autobio.tools.base import ToolRunner

# ---------------------------------------------------------------------------
# Container-internal paths
# ---------------------------------------------------------------------------

_CHECKPOINT_SC = "/app/LigandMPNN/model_params/ligandmpnn_sc_v_32_002_16.pt"
_CHECKPOINT_BB = "/app/LigandMPNN/model_params/ligandmpnn_v_32_010_25.pt"

# ---------------------------------------------------------------------------
# Mutation validation — same regex and format as EvoEF2 build_mutant
# ---------------------------------------------------------------------------

_MUTATION_RE = re.compile(r"^[ACDEFGHIKLMNPQRSTVWY][A-Za-z]\d+[ACDEFGHIKLMNPQRSTVWY]$")

_MUTATION_FORMAT_HELP = (
    "Mutations must be strings like 'EA63Q' (WT-chain-resnum-new). "
    "Each character: single-letter amino acid code for wild-type, "
    "chain ID letter, residue number, single-letter amino acid code for mutant."
)

# ---------------------------------------------------------------------------
# Mode notes — surfaced on the ligandmpnn build_mutant Mode
# ---------------------------------------------------------------------------

BUILD_MUTANT_NOTES = (
    "Builds mutant protein structures using LigandMPNN's neural network "
    "sidechain packing model, which predicts chi1–chi4 torsion angles as "
    "mixtures of von Mises distributions. Produces full-atom PDB structures.",
    "Mutations are specified as a list of strings. Format: 'EA63Q' means "
    "chain E, Ala-63 -> Gln. Multiple mutations are applied simultaneously.",
    "Scores are chi-angle log-probabilities from the packing model "
    "(higher = more confident). These are NOT energy scores like EvoEF2.",
    "The packer is ligand-aware: if the input PDB contains bound ligands "
    "(HETATM records), they are used as context during sidechain packing. "
    "This is an advantage over physics-based methods for ligand-binding sites.",
    "Proline ring geometry and disulfide bonds are predicted by the model "
    "but may benefit from downstream energy minimization for accuracy.",
)


def _validate_build_mutant_inputs(input_data: LigandMPNNPackerInput) -> None:
    """Host-side validation — catch errors before container launch."""
    if not input_data.structure_path.exists():
        raise AutobioError(f"Input structure file does not exist: {input_data.structure_path}")

    suffix = input_data.structure_path.suffix.lower()
    if suffix not in (".pdb",):
        raise AutobioError(
            f"LigandMPNN sidechain packer only supports PDB format, got '{suffix}'. "
            "Convert mmCIF/other formats to PDB before using this tool."
        )

    mutations = input_data.mutations
    if not mutations:
        raise AutobioError(
            f"LigandMPNN packer requires at least one mutation. {_MUTATION_FORMAT_HELP}"
        )
    for m in mutations:
        if not _MUTATION_RE.match(m):
            raise AutobioError(f"Invalid mutation format: {m!r}. {_MUTATION_FORMAT_HELP}")


def prepare_build_mutant(runner: ToolRunner, input_data: BaseInput, workspace: Workspace) -> None:
    """Write config.json and copy the input structure for the build_mutant mode."""
    assert isinstance(input_data, LigandMPNNPackerInput)

    _validate_build_mutant_inputs(input_data)

    src_path = input_data.structure_path
    dest_name = src_path.name
    shutil.copy2(src_path, workspace.inputs_dir / dest_name)
    container_structure_path = f"/workspace/inputs/{dest_name}"

    config: dict[str, Any] = {
        "structure_path": container_structure_path,
        "mutations": input_data.mutations,
        "checkpoint_sc": _CHECKPOINT_SC,
        "checkpoint_bb": _CHECKPOINT_BB,
        "num_packs": input_data.num_packs,
        "num_denoising_steps": input_data.num_denoising_steps,
        "num_samples": input_data.num_samples,
        "repack_everything": input_data.repack_everything,
        "pack_with_ligand_context": input_data.pack_with_ligand_context,
    }

    runner._apply_extra(config, input_data)

    workspace.write_config(config)


def parse_build_mutant_output(runner: ToolRunner, workspace: Workspace) -> ScoringOutput:
    """Read standardised outputs and return a ``ScoringOutput``."""
    result_data_path = workspace.std_output_dir / "result_data.json"
    data = json.loads(result_data_path.read_text())

    scores = []
    for s in data["scores"]:
        structure_path = None
        if s.get("structure_path"):
            structure_path = runner._resolve_container_path(s["structure_path"], workspace)

        scores.append(
            ScoredStructure(
                total_score=s["total_score"],
                per_residue_scores=s.get("per_residue_scores"),
                score_breakdown=s.get("score_breakdown"),
                units=s.get("units"),
                structure_path=structure_path,
                ddg=s.get("ddg"),
                mutations=s.get("mutations"),
            )
        )

    # Placeholder metadata — overwritten by base class run().
    return ScoringOutput(
        scores=scores,
        metadata=runner._build_metadata(workspace, 0.0, [], ""),
        raw_output_path=workspace.raw_output_dir,
    )
```

(`SLF001`/private-access is not in the ruff select list, so `runner._apply_extra` / `runner._resolve_container_path` / `runner._build_metadata` need no `# noqa`.)

- [ ] **Step 4: Add the build_mutant mode to the ligandmpnn Tool and delegate in MPNNRunner**

In `src/autobio/tools/mpnn.py`:

1. Add imports. After the `inverse_folding` import block (lines 20-24) add:

```python
from autobio.schemas.scoring import LigandMPNNPackerInput, ScoringOutput
```

and after the `from autobio.tools.base import ToolRunner` line add:

```python
from autobio.tools.ligandmpnn_packer import (
    BUILD_MUTANT_NOTES,
    parse_build_mutant_output,
    prepare_build_mutant,
)
```

2. Add a mode guard at the top of `prepare_workspace` (replace lines 62-65):

```python
    def prepare_workspace(self, input_data: BaseInput, workspace: Workspace) -> None:
        """Write config.json and copy the input structure into the workspace."""
        assert self.current_mode is not None
        if self.current_mode.name == "build_mutant":
            prepare_build_mutant(self, input_data, workspace)
            return

        assert isinstance(input_data, MPNNInput)
        model_cfg = _MODEL_CONFIG[self.tool_name]
```

3. Widen `parse_output` and add the guard (replace lines 104-107):

```python
    def parse_output(self, workspace: Workspace) -> InverseFoldingOutput | ScoringOutput:
        """Read standardised outputs and return the mode-appropriate output model."""
        assert self.current_mode is not None
        if self.current_mode.name == "build_mutant":
            return parse_build_mutant_output(self, workspace)

        result_data_path = workspace.std_output_dir / "result_data.json"
        data = json.loads(result_data_path.read_text())
```

4. Replace the `LIGANDMPNN_TOOL` definition (lines 180-202) with the two-mode Tool:

```python
LIGANDMPNN_TOOL = Tool(
    name="ligandmpnn",
    display_name="LigandMPNN",
    category=ToolCategory.INVERSE_FOLDING,
    description="Design protein sequences with ligand awareness using LigandMPNN.",
    version="1.0.0",
    image_tag="mpnn:1.0.0",
    requires_gpu=True,
    gpu_count=1,
    default_mode="design",
    modes={
        "design": Mode(
            name="design",
            display_name="Design sequences",
            description="Design protein sequences (ligand-aware) for a backbone structure.",
            input_schema=MPNNInput,
            output_schema=InverseFoldingOutput,
            default_timeout=600,
            notes=_MPNN_NOTES + (_LIGANDMPNN_LIGAND_NOTE,),
        ),
        "build_mutant": Mode(
            name="build_mutant",
            display_name="Build mutant",
            description="Introduce mutations and repack sidechains into full-atom structures.",
            input_schema=LigandMPNNPackerInput,
            output_schema=ScoringOutput,
            default_timeout=600,
            image_tag="ligandmpnn-packer:1.0.0",
            category=ToolCategory.SCORING,
            notes=BUILD_MUTANT_NOTES,
        ),
    },
    keywords=(
        "ligandmpnn",
        "inverse folding",
        "sequence design",
        "ligand",
        "mpnn",
        "mutant",
        "sidechain packing",
        "repack",
        "mutation",
    ),
)
```

- [ ] **Step 5: Drop the packer runner from TOOL_RUNNERS**

In `src/autobio/tools/__init__.py`:
- Delete the import line (line 19): `from autobio.tools.ligandmpnn_packer import LigandMPNNPackerRunner`.
- Delete the `TOOL_RUNNERS` entry (line 60): `"ligandmpnn_build_mutant": LigandMPNNPackerRunner,`.

(`ligandmpnn_packer.py` is still loaded — `mpnn.py` imports its helpers — so its module code runs; it just no longer registers a Tool or a runner.)

- [ ] **Step 6: Run the LigandMPNN unit tests to verify they pass**

Run: `python -m pytest tests/unit/test_ligandmpnn_packer_e2e.py tests/unit/test_mpnn.py -q`
Expected: PASS for `test_ligandmpnn_packer_e2e.py`. `test_mpnn.py` may still FAIL on `test_ligandmpnn_registered_as_catalog_tool` (asserts `modes == {"design"}`) — fixed next.

- [ ] **Step 7: Update test_mpnn.py registration assertions**

In `tests/unit/test_mpnn.py`:

1. In `test_ligandmpnn_registered_as_catalog_tool` (line 433), change the modes assertion:

```python
        assert set(tool.modes) == {"design", "build_mutant"}
```

2. Add a new test to `TestMPNNRegistration` for the cross-category mode:

```python
    def test_ligandmpnn_build_mutant_mode(self) -> None:
        from autobio.core.catalog import tool_categories
        from autobio.schemas.scoring import LigandMPNNPackerInput, ScoringOutput

        tool = get_tool("ligandmpnn")
        bm = tool.modes["build_mutant"]
        assert bm.input_schema is LigandMPNNPackerInput
        assert bm.output_schema is ScoringOutput
        assert bm.image_tag == "ligandmpnn-packer:1.0.0"
        assert bm.category == ToolCategory.SCORING
        assert tool_categories(tool) == (
            ToolCategory.INVERSE_FOLDING,
            ToolCategory.SCORING,
        )

    def test_ligandmpnn_build_mutant_not_a_tool_name(self) -> None:
        from autobio.core.catalog import CATALOG

        assert "ligandmpnn_build_mutant" not in CATALOG
        assert "ligandmpnn_build_mutant" not in TOOL_RUNNERS
```

- [ ] **Step 8: Run test_mpnn.py to verify it passes**

Run: `python -m pytest tests/unit/test_mpnn.py -q`
Expected: PASS.

- [ ] **Step 9: Update the packer integration test (edit-only, not run locally)**

In `tests/integration/test_ligandmpnn_packer_integration.py`, in all three test methods:
- Change `get_runner("ligandmpnn_build_mutant", autobio_config)` → `get_runner("ligandmpnn", autobio_config)`.
- Change `runner.run(input_data, output_dir=tmp_path / "ws")` → `runner.run(input_data, output_dir=tmp_path / "ws", mode="build_mutant")`.
- In `test_single_mutation`, change `output.metadata.tool_name == "ligandmpnn_build_mutant"` → `== "ligandmpnn"`.

Verify: `grep -rn "ligandmpnn_build_mutant" src/ tests/` returns nothing.

- [ ] **Step 10: Lint, format, and commit**

Run:
```bash
ruff check --fix src/ tests/ && ruff format src/ tests/
python -m pytest tests/unit/test_ligandmpnn_packer_e2e.py tests/unit/test_mpnn.py -q
git add src/autobio/tools/ligandmpnn_packer.py src/autobio/tools/mpnn.py \
        src/autobio/tools/__init__.py tests/unit/test_ligandmpnn_packer_e2e.py \
        tests/unit/test_mpnn.py tests/integration/test_ligandmpnn_packer_integration.py
git commit -m "ligandmpnn: consolidate build_mutant into one Tool with two modes"
```

---

## Task 3: Normalize the embedding mode name (esm `embed` → `embedding`)

**Files:**
- Modify: `src/autobio/tools/esm.py` (six literals: `default_mode`, dict key, `Mode(name=...)` for esm1b and esm2)
- Test: `tests/unit/test_esm.py`
- Edit-only (not run locally): `tests/integration/test_esm_integration.py`

**Interfaces:**
- Produces: `esm1b` and `esm2` each expose a single mode named `embedding` (was `embed`). `ESMRunner` does not reference the mode name (it branches on `self.tool.name`), so no runner change.
- Container check: the ESM container `config.json` has **no** `mode` key (only `model_name`, `input_fasta`, `output_dir`, `layer`, `pooling`, `hf_cache`), so this rename is host-only.

- [ ] **Step 1: Update the ESM unit tests (write failing tests)**

In `tests/unit/test_esm.py`:
- Line 1 docstring: `(mode: embed)` → `(mode: embedding)`.
- Line 23: `.modes["embed"]` → `.modes["embedding"]`.
- Line 42: `== {"embed"}` → `== {"embedding"}`.
- Line 202: `parsed["modes"][0]["name"] == "embed"` → `== "embedding"`.

- [ ] **Step 2: Run the ESM tests to verify they fail**

Run: `python -m pytest tests/unit/test_esm.py -q`
Expected: FAIL — `KeyError: 'embedding'` in `_make_runner` (mode not yet renamed).

- [ ] **Step 3: Rename the mode in esm.py**

In `src/autobio/tools/esm.py`, for **both** `ESM1B_TOOL` (lines 186-198) and `ESM2_TOOL` (lines 218-230), change the three literals each:
- `default_mode="embed"` → `default_mode="embedding"`
- the `modes` dict key `"embed":` → `"embedding":`
- `Mode(name="embed", ...)` → `Mode(name="embedding", ...)`

Leave `display_name="Embeddings"` and everything else unchanged.

- [ ] **Step 4: Run the ESM tests to verify they pass**

Run: `python -m pytest tests/unit/test_esm.py -q`
Expected: PASS.

- [ ] **Step 5: Update the ESM integration test (edit-only, not run locally)**

In `tests/integration/test_esm_integration.py`, change all five occurrences of `mode="embed"` to `mode="embedding"`.

Verify: `grep -rn 'mode="embed"\|modes\["embed"\]\|"embed"' src/ tests/` returns nothing (the generic `_mode("embed")` fixtures in `test_catalog.py` are unrelated synthetic names and are out of scope — confirm none of the matches are in `esm.py`/`test_esm.py`/`test_esm_integration.py`).

- [ ] **Step 6: Lint, format, and commit**

Run:
```bash
ruff check --fix src/ tests/ && ruff format src/ tests/
python -m pytest tests/unit/test_esm.py -q
git add src/autobio/tools/esm.py tests/unit/test_esm.py tests/integration/test_esm_integration.py
git commit -m "esm: rename embedding mode embed -> embedding for cross-tool consistency"
```

---

## Task 4: Self-consistent CLI JSON contract

**Files:**
- Modify: `src/autobio/cli/formatters.py` (`list` JSON rows lines 44-59; `info` per-mode dict lines 96-109)
- Test: `tests/unit/test_formatters.py`

**Interfaces:**
- Produces: `list` JSON rows emit `"requires_gpu": bool` and `"gpu_count": int` (the key `"gpu"` is removed). `info` JSON per-mode dicts gain `"image_tag": mode.image_tag or tool.image_tag`.
- The only in-repo consumer of the old `list` `"gpu"` key is `tests/unit/test_formatters.py:113` (verified by grep).

- [ ] **Step 1: Update the formatter tests (write failing tests)**

In `tests/unit/test_formatters.py`:

1. In `TestFormatToolList.test_json_populated`, replace the GPU assertion (line 113) with:

```python
        assert parsed[0]["requires_gpu"] is True
        assert parsed[0]["gpu_count"] == 1
```

2. In `test_format_tool_info_catalog_json_shape`, add per-mode image assertions before the end of the function:

```python
    assert mode_a["image_tag"] == "demo:1.0.0"  # falls back to Tool default
    assert parsed["modes"][1]["image_tag"] == "demo:1.0.0"
```

3. Add a dedicated test for a per-mode image override (place after `test_format_tool_info_catalog_json_shape`):

```python
def test_format_tool_info_catalog_json_per_mode_image_override() -> None:
    tool = Tool(
        name="multi-image",
        display_name="Multi Image",
        category=ToolCategory.SCORING,
        description="tool with a per-mode image override",
        version="1.0.0",
        image_tag="base:1.0.0",
        requires_gpu=False,
        gpu_count=0,
        default_mode="a",
        modes={
            "a": Mode("a", "A", "a mode", _InInfo, _OutInfo, default_timeout=300),
            "b": Mode(
                "b",
                "B",
                "b mode",
                _InInfo,
                _OutInfo,
                default_timeout=300,
                image_tag="override:2.0.0",
            ),
        },
    )
    parsed = json.loads(format_tool_info_catalog(tool, OutputFormat.JSON))
    by_name = {m["name"]: m for m in parsed["modes"]}
    assert by_name["a"]["image_tag"] == "base:1.0.0"       # tool default
    assert by_name["b"]["image_tag"] == "override:2.0.0"   # mode override
```

- [ ] **Step 2: Run the formatter tests to verify they fail**

Run: `python -m pytest tests/unit/test_formatters.py -q`
Expected: FAIL — `KeyError: 'requires_gpu'` in `test_json_populated`; `KeyError: 'image_tag'` in the info tests.

- [ ] **Step 3: Update the list JSON rows**

In `src/autobio/cli/formatters.py`, replace the `"gpu"` line in the `format_tool_list` JSON rows (line 51):

```python
                "requires_gpu": tool.requires_gpu,
                "gpu_count": tool.gpu_count,
```

(Leave the TABLE branch's GPU column as-is; the table is not a machine contract.)

- [ ] **Step 4: Emit per-mode image_tag in info JSON**

In `format_tool_info_catalog`, add `image_tag` to the per-mode dict (inside the `modes = [ {...} ...]` comprehension, after the `"category"` entry, around line 100):

```python
                "image_tag": mode.image_tag or tool.image_tag,
```

- [ ] **Step 5: Run the formatter tests to verify they pass**

Run: `python -m pytest tests/unit/test_formatters.py -q`
Expected: PASS.

- [ ] **Step 6: Lint, format, and commit**

Run:
```bash
ruff check --fix src/ tests/ && ruff format src/ tests/
python -m pytest tests/unit/test_formatters.py -q
git add src/autobio/cli/formatters.py tests/unit/test_formatters.py
git commit -m "cli: unify list GPU fields and emit per-mode image_tag in info"
```

---

## Task 5: Full verification

**Files:** none (verification only).

- [ ] **Step 1: Run the full unit suite**

Run: `python -m pytest tests/unit -q -m "not docker and not gpu"`
Expected: PASS (no failures, no errors, no collection errors).

- [ ] **Step 2: Lint and format check**

Run: `ruff check src/ tests/ && ruff format --check src/ tests/`
Expected: no findings; formatting clean.

- [ ] **Step 3: Type check**

Run: `mypy src/`
Expected: no new errors versus `main` (`parse_output`'s widened `InverseFoldingOutput | ScoringOutput` return type type-checks; `getattr`-based antibody config type-checks).

- [ ] **Step 4: Confirm removed names are gone and the catalog is consistent**

Run:
```bash
grep -rn "ligandmpnn_build_mutant\|AntibodyInput\b" src/ tests/   # expect: no output
python -c "from autobio.core.catalog import CATALOG, tool_categories; \
t=CATALOG['ligandmpnn']; assert set(t.modes)=={'design','build_mutant'}; \
assert tuple(c.value for c in tool_categories(t))==('inverse-folding','scoring'); \
assert 'ligandmpnn_build_mutant' not in CATALOG; \
assert set(CATALOG['esm2'].modes)=={'embedding'}; \
print('catalog OK; tools:', len(CATALOG))"
```
Expected: no grep output; prints `catalog OK; tools: 27` (28 → 27 after LigandMPNN consolidation).

---

## Self-Review

**Spec coverage:**
- CLEANUP.md §1 (per-mode antibody schemas) → Task 1 (schema split, runner rewiring, per-mode `input_schema`, mode-scoped validation, integration-test updates, "no leaked field" assertions).
- CLEANUP.md §2 (consolidate LigandMPNN) → Task 2 (one Tool + two modes, helper extraction, `MPNNRunner` delegation, `TOOL_RUNNERS` cleanup, category union, image override, tests + integration).
- CLEANUP.md §3 (embed→embedding) → Task 3 (six literals, container-contract confirmation, unit + integration tests).
- CLEANUP.md §4 (CLI JSON contract) → Task 4 (`list` `requires_gpu`+`gpu_count`, `info` per-mode `image_tag`, tests).
- CLEANUP.md "Testing notes" bullets #1–#4 → covered by the assertions added in each task; catalog-wide consistency checked in Task 5.
- Non-goals respected: no container edits, no output-schema changes, no compat shims, no mode-naming audit beyond `embed`→`embedding`, FASTA-vs-schema untouched.

**Type consistency:** helper names (`prepare_build_mutant`, `parse_build_mutant_output`, `BUILD_MUTANT_NOTES`) are used identically in `ligandmpnn_packer.py` (definition) and `mpnn.py` (import + call). Schema names (`AntibodyBaseInput`/`AntibodyEmbeddingInput`/`AntibodyPLLInput`) match across `antibody.py`, `schemas/__init__.py`, `antibody_lm.py`, and the tests. `MPNNRunner.parse_output` return type `InverseFoldingOutput | ScoringOutput` matches `parse_build_mutant_output`'s `ScoringOutput`.

**Placeholder scan:** every source step contains the literal code; test steps give full code for semantic changes and precise, enumerated find/replace rules (with the exception lists) for mechanical repeats. No TBD/TODO left.
