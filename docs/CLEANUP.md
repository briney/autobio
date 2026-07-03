# autobio Tool→Modes cleanup

**Status:** Design (approved direction)
**Date:** 2026-07-03
**Related:** `docs/REFACTOR.md` (the Tool→Modes refactor this cleans up after)

## Context

The Tool→Modes refactor (`docs/REFACTOR.md`, shipped in PRs #16–#21) is implemented and
merged: a `Tool`/`Mode` catalog replaces the flat `TOOL_REGISTRY`, `info --format json`
returns a `modes[]` array with a resolved per-mode JSON Schema, `run` takes a `--mode`
flag, `x-autobio` UI hints ride in the schemas, and FASTA (`>pair_id|chain`) pairing is
normalized server-side.

A review of the as-built package against the refactor's locked decisions surfaced four
gaps/inconsistencies. This document specs the fixes. They are the last conformance items
before the downstream consumer (fold@Scripps) builds its Tool→Modes UI against this
contract.

**No backward compatibility is required.** autobio is single-owner; every change here is a
clean break, and the downstream fold@Scripps consumer will adapt in its own spec (see
"Downstream impact"). No deprecation shims.

## Summary of changes

| # | Change | Priority | Kind |
|---|--------|----------|------|
| 1 | Split the antibody-LM `AntibodyInput` into per-mode schemas (shared base + mode fields) | High | Schema + runner |
| 2 | Consolidate `ligandmpnn` + `ligandmpnn_build_mutant` into one Tool with modes `{design, build_mutant}` | Medium | Catalog + runner |
| 3 | Normalize the embedding mode name: rename ESM's `embed` → `embedding` | Low | Catalog |
| 4 | Make the CLI JSON contract self-consistent: unify the `list` GPU field; emit per-mode `image_tag` in `info` | Cosmetic | Formatters |

---

## 1. Per-mode antibody-LM schemas

### Problem

All six antibody LMs (`currab`, `ft_esm`, `balm_paired`, `balm_unpaired`, `ablang2`,
`antiberta2`) declare both their `embedding` and `pll` modes with the **same**
`input_schema=AntibodyInput` (`src/autobio/tools/antibody_lm.py:330-377`). That one schema
bundles fields from both tasks:

- `sequences` — shared (required).
- `layer`, `pooling` — embedding-only ("Only used in embedding mode").
- `per_position` — pll-only ("pll mode only").

So each mode's resolved `input_schema` leaks the other mode's fields. This violates the
refactor's per-mode-schema decision (Q1: "each mode's input schema = shared tool base +
mode-specific fields") and forces a schema-driven consumer to render irrelevant controls
(e.g. `layer`/`pooling` under the `pll` form). Every other multi-mode tool already ships
distinct per-mode schemas (`evoef2`, `freesasa`, `rosetta`, `antifold`, `esm_if1`) — the
antibody LMs are the sole exception.

### Design

Split `AntibodyInput` (`src/autobio/schemas/antibody.py`) into a shared base plus two
mode-specific subclasses:

```python
class AntibodyBaseInput(BaseInput):
    """Shared antibody-LM input: the sequence set (plus inherited `extra`)."""
    sequences: AntibodySequenceSet = Field(
        description="One or more antibody sequences: a list of AntibodySequence/dicts, "
                    "FASTA text, or a path to a .fasta/.fa file.",
        json_schema_extra=ui(widget=Widget.SEQUENCE, flavor="antibody", tier=Tier.PRIMARY, order=0),
    )

class AntibodyEmbeddingInput(AntibodyBaseInput):
    """Input for the `embedding` mode."""
    layer: int | None = Field(default=None, description="Model layer to extract embeddings "
        "from. None uses the final layer.",
        json_schema_extra=ui(widget=Widget.NUMBER, tier=Tier.ADVANCED, order=10))
    pooling: str | None = Field(default=None, description="Pooling strategy for per-residue "
        "embeddings ('mean', 'cls', 'per_residue').",
        json_schema_extra=ui(widget=Widget.SELECT, tier=Tier.PRIMARY, order=1,
            enum_labels={"mean": "Mean pool", "cls": "CLS token", "per_residue": "Per-residue"}))

class AntibodyPLLInput(AntibodyBaseInput):
    """Input for the `pll` (pseudo log-likelihood) mode."""
    per_position: bool = Field(default=False, description="Return per-position PLL scores. Slower.",
        json_schema_extra=ui(widget=Widget.TOGGLE, tier=Tier.ADVANCED, order=11))
```

Drop the "Only used in embedding mode" / "pll mode only" caveats from the descriptions —
the field only exists in the mode it applies to now.

Point each mode at its schema in `_register_antibody_lm` (`antibody_lm.py:330-377`):

```python
"embedding": Mode(..., input_schema=AntibodyEmbeddingInput, output_schema=EmbeddingOutput, ...),
"pll":       Mode(..., input_schema=AntibodyPLLInput,       output_schema=AntibodyPLLOutput, ...),
```

### Runner changes (`AntibodyLMRunner`)

`prepare_workspace` (`antibody_lm.py:140-179`) and `_validate_inputs` (`:285-296`) currently
read `input_data.layer` / `input_data.pooling` / `input_data.per_position` unconditionally
off a single `AntibodyInput`. With split schemas those attributes no longer exist on both
inputs, so:

- **Preserve the container contract.** The container's `config.json` today always carries
  `mode`, `layer`, `pooling`, `per_position`. Keep that shape — the container script is out
  of scope — by sourcing each mode's fields from the typed input and defaulting the rest:

  ```python
  mode = self.current_mode.name
  config = { ...common..., "mode": mode,
             "layer":        getattr(input_data, "layer", None),
             "pooling":      getattr(input_data, "pooling", None) or "mean",
             "per_position": getattr(input_data, "per_position", False) }
  ```

  (Or branch on `isinstance(input_data, AntibodyEmbeddingInput | AntibodyPLLInput)` if you
  prefer explicit narrowing over `getattr`.)
- **Scope the embedding-only validation.** Guard the `layer`/`pooling` checks in
  `_validate_inputs` behind the embedding mode (they're absent in `pll`).
- `_is_pll_mode()` / `parse_output` (`:181-193`) are unchanged — they already branch on
  `self.current_mode.name`.
- `_apply_extra` needs no change: it already keys off
  `self.current_mode.input_schema.model_fields`, so per-mode typed-field collision checks
  stay correct automatically.

### Affected files

- `src/autobio/schemas/antibody.py` — split the model.
- `src/autobio/tools/antibody_lm.py` — mode `input_schema=` wiring; `prepare_workspace` /
  `_validate_inputs` mode-awareness.
- Tests covering antibody-LM schemas / config generation.

---

## 2. Consolidate LigandMPNN into one Tool

### Problem

LigandMPNN is registered as **two** Tools:

- `ligandmpnn` (category `inverse-folding`, single mode `design`, `MPNNRunner`,
  `MPNNInput → InverseFoldingOutput`, image `mpnn:1.0.0`) — `src/autobio/tools/mpnn.py:180-205`.
- `ligandmpnn_build_mutant` (category `scoring`, single mode `build_mutant`,
  `LigandMPNNPackerRunner`, `LigandMPNNPackerInput → ScoringOutput`, image
  `ligandmpnn-packer:1.0.0`) — `src/autobio/tools/ligandmpnn_packer.py:171-201`.

The refactor's migration table (`REFACTOR.md` §10) and the "model = card" decision call for
**one** `ligandmpnn` Tool with cross-category modes `{design, build_mutant}`. The current
split (`ligandmpnn_build_mutant`) is exactly the `(model × task)` flat naming the refactor
set out to eliminate; `antifold`, `esm_if1`, and `openmm` were consolidated correctly but
LigandMPNN was missed.

### Constraint (verified)

autobio binds **one runner class per tool name**: `TOOL_RUNNERS: dict[str, type[ToolRunner]]`
(`src/autobio/tools/__init__.py`) has no mode dimension, and `get_runner(tool_name, config)`
instantiates a single class. `run()` (`src/autobio/tools/base.py`) only sets
`self.current_mode` and calls that one instance's `prepare_workspace`/`parse_output`. So a
single Tool's modes must all be served by one runner that branches internally — the
established idiom (`AntibodyLMRunner` branches on mode; `MPNNRunner`/`ESMRunner` branch on
`tool_name`).

### Design (chosen approach: keep `MPNNRunner` shared, delegate `build_mutant` to helpers)

Register one Tool:

```python
LIGANDMPNN_TOOL = Tool(
    name="ligandmpnn", display_name="LigandMPNN",
    category=ToolCategory.INVERSE_FOLDING,          # primary category
    version="1.0.0", image_tag="mpnn:1.0.0",        # tool-default image (design)
    requires_gpu=True, gpu_count=1,
    default_mode="design",
    modes={
        "design": Mode(
            name="design", display_name="Design sequences",
            description="Design protein sequences (ligand-aware) for a backbone structure.",
            input_schema=MPNNInput, output_schema=InverseFoldingOutput,
            default_timeout=600, notes=_MPNN_NOTES + (_LIGANDMPNN_LIGAND_NOTE,)),
        "build_mutant": Mode(
            name="build_mutant", display_name="Build mutant",
            description="Introduce mutations and repack sidechains into full-atom structures.",
            input_schema=LigandMPNNPackerInput, output_schema=ScoringOutput,
            default_timeout=600,
            image_tag="ligandmpnn-packer:1.0.0",     # per-mode image override
            category=ToolCategory.SCORING,            # cross-category mode
            notes=_PACKER_NOTES),
    },
    keywords=("ligandmpnn", "inverse folding", "sequence design", "ligand", "mpnn",
              "mutant", "sidechain packing", "repack", "mutation"),
)
```

`tool_categories(LIGANDMPNN_TOOL)` then yields `[inverse-folding, scoring]` (union of the
tool's primary + the `build_mutant` mode override), so it surfaces under both categories in
`list`/`info` — no extra work, the catalog already computes this.

**Runner wiring:**

- `TOOL_RUNNERS`: delete the `"ligandmpnn_build_mutant"` entry. `"ligandmpnn"` stays mapped
  to `MPNNRunner` (which also serves `proteinmpnn`).
- Delete the `LigandMPNNPackerRunner` **class** and the `ligandmpnn_build_mutant` Tool
  registration. Extract the packer's `prepare_workspace` / `parse_output` / `_validate_inputs`
  bodies into **module-level helper functions** (e.g. keep `ligandmpnn_packer.py` but export
  `prepare_build_mutant(input_data, workspace)` / `parse_build_mutant_output(workspace)`
  instead of a `ToolRunner` subclass).
- `MPNNRunner` (`mpnn.py`) gains an early mode guard in each hook; the design path (which
  already branches on `tool_name` for proteinmpnn vs ligandmpnn) is otherwise untouched:

  ```python
  def prepare_workspace(self, input_data, workspace):
      if self.current_mode.name == "build_mutant":
          return prepare_build_mutant(input_data, workspace)
      ... existing design logic ...

  def parse_output(self, workspace):
      if self.current_mode.name == "build_mutant":
          return parse_build_mutant_output(workspace)          # -> ScoringOutput
      ... existing design logic ...                            # -> InverseFoldingOutput
  ```

  `parse_output`'s return type widens to `InverseFoldingOutput | ScoringOutput` (as
  `AntibodyLMRunner` already returns a union).

- `proteinmpnn` is unaffected: it declares only a `design` mode, so it never reaches the
  `build_mutant` branch.

**Framework already carries the per-mode differences:** the `build_mutant` container image
resolves via `Mode.image_tag` in `_image_tag()` (`base.py:222-225`,
`mode.image_tag or tool.image_tag`); the differing input schema is validated in `run.py`
against `mode.input_schema` before the runner is called; and `_apply_extra` keys off the
active mode's schema.

### Affected files

- `src/autobio/tools/mpnn.py` — one `ligandmpnn` Tool with two modes; mode guards in the
  runner hooks.
- `src/autobio/tools/ligandmpnn_packer.py` — convert the runner class to helper functions;
  drop the second Tool registration.
- `src/autobio/tools/__init__.py` — remove `ligandmpnn_build_mutant` from `TOOL_RUNNERS`
  (and the `LigandMPNNPackerRunner` import).
- Tests: port `LigandMPNNPackerRunner` tests to the extracted helpers; assert `ligandmpnn`
  exposes both modes and the `{inverse-folding, scoring}` category union; assert
  `proteinmpnn` still works.

### Result

Tool count drops 28 → 27. `ligandmpnn_build_mutant` disappears as a tool name; the packing
task is addressed as `autobio run ligandmpnn --config … --mode build_mutant`.

---

## 3. Normalize the embedding mode name

### Problem

The "extract embeddings" task is named inconsistently: `embed` on the ESM tools
(`esm1b`, `esm2` — `src/autobio/tools/esm.py:186-198, 217-229`) but `embedding` on the six
antibody LMs (`antibody_lm.py`). A consumer keying UI/logic on the mode-name string must
handle both spellings for the same concept.

### Design

Rename ESM's mode `embed` → `embedding`, so every embedding-capable tool uses `embedding`
(which also matches the `ToolCategory.EMBEDDING` value `"embedding"`). In `esm.py`, update
the three literals per tool — `default_mode="embed"`, the `modes` dict key `"embed"`, and
`Mode(name="embed", …)` — for both `esm1b` and `esm2` (six literals total). `ESMRunner`
does not reference the mode name (it branches on `self.tool.name`), so there is no host-side
runner change.

**Verify the container contract before renaming.** The mode name is primarily a host-side
catalog label, but confirm the ESM container script does not key on the literal string
`"embed"` (e.g. via a `mode` field in `config.json`). If it does, either the container
must be updated in lockstep or `ESMRunner.prepare_workspace` must map the catalog mode name
to whatever token the container expects.

Scope is limited to the `embed`↔`embedding` duplication. Other mode names (`predict`,
`score`, `generate`, `repair`, `design`, …) name genuinely distinct tasks and are left
as-is; a broader mode-naming taxonomy audit is a non-goal.

### Affected files

- `src/autobio/tools/esm.py` — six string literals.
- Tests asserting ESM mode names.

---

## 4. Self-consistent CLI JSON contract

### Problem

Two cosmetic inconsistencies in `src/autobio/cli/formatters.py`:

1. **GPU field naming diverges between `list` and `info`.** `list` emits a single
   `"gpu": tool.requires_gpu` with no count (`:44-59`); `info` emits
   `"requires_gpu": tool.requires_gpu` **and** `"gpu_count": tool.gpu_count` (`:110-125`).
2. **`info` omits per-mode `image_tag`.** The only image in the output is the tool-level
   default (`:116`), but modes can override it (`Mode.image_tag`) — `rosetta`, `openmm`, and
   (after change #2) `ligandmpnn`'s `build_mutant` actually run different images than the
   tool default. The emitted metadata therefore misstates what a given mode runs.

### Design

1. In the `list` JSON rows, replace `"gpu": tool.requires_gpu` with
   `"requires_gpu": tool.requires_gpu` and add `"gpu_count": tool.gpu_count`, matching
   `info`. (Update the `list` table output for parity if desired; the table is not a
   machine contract.)
2. In the `info` per-mode dict (`:96-109`), add
   `"image_tag": mode.image_tag or tool.image_tag` so each mode reports the image it
   actually runs.

### Affected files

- `src/autobio/cli/formatters.py` — `list` and `info` JSON builders.
- Tests asserting the emitted JSON shape.

---

## Downstream impact (fold@Scripps)

These are intentional breaking changes to the `list`/`info`/`run` contract. The
fold@Scripps consumer (which shells the autobio CLI) must adapt in its own consumer spec —
independent of this work, its catalog sync is already broken against the merged refactor
(it still reads a top-level `input_schema`/`default_timeout`/`supports_batch`, which are now
per-mode). Specific items from this cleanup the consumer must track:

- **#2:** the tool name `ligandmpnn_build_mutant` no longer exists — address it as
  `ligandmpnn` + `--mode build_mutant`; expect `ligandmpnn` under two categories.
- **#3:** ESM's embedding mode is `embedding`, not `embed`.
- **#4:** `list` JSON key `gpu` becomes `requires_gpu` (+ new `gpu_count`); per-mode
  `image_tag` is now available in `info`.

## Non-goals

- No backward-compat shims or deprecation paths (single-owner package; consumer adapts).
- No changes to execution machinery (`container`/`gpu`/`workspace`), output schemas, or the
  CLI command surface beyond mode-name literals.
- No broader mode-naming taxonomy audit beyond `embed`→`embedding` (#3).
- **The FASTA-vs-schema question is not an autobio change.** The `sequences` JSON Schema
  advertises only the structured form while the `BeforeValidator` also accepts FASTA
  text/paths. This is resolved on the consumer side (fold defers shape-validation of
  `widget:"sequence"` fields to autobio, which re-validates server-side); no schema widening
  is specified here.

## Testing notes

- **#1:** assert the `embedding` mode's resolved schema has no `per_position` and the `pll`
  mode's has no `layer`/`pooling`; assert the runner writes an unchanged-shape `config.json`
  with mode-appropriate values.
- **#2:** assert `ligandmpnn` exposes `{design, build_mutant}`, the category union
  `{inverse-folding, scoring}`, and that `build_mutant` resolves the `ligandmpnn-packer`
  image; assert `proteinmpnn`/`ligandmpnn` design paths are unchanged; port the packer's
  runner tests onto the extracted helpers.
- **#3:** assert `esm1b`/`esm2` report mode `embedding`.
- **#4:** assert `list` JSON emits `requires_gpu` + `gpu_count`, and `info` modes emit an
  `image_tag` reflecting per-mode overrides.
