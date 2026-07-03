# Tools→Modes Teardown — Cleanups + Docs (Plan) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** The second and final teardown PR. Remove the 3 now-unused category *Input* schemas, apply the accumulated behavior-preserving carry-forward refactors, tidy test-import hygiene, and rewrite the user/dev docs (README + `TOOL_SPEC.md` + `SCHEMA_SPEC.md`) to the catalog `Tool`/`Mode` world.

**Architecture:** All five tasks are behavior-preserving cleanup or docs — none change any container `config.json` or generated input file. PR #20 already removed the legacy registry; this PR removes leftover dead schema classes, de-duplicates a helper, and updates prose.

**Tech Stack:** Python 3.11+, Pydantic, pytest, Markdown.

**Recon fact sheet (READ IT):** `.superpowers/sdd/recon/teardown.md` — §3 (unused schemas, exact consumer counts), §4 (carry-forwards with file:line), §5 (README/docs stale-name inventory with line ranges), §4g (test-import hygiene file list).

## Global Constraints

- **No `config.json` / generated-input changes.** Every code task here is a pure refactor or dead-code removal; behavior, error messages, and emitted files must be identical. (No byte-compat risk, but preserve error-message text where refactoring validation.)
- **Out of scope (explicitly EXCLUDED from this PR):** the openfold3/chai `templates`/`msa_paths` latent bug — the user will fix it (wiring into the container payload) as a dedicated follow-up immediately after this PR, so do NOT document-or-wire it here. Do NOT touch `containers/`. Do NOT touch the untracked `docs/CREDIT_SYSTEM.md`.
- **Keep the *Output* siblings.** Only the three *Input* classes are dead; `StructurePredictionOutput`/`StructureDesignOutput`/`EmbeddingOutput` (and their nested models) are used — do not remove them.
- Env: `python -m pytest` (bare = wrong env); this config omits the "N passed" line — verify via exit code. Every commit green.

---

## Task 1: Remove the 3 unused category Input schemas

**Files:** `src/autobio/schemas/structure_prediction.py`, `src/autobio/schemas/structure_design.py`, `src/autobio/schemas/embedding.py`, `src/autobio/schemas/__init__.py`, `tests/unit/test_schemas.py`

Recon §3 confirms zero production consumers for `StructurePredictionInput`, `StructureDesignInput`, `EmbeddingInput` (every tool uses its own dedicated input schema now).

- Delete the `class StructurePredictionInput(...)`, `class StructureDesignInput(...)`, and `class EmbeddingInput(...)` definitions. KEEP everything else in those files (the Output classes + nested models). Remove any imports that only served the deleted Input class (e.g. if `BaseInput` becomes unused in a file — check each; several files still use `BaseInput` for the Output? no, outputs use `BaseOutput` — verify and drop truly-unused imports).
- `schemas/__init__.py`: remove the three names from both the imports and `__all__`.
- `tests/unit/test_schemas.py`: delete the test classes/functions that construct/round-trip these three Input classes (recon §3: ~7 hits each). Leave all other schema tests intact.
- Verify no dangling references: `grep -rn "StructurePredictionInput\|StructureDesignInput\|EmbeddingInput" src/ tests/` → zero.
- Commit: `schemas: remove unused category Input classes (StructurePrediction/StructureDesign/Embedding)`

---

## Task 2: Hoist `_resolve_container_path` onto `ToolRunner`

**Files:** `src/autobio/tools/base.py` + the 12 runner files listed in recon §4a (`openmm.py`, `boltz.py`, `evoef2.py`, `esmfold.py`, `openfold3.py`, `esm.py`, `ligandmpnn_packer.py`, `rfd3.py`, `chai.py`, `rosetta.py`, `antibody_lm.py`, `complexa.py`)

All 12 define an identical path-mapping helper (11 as `@staticmethod`, 1 as a module-level function in `ligandmpnn_packer.py`). Consolidate:

- Add to `ToolRunner` in `base.py` (needs `from pathlib import Path` at runtime — it's currently under `TYPE_CHECKING`; move `Path` to a runtime import or reference it appropriately):
  ```python
  @staticmethod
  def _resolve_container_path(container_path_str: str, workspace: Workspace) -> Path:
      """Map a container-internal ``/workspace/...`` path to the host workspace."""
      container_path = Path(container_path_str)
      try:
          relative = container_path.relative_to("/workspace")
      except ValueError:
          return container_path
      return workspace.root / relative
  ```
- Delete the 11 `@staticmethod _resolve_container_path` copies from the runner files (they inherit the base method). Before deleting each, confirm its body is logically identical (recon §4a says all 11 are; a few have a longer docstring — fine).
- `ligandmpnn_packer.py`: delete the module-level `_resolve_container_path` function (line ~153) AND change the call site (line ~108) from `_resolve_container_path(...)` to `self._resolve_container_path(...)`.
- Confirm `Path` import in each edited runner file is still needed (many use it elsewhere); only remove if now unused.
- Verify: `grep -rn "def _resolve_container_path" src/autobio/tools/` → only `base.py`. Full suite green (the output-path-resolution tests for boltz/openmm/etc. still pass via inheritance).
- Commit: `tools: hoist _resolve_container_path onto ToolRunner; drop 12 duplicates`

---

## Task 3: Runner refactors — esm_if1/antifold staging + explicit branch; dead `_ROSETTA_BIN`; rfd3/complexa dedup

**Files:** `src/autobio/tools/esm_if1.py`, `src/autobio/tools/antifold.py`, `src/autobio/tools/rosetta.py`, `src/autobio/tools/rfd3.py`, `src/autobio/tools/complexa.py` (+ their unit tests if any assertion is affected)

All behavior-preserving. (Recon §4b/§4c/§4d/§4f.)

**3a. esm_if1 + antifold — hoist staging block + make the score branch explicit.**
Both `prepare_workspace` methods repeat this 3-line block once per mode branch (recon §4c):
```python
src_path = input_data.structure_path
dest_name = src_path.name
shutil.copy2(src_path, workspace.inputs_dir / dest_name)
```
Hoist it ABOVE the `if self.current_mode.name == "design":` branch (both `InverseFoldingInput` and `ScoringInput` have `structure_path`, so the copy is mode-agnostic). Then in BOTH `prepare_workspace` and `parse_output`, replace the implicit `else:  # "score"` with an explicit guard so a future third mode can't silently take the score path — e.g.:
```python
if self.current_mode.name == "design":
    ...
else:
    assert self.current_mode.name == "score", self.current_mode.name
    ...
```
(Do NOT change the config dict contents or key order — byte-compat. The `assert` is the only added line in the score branch; the staging lines move up, not change.) For antifold, keep the unconditional `_validate_chain_ids(...)` call where it is (before the branch).

**3b. Delete dead `_ROSETTA_BIN`.** `rosetta.py:45` — the constant is defined and never used (recon §4d confirmed; `_ROSETTA_DB` right below IS used — keep it). Delete only the `_ROSETTA_BIN = ...` line.

**3c. Dedupe rfd3/complexa "no matching file" reference check.** Both `rfd3.py` and `complexa.py` validate that each `design_specs[...]["input"]` filename matches a provided structure file TWICE — once in `prepare_workspace` (while rewriting paths) and once in `_validate_inputs` (recon §4f, exact line ranges). Extract the check into a small private helper on each runner (e.g. `_check_input_references(self, design_specs, provided_names) -> None`) and call it from both sites, so the logic + error message exist once. **Preserve the exact error message(s)** currently raised. Do not merge across the two runners (their schemas differ) — one helper per runner file.

Run the esm_if1/antifold/rosetta/rfd3/complexa unit tests + full suite; all green (byte-compat config tests must still pass unchanged). Commit: `tools: dedup esm_if1/antifold staging + rfd3/complexa ref-check; drop dead _ROSETTA_BIN`

---

## Task 4: Test-import hygiene — populate the catalog once via conftest

**Files:** `tests/conftest.py` + the unit test files carrying per-method `import autobio.tools` (recon §4g: 133 occurrences across 21 files)

- In `tests/conftest.py`, add a module-level `import autobio.tools  # noqa: F401 - populate the CATALOG for the whole test session` (conftest is imported before any test, so this registers all tools once).
- Remove the redundant per-method/per-function `import autobio.tools  # noqa: F401[ - ...]` lines from the unit test files. **Only remove the generic catalog-population import** (`import autobio.tools` with an F401 noqa). **LEAVE** the two aliased submodule imports (`import autobio.tools.esm_if1 as esm_if1_module`, `import autobio.tools.antifold as antifold_module`) — those are used with their alias for a specific purpose, not catalog population.
- After removal, if a test method's only remaining first line was that import, ensure the method body is still valid (it will be — the import was side-effect only).
- Verify: `grep -rn "^\s*import autobio.tools  # noqa" tests/unit/` → zero (only the aliased imports remain); full suite green (catalog still populated via conftest).
- Commit: `tests: populate CATALOG once via conftest; drop 133 per-method tool imports`

---

## Task 5: Rewrite user + dev docs to the catalog Tool/Mode model

**Files:** `README.md`, `src/autobio/tools/TOOL_SPEC.md`, `src/autobio/schemas/SCHEMA_SPEC.md`

Recon §5 has the exact stale-name inventory + line ranges. Do NOT touch the historical design docs (`docs/WORKPLAN.md`, `docs/DESIGN.md`, `docs/REFACTOR.md`, `docs/superpowers/**`) or the untracked `docs/CREDIT_SYSTEM.md`.

**5a. `README.md`:**
- Fix every stale flat tool name in the tool tables (recon §5a): the tables currently list names that no longer exist as invocable tools. Rewrite them as `<tool>` + `--mode <mode>`:
  - Structure design: `complexa_ligand`/`complexa_ame` → `complexa --mode ligand_binder` / `--mode ame`.
  - Embedding: the six `*_pll` rows → `<model> --mode pll` (esm1b/esm2 rows are correct — leave).
  - Structure utilities: `evoef2_build_mutant`/`evoef2_repair` → `evoef2 --mode build_mutant`/`--mode repair` (`ligandmpnn_build_mutant` is a real tool — leave).
  - Energy minimization: `rosetta_relax`/`rosetta_minimize`/`openmm_amber_minimize`/`openmm_amber_relax` → `rosetta --mode relax`/`--mode minimize`, `openmm --mode amber_minimize`/`--mode amber_relax`.
  - Scoring: `rosetta_score`/`rosetta_flexddg`/`evoef2_binding`/`antifold_score`/`esm_if1_score` → the corresponding tool + `--mode`. (The `freesasa` row is already mode-aware — use it as the style template.)
  - Simulation: `openmm_md_simulate` → `openmm --mode md_simulate`.
- Add `--mode` to the CLI reference (README ~lines 69–84 and `src/autobio/cli/README.md`): document `autobio run <tool> --mode <mode>` (default mode used if omitted) and, for `autobio list`/`autobio info`, that modes appear per-Tool. Keep the accurate `proteinmpnn` quick-start/Python-API examples as-is.
- Verify every tool/mode name you write actually exists: cross-check against `python -c "import autobio.tools; from autobio.core.catalog import CATALOG; [print(n, sorted(t.modes)) for n,t in sorted(CATALOG.items())]"`.

**5b. `src/autobio/tools/TOOL_SPEC.md`** — the authoritative "Adding a New Tool" dev spec, currently written entirely against `TOOL_REGISTRY`/`ToolEntry` (recon §1b). Rewrite the walkthrough to the catalog pattern: define a `Tool` with its `modes={...}` `Mode`s and `register(TOOL)` in the runner module; runner subclasses `ToolRunner` implementing `prepare_workspace`/`parse_output` and dispatching on `self.current_mode.name`; per-mode `input_schema`/`output_schema`/`default_timeout`/optional `image_tag`/`category`; `_apply_extra` for the `extra` escape hatch; wire the runner into `TOOL_RUNNERS`. Use an existing simple catalog tool (e.g. `esmfold` single-mode, or `freesasa` multi-mode) as the reference example. Align with the current `CLAUDE.md` "Adding a New Tool" summary.

**5c. `src/autobio/schemas/SCHEMA_SPEC.md`** — replace the `TOOL_REGISTRY["diffdock"] = ToolEntry(...)` example snippet (recon §1b, lines ~201/204/313) with the catalog equivalent (`register(Tool(... modes={...}))`), and update any surrounding prose that assumes the flat registry.

Commit: `docs: rewrite README + TOOL_SPEC + SCHEMA_SPEC to the catalog Tool/Mode model`

---

## Self-Review checklist (controller, before dispatch)
- [ ] Tasks 1–4 change no `config.json`/generated files; error messages preserved in Task 3c.
- [ ] Task 2: only `base.py` defines `_resolve_container_path`; ligandmpnn call site updated.
- [ ] Task 4: only the generic catalog-population imports removed; aliased submodule imports kept; conftest populates CATALOG.
- [ ] Task 5: every tool/mode name in README verified against CATALOG; historical docs + CREDIT_SYSTEM.md untouched.
- [ ] templates/msa bug NOT touched (user's dedicated follow-up); `containers/` untouched.
