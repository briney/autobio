# OpenFold3/Chai templates + MSA bug fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Fix the latent bug where `OpenFold3Input.templates`/`msa_paths` and `Chai1Input.templates` are copied into the workspace and validated but never wired into the container payload (so they're silently ignored). Resolution, per ground-truth recon: **wire `openfold3.msa_paths`** (genuinely supported by the tool) and **remove the two `templates` fields** (the tools accept only template-*hits* alignment files, not raw structure files — the fields have never worked, and openfold3's notes falsely claim they do).

**Architecture:** Entirely host + schema + tests — **NO `containers/` changes.** For `msa_paths`: the host writes `main_msa_file_paths` into the generated `query.json`; the openfold3 container already loads that JSON via `InferenceQuerySet.from_json`, whose `Chain` model natively supports `main_msa_file_paths` — so no container edit is needed. For `templates`: it's a removal. boltz is the working reference for the MSA path-rewrite pattern.

**Tech Stack:** Python 3.11+, Pydantic, pytest.

**Recon fact sheet (READ IT):** `.superpowers/sdd/recon/templates_msa.md` — ground-truth analysis (pulled from the built Docker images' vendored source). §2 (openfold3 gap + the real `Chain.main_msa_file_paths` API), §3 (chai), §4 (feasibility verdict per field), §5 (test impact).

## Global Constraints

- **NO `containers/` changes.** msa_paths flows through the existing query.json → container `InferenceQuerySet.from_json` path; templates is a removal.
- **Wire only what the tool truly supports.** `openfold3.msa_paths` → per-chain `main_msa_file_paths` in query.json (real API). Do NOT attempt to wire either `templates` field (the tools need a `.m8`/`.a3m`/`.sto` template-*hits* file from a homology search, not raw structures — see recon §4).
- **Byte-compat where unchanged:** the `msa_paths` wiring changes `query.json` ONLY when `msa_paths` is provided; existing config/query byte-compat tests (no msa_paths) must stay green. Removing `templates` must not change config for any input (templates never appeared in config).
- Keep the server-side template toggles (`OpenFold3Input.use_templates`, `Chai1Input.use_templates_server`) — those work and are separate from the removed raw-file `templates` field.
- Env: `python -m pytest` (bare = wrong env); this config omits the "N passed" line — verify via exit code. Every commit green.

---

## Task 1: Wire `openfold3.msa_paths` into the query JSON

**Files:** `src/autobio/tools/openfold3.py`, `tests/unit/test_openfold3.py`

Per recon §2/§4: OpenFold3's query-JSON `Chain` model has `main_msa_file_paths` (accepts user `.a3m`/`.npz`/`.sto` files per chain). `prepare_workspace` already copies the MSA files into `inputs/`; only the query-JSON wiring is missing.

### Steps
1. In `_build_query_json` (currently `@staticmethod`, takes `input_data`): build a chain-ID→container-path map from `input_data.msa_paths`, mirroring boltz's convention (MSA filename stem = chain ID; e.g. `A.a3m` → chain `A`), pointing at `/workspace/inputs/<name>`. For each protein/DNA/RNA chain whose `chain_ids` is in the map, set `chain["main_msa_file_paths"] = ["/workspace/inputs/<name>"]`. (Ligand chains get no MSA.) Keep the existing chain construction otherwise unchanged. Example:
   ```python
   msa_map: dict[str, str] = {}
   for msa_path_str in input_data.msa_paths or []:
       stem = Path(msa_path_str).stem
       msa_map[stem] = f"/workspace/inputs/{Path(msa_path_str).name}"
   # ... when building each protein/dna/rna chain:
   if chain_id in msa_map:
       chain["main_msa_file_paths"] = [msa_map[chain_id]]
   ```
   (The MSA files are already copied to `inputs/` in `prepare_workspace` — keep that copy loop.)
2. **`use_msa_server` mutual-exclusion:** OpenFold3's ColabFold step overwrites `main_msa_file_paths` when `use_msa_server=True` (recon §2 caveat). So in `_validate_inputs`, raise `AutobioError` if `input_data.msa_paths` is truthy AND `input_data.use_msa_server` is True — mirroring chai's `use_msa_server`/`msa_directory` mutual-exclusion. Message e.g.: `"Cannot provide msa_paths with use_msa_server=True — set use_msa_server=False to use precomputed MSAs."` (Keep the existing "MSA file does not exist" validation.)
3. **Note update:** in `_OPENFOLD3_NOTES`, rewrite the MSA paragraph to spell out the chain-ID-stem filename convention (like boltz's note) and the `use_msa_server=False` requirement when providing `msa_paths`.
4. **Tests** (`test_openfold3.py`):
   - Extend `test_msa_files_copied` (or add a new test): assert the generated `query.json` places `main_msa_file_paths == ["/workspace/inputs/A.a3m"]` on the chain whose id matches the MSA filename stem (follow the shape of `test_generated_query_json_equality_*`). Add a full query-JSON byte-compat case WITH `msa_paths` (+ `use_msa_server=False`).
   - Add a validation test: `msa_paths` + `use_msa_server=True` raises `AutobioError` (match the new message).
   - Confirm the existing `test_config_full_dict_equality_*` and `test_generated_query_json_equality_*` (no msa_paths) still pass unchanged (msa wiring is conditional).
   - `test_msa_paths_via_extra_rejected` stays green.
5. Commit: `openfold3: wire msa_paths into query.json main_msa_file_paths`

Run: `python -m pytest tests/unit/test_openfold3.py -q` then full `-m "not docker and not gpu"` (exit 0); `ruff check src/ tests/`, `ruff format --check`, `mypy src/`.

---

## Task 2: Remove the inert `templates` fields (openfold3 + chai) + fix the false note

**Files:** `src/autobio/schemas/structure_prediction.py`, `src/autobio/tools/openfold3.py`, `src/autobio/tools/chai.py`, `tests/unit/test_openfold3.py`, `tests/unit/test_chai.py`

Per recon §4: neither tool can consume raw user template structure files (they need a template-hits alignment file + PDB-resolvable structures — a search-pipeline artifact). The fields have never worked. Remove them.

### Steps
1. **Schema** (`structure_prediction.py`): delete the `templates: list[Path] | None` field from `OpenFold3Input` (~line 267) and from `Chai1Input` (~line 201). Leave `use_templates` (OpenFold3Input) and `use_templates_server` (Chai1Input) — those are the working server-side toggles. Leave `BoltzInput.templates` untouched (boltz genuinely supports it).
2. **`openfold3.py`:**
   - Remove the template-copy block in `prepare_workspace` (~lines 82-84) and the template-existence validation in `_validate_inputs` (~lines 230-233).
   - Fix `_OPENFOLD3_NOTES`: rewrite the "Template options" paragraph (~lines 298-302) to remove the FALSE claim "To provide custom template structures, use the 'templates' field." Keep accurate guidance: server-side templates are enabled by default via `use_templates` (ColabFold retrieves them); set `use_templates=false` to disable. Do NOT claim user-supplied template files are supported.
3. **`chai.py`:** remove the template-copy block in `prepare_workspace` (~lines 94-96) and the template-existence validation in `_validate_inputs` (~lines 214-218). (Chai's `_CHAI_NOTES` does not claim template support, so no note fix needed — verify by grep.)
4. **Tests:**
   - `test_openfold3.py`: remove `test_templates_copied` (~275-290) and `test_missing_template_file_raises` (~the template one). Remove any `templates=[...]` usage from other test inputs. `test_use_templates_via_extra_rejected` stays (it guards the `use_templates` bool, which remains).
   - `test_chai.py`: remove `test_templates_copied` (~253-269) and `test_missing_template_file_raises` (~431-447). Remove any `templates=[...]` usage.
   - Confirm no test still constructs `OpenFold3Input(templates=...)` or `Chai1Input(templates=...)`.
5. Verify: `grep -rn "templates" src/autobio/tools/openfold3.py src/autobio/tools/chai.py src/autobio/schemas/structure_prediction.py` shows no `OpenFold3Input`/`Chai1Input` `templates` field usage (only `BoltzInput.templates` + `use_templates`/`use_templates_server` remain); `grep -rn "use the 'templates' field" src/` → zero.
6. Commit: `openfold3/chai: remove inert templates fields (raw structures unsupported by tools); fix false note`

Run: `python -m pytest tests/unit/test_openfold3.py tests/unit/test_chai.py tests/unit/test_boltz.py -q` then full `-m "not docker and not gpu"` (exit 0); `ruff check src/ tests/`, `ruff format --check`, `mypy src/`.

---

## Self-Review checklist (controller, before dispatch)
- [ ] Task 1: msa_paths wiring is conditional (existing byte-compat tests unaffected); use_msa_server mutual-exclusion added; note updated; NO container change.
- [ ] Task 2: only `OpenFold3Input`/`Chai1Input` `templates` removed; `BoltzInput.templates` + `use_templates`/`use_templates_server` kept; false openfold3 note fixed.
- [ ] No `containers/` files touched. Full suite green.
