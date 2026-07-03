# Experiment Scope & Credit Budget System

## Motivation

Autobio provides a unified interface to 42 computational biology tools. To support agent-driven iterative experimentation (inspired by Karpathy's autoresearch), we need project-scoped constraints: limiting which tools an agent can use and how much compute it can consume per iteration. This prevents unbounded exploration and forces deliberate choices about tool usage within a credit budget that resets each iteration.

## Design Overview

A single `ExperimentScope` class handles both tool scoping and credit tracking. State lives in a `.autobio/` directory in the experiment folder:

```
experiment-dir/
├── .autobio/
│   ├── config.yaml    # tool scoping + budget config + cost overrides
│   └── budget.txt     # single number: remaining credits
└── ...
```

### config.yaml

Supports whitelist OR blacklist (mutually exclusive). Having both is a validation error.

```yaml
# Option A: blacklist — most tools available, exclude a few
excluded_tools:
  - openmm_md_simulate
  - rosetta_flexddg

# Option B: whitelist — only these tools available
# allowed_tools:
#   - esmfold
#   - proteinmpnn
#   - rosetta_score

budget:
  credits_per_iteration: 100
  costs:           # optional per-tool cost overrides
    boltz2: 20
    esmfold: 5
```

### budget.txt

A plain text file containing a single number — the remaining credit balance. Gets overwritten on each tool run.

```
75.0
```

### Scope Discovery

`ExperimentScope.discover()` walks up from CWD looking for `.autobio/config.yaml`. If none found, no constraints apply (fully backwards-compatible). Nearest scope wins (handles nested experiments).

## New Components

### `ExperimentScope` class (`src/autobio/core/scope.py`)

| Method | Signature | Purpose |
|---|---|---|
| `__init__` | `(root: Path, config: dict)` | Validate config, store state |
| `discover` | `(start: Path \| None = None) -> Self \| None` | Walk up from CWD to find `.autobio/config.yaml` |
| `init` | `(directory: Path, **kwargs) -> Self` | Create `.autobio/` with config.yaml and budget.txt |
| `is_tool_allowed` | `(name: str) -> bool` | Check whitelist/blacklist |
| `check_tool` | `(name: str) -> None` | Raise `ToolExcludedError` if not allowed |
| `get_tool_cost` | `(name: str) -> int` | Project override > `ToolEntry.credit_cost` |
| `remaining_credits` | `() -> float` | Read budget.txt from disk |
| `check_budget` | `(name: str) -> None` | Raise `InsufficientBudgetError` if cost > remaining |
| `deduct` | `(name: str) -> float` | Subtract cost, write budget.txt, return new balance |
| `reset_budget` | `() -> float` | Reset to `credits_per_iteration`, return amount |

Key behaviors:
- `discover()` resolves symlinks, stops at filesystem root
- Constructor raises `ScopeConfigError` if both `excluded_tools` and `allowed_tools` present
- `init()` validates tool names against `TOOL_REGISTRY` to catch typos
- `remaining_credits()` reads from disk every time (no caching — simple and correct)
- `deduct()` clamps to 0 (balance never goes negative)
- When no `budget` section in config, budget methods are no-ops (tool-filtering only)

### New exceptions (`src/autobio/core/result.py`)

- `ScopeConfigError(AutobioError)` — invalid config.yaml
- `ToolExcludedError(AutobioError)` — tool not allowed by active scope
- `InsufficientBudgetError(AutobioError)` — not enough credits

### New CLI commands (`src/autobio/cli/scope.py`)

- **`autobio init`**: `--exclude` (repeatable), `--allow` (repeatable), `--credits` (int), `--dir` (Path, default CWD), `--force` (overwrite existing). Creates `.autobio/config.yaml` and `budget.txt`.
- **`autobio budget`**: Discovers scope, shows remaining / total credits and per-tool costs.
- **`autobio reset-budget`**: Discovers scope, resets budget.txt to `credits_per_iteration`.

## Modifications to Existing Code

### `ToolEntry` — add `credit_cost` field

```python
# src/autobio/core/registry.py
@dataclass
class ToolEntry:
    ...
    input_format: tuple[str, ...] = ()
    credit_cost: int = 1              # <-- new field
```

Default of 1 means all existing registrations remain valid without changes. Tools that need higher costs get explicit values (see credit cost table below).

### `ToolRunner.run()` — scope check and deduction

Two insertion points in `run()` (`src/autobio/tools/base.py`):

**Before the try block** (before workspace creation):
```python
scope = ExperimentScope.discover()
if scope is not None:
    scope.check_tool(self.tool_name)
    scope.check_budget(self.tool_name)
```

Checks happen before any workspace/GPU/container work — fail fast, zero wasted resources.

**After `parse_output`, before return** (inside try, success path only):
```python
if scope is not None:
    scope.deduct(self.tool_name)
```

Failed runs never reach this point, so they don't consume credits.

### `autobio list` — scope filtering

After the `list_tools()` call in `src/autobio/cli/list.py`:
```python
scope = ExperimentScope.discover()
if scope is not None:
    tools = {n: e for n, e in tools.items() if scope.is_tool_allowed(n)}
```

The registry's `list_tools()` stays unchanged — scope filtering is a CLI concern.

## Default Credit Costs

Added as `credit_cost=N` to each tool's `ToolEntry` registration. Costs are roughly proportional to compute (GPU requirement + typical runtime). Project configs can override any of these.

| Cost | Tools | Rationale |
|---|---|---|
| 1 | evoef2_repair, evoef2_binding, evoef2_build_mutant, rosetta_score, openmm_amber_minimize | CPU-only, fast (≤600s timeout) |
| 2 | esm1b, esm2, proteinmpnn, ligandmpnn, currab, ft_esm, balm_paired, balm_unpaired, ablang2, antiberta2, stabddg, baddg, ligandmpnn_build_mutant | GPU, fast (600s timeout) |
| 3 | currab_pll, ft_esm_pll, balm_paired_pll, balm_unpaired_pll, ablang2_pll, antiberta2_pll, rosetta_minimize | GPU or CPU, medium (1800s timeout) |
| 5 | esmfold, boltz1, chai1, openfold3, rfd3, rosetta_relax, openmm_amber_relax | Long-running (3600s timeout) |
| 10 | boltz2 | Very long GPU (7200s timeout) |
| 15 | complexa, complexa_ligand, complexa_ame | Extreme GPU (43200s timeout) |
| 20 | openmm_md_simulate, rosetta_flexddg | Extreme runtime (14400–86400s) |

### Files requiring `credit_cost` additions

Only tools with cost > 1 need changes (cost-1 tools inherit the default):

- `src/autobio/tools/esm.py` — esm1b (2), esm2 (2)
- `src/autobio/tools/esmfold.py` — esmfold (5)
- `src/autobio/tools/antibody_lm.py` — 6 embedding tools (2), 6 PLL tools (3)
- `src/autobio/tools/mpnn.py` — proteinmpnn (2), ligandmpnn (2)
- `src/autobio/tools/boltz.py` — boltz1 (5), boltz2 (10)
- `src/autobio/tools/chai.py` — chai1 (5)
- `src/autobio/tools/openfold3.py` — openfold3 (5)
- `src/autobio/tools/complexa.py` — complexa (15), complexa_ligand (15), complexa_ame (15)
- `src/autobio/tools/rfd3.py` — rfd3 (5)
- `src/autobio/tools/rosetta.py` — rosetta_minimize (3), rosetta_relax (5), rosetta_flexddg (20)
- `src/autobio/tools/openmm.py` — openmm_amber_relax (5), openmm_md_simulate (20)
- `src/autobio/tools/stabddg.py` — stabddg (2)
- `src/autobio/tools/baddg.py` — baddg (2)
- `src/autobio/tools/ligandmpnn_packer.py` — ligandmpnn_build_mutant (2)

## Testing

### New test file: `tests/unit/test_scope.py`

- Config validation (both lists raises `ScopeConfigError`, missing budget section is fine)
- `discover()` walk-up (finds nearest `.autobio/`, returns None at root, nearest wins for nested)
- `init()` creates `.autobio/config.yaml` and `budget.txt` correctly
- `is_tool_allowed()` with whitelist, blacklist, and no filter
- Cost lookup with overrides and defaults
- Budget check/deduct/reset lifecycle
- Deduct clamps to 0

### Updates: `tests/unit/test_tool_runner.py`

- Excluded tool raises `ToolExcludedError` without touching Docker
- Insufficient budget raises `InsufficientBudgetError` without touching Docker
- Successful run deducts credits
- Failed run does not deduct

### Manual smoke test

```bash
cd /tmp/test-experiment
autobio init --credits 50 --exclude openmm_md_simulate
autobio list           # openmm_md_simulate absent
autobio budget         # shows 50/50
cat .autobio/config.yaml
cat .autobio/budget.txt
autobio reset-budget   # resets to 50
```
