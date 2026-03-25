#!/usr/bin/env python3
"""Run Proteina-Complexa from autobio config.json.

Supports two modes:

* **generate** (default) — Calls ``complexa generate`` for each design
  specification. Produces candidate structures without evaluation. The reward
  model is explicitly disabled.

* **design** — Calls ``complexa design`` for each design specification. Runs
  the full pipeline: generate -> filter -> evaluate -> analyze. Requires
  community model weights (AF2, RF3, ESM2, MPNN) in the container.

Each Complexa variant (protein_binder, ligand_binder, ame) uses a different
Hydra config structure for target specification. The override builder handles
these differences:

* **protein_binder** — ``target_dict_cfg`` with ``input_spec`` / ``target_hotspots``
* **ligand_binder** — ``target_dict_cfg`` + ``ligand`` / ``ligand_only`` / ``SMILES``
* **ame** — ``motif_target_dict_cfg`` with ``contig_atoms`` (MotifFeatures) + ``ligand``
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


# Map variant → built-in pipeline config name (without .yaml extension).
# These configs ship with the proteina-complexa package.
_PIPELINE_CONFIGS: dict[str, str] = {
    "protein_binder": "search_binder_local_pipeline",
    "ligand_binder": "search_ligand_binder_local_pipeline",
    "ame": "search_ame_local_pipeline",
}


def _hydra_list(items: list) -> str:
    """Format a Python list as Hydra bracket notation: [a,b,c]."""
    return "[" + ",".join(str(x) for x in items) + "]"


def _build_common_overrides(
    config: dict,
    spec_name: str,
) -> list[str]:
    """Build overrides shared across all variants: identity, checkpoints, jobs."""
    return [
        f"++run_name={spec_name}",
        f"++generation.task_name={spec_name}",
        f"++ckpt_path={config['weights_dir']}",
        f"++ckpt_name={config['ckpt_name']}",
        f"++autoencoder_ckpt_path={config['weights_dir']}/{config['ae_ckpt_name']}",
        "++gen_njobs=1",
    ]


def _build_binder_overrides(
    spec_name: str,
    spec: dict,
) -> list[str]:
    """Build overrides for the protein_binder variant.

    Uses ``target_dict_cfg`` and ``BinderFeatures`` conditional features
    with ``input_spec`` and ``target_hotspots``.
    """
    overrides: list[str] = []
    target_path = spec.get("input", "")
    target_input = spec.get("target_input", "")
    hotspot_residues = spec.get("hotspot_residues", [])
    binder_length = spec.get("binder_length", [50, 150])

    # -- target_dict_cfg overrides --
    prefix = f"++target_dict_cfg.{spec_name}"
    overrides.extend([
        f"{prefix}.source=custom",
        f"{prefix}.target_filename={spec_name}",
        f"{prefix}.target_path={target_path}",
        f"{prefix}.target_input={target_input}",
        f"{prefix}.pdb_id={spec.get('pdb_id', '')}",
    ])
    if hotspot_residues:
        overrides.append(f"{prefix}.hotspot_residues={_hydra_list(hotspot_residues)}")
    if binder_length:
        overrides.append(f"{prefix}.binder_length={_hydra_list(binder_length)}")

    # -- conditional_features.0 (BinderFeatures) --
    cf = "++generation.dataloader.dataset.conditional_features.0"
    overrides.extend([
        f"{cf}.pdb_path={target_path}",
        f"{cf}.input_spec={target_input}",
        f"{cf}.pdb_id={spec.get('pdb_id', 'null')}",
        f"{cf}.binder_center=null",
    ])
    if hotspot_residues:
        overrides.append(f"{cf}.target_hotspots={_hydra_list(hotspot_residues)}")

    # -- nres range --
    if binder_length and len(binder_length) == 2:
        overrides.extend([
            f"++generation.dataloader.dataset.nres.low={binder_length[0]}",
            f"++generation.dataloader.dataset.nres.high={binder_length[1]}",
        ])

    # -- optional fields --
    if "binder_center" in spec:
        overrides.append(f"{cf}.binder_center={spec['binder_center']}")

    return overrides


def _build_ligand_binder_overrides(
    spec_name: str,
    spec: dict,
) -> list[str]:
    """Build overrides for the ligand_binder variant.

    Uses ``target_dict_cfg`` with ligand fields.  The pipeline config has a
    single ``conditional_features`` entry — ``LigandFeatures`` at index 0
    (no BinderFeatures).  It interpolates ``target_dict_cfg.*.ligand``,
    ``ligand_only``, ``SMILES``, and ``use_bonds_from_file``.
    """
    overrides: list[str] = []
    target_path = spec.get("input", "")
    target_input = spec.get("target_input", "")
    hotspot_residues = spec.get("hotspot_residues", [])
    binder_length = spec.get("binder_length", [100])

    ligand = spec.get("ligand", spec.get("ligand_chain", ""))
    smiles = spec.get("smiles", "")

    # -- target_dict_cfg overrides --
    prefix = f"++target_dict_cfg.{spec_name}"
    overrides.extend([
        f"{prefix}.source=custom",
        f"{prefix}.target_filename={spec_name}",
        f"{prefix}.target_path={target_path}",
        f"{prefix}.target_input={target_input}",
        f"{prefix}.pdb_id={spec.get('pdb_id', '')}",
    ])
    if hotspot_residues:
        overrides.append(f"{prefix}.hotspot_residues={_hydra_list(hotspot_residues)}")
    else:
        overrides.append(f"{prefix}.hotspot_residues=[null]")
    if binder_length:
        overrides.append(f"{prefix}.binder_length={_hydra_list(binder_length)}")

    # Ligand-specific fields on target_dict_cfg (required by interpolation)
    overrides.append(f"{prefix}.ligand={ligand}")
    overrides.append(f"{prefix}.ligand_only={str(spec.get('ligand_only', True))}")
    overrides.append(
        f"{prefix}.use_bonds_from_file={str(spec.get('use_bonds_from_file', True))}"
    )
    # SMILES is required by the pipeline config interpolation; pass empty
    # string if not provided so the interpolation resolves.
    overrides.append(f"{prefix}.SMILES='{smiles}'")

    # -- conditional_features.0 (LigandFeatures — only feature for this variant) --
    cf0 = "++generation.dataloader.dataset.conditional_features.0"
    overrides.extend([
        f"{cf0}.pdb_path={target_path}",
        f"{cf0}.ligand={ligand}",
        f"{cf0}.ligand_only={str(spec.get('ligand_only', True))}",
        f"{cf0}.SMILES='{smiles}'",
        f"{cf0}.use_bonds_from_file={str(spec.get('use_bonds_from_file', True))}",
    ])

    # -- nres range --
    if binder_length and len(binder_length) >= 1:
        overrides.append(
            f"++generation.dataloader.dataset.nres.low={binder_length[0]}"
        )
        if len(binder_length) == 2:
            overrides.append(
                f"++generation.dataloader.dataset.nres.high={binder_length[1]}"
            )

    return overrides


def _build_ame_overrides(
    spec_name: str,
    spec: dict,
) -> list[str]:
    """Build overrides for the AME (motif scaffolding) variant.

    Uses ``motif_target_dict_cfg`` instead of ``target_dict_cfg``.
    Conditional features are ``MotifFeatures`` (with ``motif_atom_spec``
    from ``contig_atoms``) and ``LigandFeatures`` (with ``ligand``).
    """
    overrides: list[str] = []
    target_path = spec.get("input", "")
    binder_length = spec.get("binder_length", [180])
    contig_atoms = spec.get("contig_atoms", "")
    ligand = spec.get("ligand", "")

    # -- motif_target_dict_cfg overrides --
    prefix = f"++motif_target_dict_cfg.{spec_name}"
    overrides.extend([
        f"{prefix}.source=custom",
        f"{prefix}.target_filename={spec_name}",
        f"{prefix}.target_path={target_path}",
        f"{prefix}.hotspot_residues=[null]",
        f"{prefix}.use_bonds_from_file={str(spec.get('use_bonds_from_file', True))}",
    ])
    if binder_length:
        overrides.append(f"{prefix}.binder_length={_hydra_list(binder_length)}")
    if contig_atoms:
        # contig_atoms contains Hydra-special chars (colons, brackets, commas).
        # Quote the VALUE so Hydra treats it as a string literal.
        overrides.append(f"{prefix}.contig_atoms='{contig_atoms}'")
    # ligand is always required (interpolated by the AME pipeline config).
    # Empty string if no ligand context is needed.
    overrides.append(f"{prefix}.ligand={ligand if ligand else 'null'}")

    # -- conditional_features.0 (MotifFeatures) --
    cf0 = "++generation.dataloader.dataset.conditional_features.0"
    overrides.extend([
        f"{cf0}.pdb_path={target_path}",
    ])
    if contig_atoms:
        overrides.append(f"{cf0}.motif_atom_spec='{contig_atoms}'")

    # -- conditional_features.1 (LigandFeatures) --
    cf1 = "++generation.dataloader.dataset.conditional_features.1"
    overrides.extend([
        f"{cf1}.pdb_path={target_path}",
    ])
    overrides.append(f"{cf1}.ligand={ligand if ligand else 'null'}")
    # When ligand is null/empty, ligand_only must be True (the entire PDB
    # file is used as ligand context for the motif).
    ligand_only = str(spec.get("ligand_only", not bool(ligand)))
    overrides.append(f"{cf1}.ligand_only={ligand_only}")

    # -- nres range --
    if binder_length and len(binder_length) >= 1:
        overrides.append(
            f"++generation.dataloader.dataset.nres.low={binder_length[0]}"
        )
        if len(binder_length) == 2:
            overrides.append(
                f"++generation.dataloader.dataset.nres.high={binder_length[1]}"
            )

    return overrides


def _build_overrides(
    config: dict,
    spec_name: str,
    spec: dict,
    *,
    mode: str = "generate",
) -> list[str]:
    """Build Hydra ``++override`` arguments for one design spec.

    Dispatches to variant-specific builders for the target definition and
    conditional features, then appends common generation-level and
    mode-specific overrides.

    Args:
        config: Full config.json contents.
        spec_name: Name of the design specification.
        spec: Per-spec parameter dict.
        mode: ``"generate"`` or ``"design"``.
    """
    variant = config["variant"]

    # Common overrides: identity, checkpoints, jobs
    overrides = _build_common_overrides(config, spec_name)

    # Variant-specific target/feature overrides
    if variant == "ame":
        overrides.extend(_build_ame_overrides(spec_name, spec))
    elif variant == "ligand_binder":
        overrides.extend(_build_ligand_binder_overrides(spec_name, spec))
    else:
        overrides.extend(_build_binder_overrides(spec_name, spec))

    # -- Generation-level parameters from top-level config -----------------
    if "batch_size" in config:
        overrides.append(
            f"++generation.dataloader.batch_size={config['batch_size']}"
        )
    if "n_samples_per_length" in config:
        overrides.append(
            f"++generation.dataloader.dataset.nres.nsamples="
            f"{config['n_samples_per_length']}"
        )
    if "binder_length_samples" in config:
        overrides.append(
            f"++generation.dataloader.dataset.nres.nsamples="
            f"{config['binder_length_samples']}"
        )
    if "seed" in config:
        overrides.append(f"++seed={config['seed']}")
    if "search_algorithm" in config:
        overrides.append(
            f"++generation.search.algorithm={config['search_algorithm']}"
        )

    # -- Mode-specific overrides -------------------------------------------
    if mode == "generate":
        overrides.append("++generation.reward_model=null")
    elif mode == "design":
        if "eval_njobs" in config:
            overrides.append(f"++eval_njobs={config['eval_njobs']}")
        else:
            overrides.append("++eval_njobs=1")
        if "gen_njobs" in config:
            overrides.append(f"++gen_njobs={config['gen_njobs']}")

    # -- Hydra output directory → workspace logs ---------------------------
    overrides.append("++hydra.run.dir=/workspace/logs/hydra_outputs")

    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Proteina-Complexa.")
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()

    workspace = args.workspace
    config_path = workspace / "config.json"
    config = json.loads(config_path.read_text())

    variant = config["variant"]
    mode = config.get("mode", "generate")
    pipeline_config = config.get("pipeline_config", _PIPELINE_CONFIGS[variant])
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    project_root = Path("/app/proteina-complexa")
    config_file = str(project_root / "configs" / f"{pipeline_config}.yaml")
    run_cwd = str(out_dir)

    subcommand = "design" if mode == "design" else "generate"

    design_specs: dict[str, dict] = config["design_specs"]

    for spec_name, spec in design_specs.items():
        print(f"[complexa] Running {subcommand} for spec: {spec_name}")

        overrides = _build_overrides(config, spec_name, spec, mode=mode)

        cmd = ["complexa", subcommand, config_file] + overrides
        print(f"[complexa] Command: {' '.join(cmd[:5])} ... ({len(overrides)} overrides)")

        result = subprocess.run(
            cmd,
            cwd=run_cwd,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        if result.returncode != 0:
            print(
                f"[complexa] ERROR: {subcommand.capitalize()} failed for spec "
                f"'{spec_name}' with exit code {result.returncode}",
                file=sys.stderr,
            )
            sys.exit(result.returncode)

        print(f"[complexa] Completed {subcommand} for spec: {spec_name}")

    print(f"[complexa] All specs completed successfully (mode={mode}).")


if __name__ == "__main__":
    main()
