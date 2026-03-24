#!/usr/bin/env python3
"""Run Proteina-Complexa generation from autobio config.json.

Translates the autobio config.json into Hydra CLI overrides and invokes
``complexa generate`` for each design specification.  Each spec runs as a
separate generation pass with its own target configuration.

Output PDBs and reward CSVs land in the workspace raw output directory.
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


def _build_overrides(
    config: dict,
    spec_name: str,
    spec: dict,
) -> list[str]:
    """Build Hydra ``++override`` arguments for one design spec."""
    weights_dir = config["weights_dir"]
    ckpt_name = config["ckpt_name"]
    ae_ckpt_name = config["ae_ckpt_name"]

    overrides: list[str] = [
        # Run identity
        f"++run_name={spec_name}",
        f"++generation.task_name={spec_name}",
        # Checkpoints
        f"++ckpt_path={weights_dir}",
        f"++ckpt_name={ckpt_name}",
        f"++autoencoder_ckpt_path={weights_dir}/{ae_ckpt_name}",
        # Single-GPU, single-job
        "++gen_njobs=1",
    ]

    # -- Target definition via target_dict_cfg override --------------------
    target_path = spec.get("input", "")
    target_input = spec.get("target_input", "")
    hotspot_residues = spec.get("hotspot_residues", [])
    binder_length = spec.get("binder_length", [50, 150])

    prefix = f"++target_dict_cfg.{spec_name}"
    overrides.extend([
        f"{prefix}.source=custom",
        f"{prefix}.target_filename={spec_name}",
        f"{prefix}.target_path={target_path}",
        f"{prefix}.target_input={target_input}",
        # pdb_id: use provided value or empty string (not null/omitted, as
        # the default config interpolates this field and null breaks OmegaConf)
        f"{prefix}.pdb_id={spec.get('pdb_id', '')}",
    ])

    # Lists need Hydra bracket notation: [A37,A39,A49]
    if hotspot_residues:
        hs_str = "[" + ",".join(str(h) for h in hotspot_residues) + "]"
        overrides.append(f"{prefix}.hotspot_residues={hs_str}")

    if binder_length:
        bl_str = "[" + ",".join(str(x) for x in binder_length) + "]"
        overrides.append(f"{prefix}.binder_length={bl_str}")

    # -- Also override conditional_features to match (belt and suspenders) -
    # Direct overrides break Hydra interpolation chains that reference
    # target_dict_cfg, avoiding InterpolationKeyError for custom targets.
    cf_prefix = "++generation.dataloader.dataset.conditional_features.0"
    overrides.extend([
        f"{cf_prefix}.pdb_path={target_path}",
        f"{cf_prefix}.input_spec={target_input}",
        f"{cf_prefix}.pdb_id={spec.get('pdb_id', 'null')}",
        f"{cf_prefix}.binder_center=null",
    ])
    if hotspot_residues:
        overrides.append(f"{cf_prefix}.target_hotspots={hs_str}")

    # -- Binder length range → dataset nres --------------------------------
    if binder_length and len(binder_length) == 2:
        overrides.extend([
            f"++generation.dataloader.dataset.nres.low={binder_length[0]}",
            f"++generation.dataloader.dataset.nres.high={binder_length[1]}",
        ])

    # -- Optional spec-level fields ----------------------------------------
    if "binder_center" in spec:
        overrides.append(f"{cf_prefix}.binder_center={spec['binder_center']}")
    if "ligand_chain" in spec:
        overrides.append(f"{cf_prefix}.ligand_chain={spec['ligand_chain']}")

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

    # -- Disable reward model (not available in generate-only container) ----
    # The default pipeline config includes AF2 reward model which requires
    # AlphaFold2 parameters.  For single-pass generation, no reward model
    # is needed.
    overrides.append("++generation.reward_model=null")

    # -- Hydra output directory → workspace logs ---------------------------
    overrides.append("++hydra.run.dir=/workspace/logs/hydra_outputs")

    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Proteina-Complexa generation.")
    parser.add_argument("--workspace", type=Path, required=True)
    args = parser.parse_args()

    workspace = args.workspace
    config_path = workspace / "config.json"
    config = json.loads(config_path.read_text())

    variant = config["variant"]
    pipeline_config = config.get("pipeline_config", _PIPELINE_CONFIGS[variant])
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # The proteina-complexa repo is at /app/proteina-complexa/ in the container.
    # Pipeline configs live at /app/proteina-complexa/configs/.
    # We run from the raw output dir so ./inference/ outputs land there, and
    # use an absolute path for the config file.
    project_root = Path("/app/proteina-complexa")
    config_file = str(project_root / "configs" / f"{pipeline_config}.yaml")
    run_cwd = str(out_dir)

    design_specs: dict[str, dict] = config["design_specs"]

    for spec_name, spec in design_specs.items():
        print(f"[complexa] Generating binders for spec: {spec_name}")

        overrides = _build_overrides(config, spec_name, spec)

        cmd = ["complexa", "generate", config_file] + overrides
        print(f"[complexa] Command: {' '.join(cmd[:5])} ... ({len(overrides)} overrides)")

        result = subprocess.run(
            cmd,
            cwd=run_cwd,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        if result.returncode != 0:
            print(
                f"[complexa] ERROR: Generation failed for spec '{spec_name}' "
                f"with exit code {result.returncode}",
                file=sys.stderr,
            )
            sys.exit(result.returncode)

        print(f"[complexa] Completed generation for spec: {spec_name}")

    print("[complexa] All specs completed successfully.")


if __name__ == "__main__":
    main()
