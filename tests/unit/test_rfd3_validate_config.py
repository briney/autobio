"""Tests for containers/rfd3/validate_config.sh — container-side config validation.

These tests run the validation script directly against various config files
to verify it catches errors with useful messages before the slow RFD3 model loads.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

# Path to the validation script
_SCRIPT = os.path.join(
    os.path.dirname(__file__), "..", "..", "containers", "rfd3", "validate_config.sh"
)
_SCRIPT = os.path.normpath(_SCRIPT)


def _run_validate(config: dict, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Write config to a temp file and run validate_config.sh against it."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return subprocess.run(
        ["bash", _SCRIPT, str(config_path)],
        capture_output=True,
        text=True,
    )


def _has_jq() -> bool:
    """Check if jq is available (required by validate_config.sh)."""
    try:
        subprocess.run(["jq", "--version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return True


pytestmark = pytest.mark.skipif(not _has_jq(), reason="jq not installed")


# ---------------------------------------------------------------------------
# Valid configs — should pass
# ---------------------------------------------------------------------------


class TestValidConfigs:
    def test_minimal_unconditioned(self, tmp_path: Path) -> None:
        config = {
            "design_specs": {"test": {"length": "50"}},
            "n_batches": 1,
            "out_dir": "/workspace/outputs/raw",
        }
        result = _run_validate(config, tmp_path)
        assert result.returncode == 0, result.stderr

    def test_full_binder_config(self, tmp_path: Path) -> None:
        # Write a dummy PDB so file-existence check passes
        pdb = tmp_path / "target.pdb"
        pdb.write_text("ATOM\nEND\n")
        config = {
            "design_specs": {
                "binder": {
                    "input": str(pdb),
                    "contig": "40-80,/0,A1-50",
                    "length": "90-130",
                    "select_hotspots": {"A10": "CD2,CZ"},
                    "infer_ori_strategy": "hotspots",
                    "is_non_loopy": True,
                    "plddt_enhanced": True,
                }
            },
            "n_batches": 2,
            "out_dir": "/workspace/outputs/raw",
            "diffusion_batch_size": 8,
            "step_scale": 1.5,
            "gamma_0": 0.6,
        }
        result = _run_validate(config, tmp_path)
        assert result.returncode == 0, result.stderr


# ---------------------------------------------------------------------------
# Known-key validation (typos, unknown keys)
# ---------------------------------------------------------------------------


class TestKnownKeyValidation:
    def test_typo_in_spec_key_suggests_correction(self, tmp_path: Path) -> None:
        config = {
            "design_specs": {"test": {"select_hotspot": {"A10": "CZ"}, "length": "50"}},
            "n_batches": 1,
            "out_dir": "/workspace/outputs/raw",
        }
        result = _run_validate(config, tmp_path)
        assert result.returncode != 0
        assert "select_hotspot" in result.stderr
        assert "select_hotspots" in result.stderr  # did-you-mean

    def test_typo_in_contig(self, tmp_path: Path) -> None:
        config = {
            "design_specs": {"test": {"contigs": "40-80", "length": "50"}},
            "n_batches": 1,
            "out_dir": "/workspace/outputs/raw",
        }
        result = _run_validate(config, tmp_path)
        assert result.returncode != 0
        assert "contigs" in result.stderr
        assert "contig" in result.stderr  # did-you-mean

    def test_unknown_top_level_key(self, tmp_path: Path) -> None:
        config = {
            "design_specs": {"test": {"length": "50"}},
            "n_batches": 1,
            "out_dir": "/workspace/outputs/raw",
            "unknown_key_xyz": 42,
        }
        result = _run_validate(config, tmp_path)
        # Unknown top-level keys generate warnings, not errors
        assert "unknown_key_xyz" in result.stderr


# ---------------------------------------------------------------------------
# Type validation
# ---------------------------------------------------------------------------


class TestTypeValidation:
    def test_boolean_field_given_string(self, tmp_path: Path) -> None:
        config = {
            "design_specs": {"test": {"length": "50", "is_non_loopy": "yes"}},
            "n_batches": 1,
            "out_dir": "/workspace/outputs/raw",
        }
        result = _run_validate(config, tmp_path)
        assert result.returncode != 0
        assert "is_non_loopy" in result.stderr
        assert "boolean" in result.stderr

    def test_n_batches_string_fails(self, tmp_path: Path) -> None:
        config = {
            "design_specs": {"test": {"length": "50"}},
            "n_batches": "three",
            "out_dir": "/workspace/outputs/raw",
        }
        result = _run_validate(config, tmp_path)
        assert result.returncode != 0
        assert "n_batches" in result.stderr


# ---------------------------------------------------------------------------
# Range validation
# ---------------------------------------------------------------------------


class TestRangeValidation:
    def test_negative_partial_t(self, tmp_path: Path) -> None:
        config = {
            "design_specs": {"test": {"length": "50", "partial_t": -5.0}},
            "n_batches": 1,
            "out_dir": "/workspace/outputs/raw",
        }
        result = _run_validate(config, tmp_path)
        assert result.returncode != 0
        assert "partial_t" in result.stderr

    def test_zero_diffusion_batch_size(self, tmp_path: Path) -> None:
        config = {
            "design_specs": {"test": {"length": "50"}},
            "n_batches": 1,
            "out_dir": "/workspace/outputs/raw",
            "diffusion_batch_size": 0,
        }
        result = _run_validate(config, tmp_path)
        assert result.returncode != 0
        assert "diffusion_batch_size" in result.stderr


# ---------------------------------------------------------------------------
# Enum validation
# ---------------------------------------------------------------------------


class TestEnumValidation:
    def test_invalid_infer_ori_strategy(self, tmp_path: Path) -> None:
        config = {
            "design_specs": {"test": {"length": "50", "infer_ori_strategy": "invalid"}},
            "n_batches": 1,
            "out_dir": "/workspace/outputs/raw",
        }
        result = _run_validate(config, tmp_path)
        assert result.returncode != 0
        assert "infer_ori_strategy" in result.stderr
        assert "'com' or 'hotspots'" in result.stderr

    def test_invalid_dialect(self, tmp_path: Path) -> None:
        config = {
            "design_specs": {"test": {"length": "50", "dialect": 3}},
            "n_batches": 1,
            "out_dir": "/workspace/outputs/raw",
        }
        result = _run_validate(config, tmp_path)
        assert result.returncode != 0
        assert "dialect" in result.stderr

    def test_invalid_kind(self, tmp_path: Path) -> None:
        config = {
            "design_specs": {"test": {"length": "50"}},
            "n_batches": 1,
            "out_dir": "/workspace/outputs/raw",
            "kind": "invalid_sampler",
        }
        result = _run_validate(config, tmp_path)
        assert result.returncode != 0
        assert "kind" in result.stderr


# ---------------------------------------------------------------------------
# Structural coherence
# ---------------------------------------------------------------------------


class TestStructuralCoherence:
    def test_contig_with_chain_but_no_input(self, tmp_path: Path) -> None:
        """Contig references chain IDs but no input structure is provided."""
        config = {
            "design_specs": {"test": {"contig": "A1-10,50-80"}},
            "n_batches": 1,
            "out_dir": "/workspace/outputs/raw",
        }
        result = _run_validate(config, tmp_path)
        assert result.returncode != 0
        assert "input" in result.stderr.lower()

    def test_symmetry_without_id(self, tmp_path: Path) -> None:
        config = {
            "design_specs": {
                "test": {
                    "length": "100",
                    "symmetry": {"is_symmetric_motif": True},
                }
            },
            "n_batches": 1,
            "out_dir": "/workspace/outputs/raw",
        }
        result = _run_validate(config, tmp_path)
        assert result.returncode != 0
        assert "symmetry.id" in result.stderr

    def test_invalid_symmetry_group(self, tmp_path: Path) -> None:
        config = {
            "design_specs": {
                "test": {
                    "length": "100",
                    "symmetry": {"id": "T3"},
                }
            },
            "n_batches": 1,
            "out_dir": "/workspace/outputs/raw",
        }
        result = _run_validate(config, tmp_path)
        assert result.returncode != 0
        assert "C or D group" in result.stderr


# ---------------------------------------------------------------------------
# Error message quality
# ---------------------------------------------------------------------------


class TestErrorMessageQuality:
    def test_multiple_errors_all_reported(self, tmp_path: Path) -> None:
        """Multiple errors in one config should all be reported."""
        config = {
            "design_specs": {
                "test": {
                    "length": "50",
                    "is_non_loopy": "yes",
                    "infer_ori_strategy": "invalid",
                }
            },
            "n_batches": -1,
            "out_dir": "/workspace/outputs/raw",
        }
        result = _run_validate(config, tmp_path)
        assert result.returncode != 0
        # Should report all errors, not just the first
        assert "is_non_loopy" in result.stderr
        assert "infer_ori_strategy" in result.stderr
        assert "n_batches" in result.stderr
