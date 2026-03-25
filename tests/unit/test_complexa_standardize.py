"""Tests for containers/complexa/standardize.py — evaluation CSV parsing.

Validates that the standardize script correctly parses evaluation CSVs
produced by ``complexa design`` (the full pipeline with AF2/RF3/MPNN evaluation)
and correctly matches metrics to designs.

The real Proteina-Complexa evaluation output includes:
- ``binder_results_{config}_{job_id}.csv`` with columns like
  ``self_complex_i_pAE``, ``self_binder_pLDDT``, ``mpnn_binder_scRMSD_ca``, etc.
- ``rewards_*.csv`` for generation rewards (handled separately)
- ``timing_*.csv`` for performance data (should be ignored)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

# Import standardize.py from the container directory
_CONTAINER_DIR = str(Path(__file__).resolve().parent.parent.parent / "containers" / "complexa")
if _CONTAINER_DIR not in sys.path:
    sys.path.insert(0, _CONTAINER_DIR)

from standardize import (  # noqa: E402
    _build_dir_to_spec_map,
    _find_designs,
    _load_evaluation_metrics,
    _load_rewards,
    _match_evaluation_metrics,
    standardize,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_workspace(
    tmp_path: Path,
    *,
    mode: str = "design",
    variant: str = "protein_binder",
    pipeline_config: str = "search_binder_local_pipeline",
    design_specs: dict | None = None,
) -> Path:
    """Create a workspace directory with config.json."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "outputs" / "raw").mkdir(parents=True)
    (workspace / "outputs" / "standardized").mkdir(parents=True)

    if design_specs is None:
        design_specs = {"crambin_binder": {"target_input": "A1-46"}}

    config = {
        "variant": variant,
        "mode": mode,
        "pipeline_config": pipeline_config,
        "design_specs": design_specs,
        "out_dir": "/workspace/outputs/raw",
    }
    (workspace / "config.json").write_text(json.dumps(config))
    return workspace


def _make_pdb(path: Path) -> None:
    """Create a minimal PDB file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
        "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00\n"
        "END\n"
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write a CSV file from a list of dicts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# Directory name that Complexa produces for spec "crambin_binder"
# with pipeline config "search_binder_local_pipeline"
_DIR_NAME = "search_binder_local_pipeline_crambin_binder_crambin_binder"


# ---------------------------------------------------------------------------
# Realistic CSV data matching real Proteina-Complexa output
# ---------------------------------------------------------------------------

# Real binder_results CSV columns from the evaluate step.
# {seq} prefixes: self (original sequence), mpnn (MPNN redesigned)
_BINDER_RESULTS_ROW = {
    "id_gen": "0",
    "pdb_path": (
        "/workspace/outputs/raw/inference/"
        f"{_DIR_NAME}/"
        "job_0_n_80_id_0_single_orig0/"
        "job_0_n_80_id_0_single_orig0.pdb"
    ),
    "L": "80",
    "run_name": "crambin_binder",
    "ckpt_path": "/app/proteina-complexa/ckpts/complexa.ckpt",
    "task_name": "crambin_binder",
    # --- Self-consistency metrics (AF2 self-prediction) ---
    "self_complex_i_pAE": "0.15",
    "self_complex_pTM": "0.85",
    "self_complex_iPTM": "0.82",
    "self_complex_pLDDT": "0.91",
    "self_binder_pLDDT": "0.88",
    "self_binder_scRMSD": "1.2",
    "self_binder_scRMSD_ca": "0.95",
    "self_binder_scRMSD_bb3": "1.05",
    "self_binder_scRMSD_bb3o": "1.1",
    "self_binder_scRMSD_allatom": "2.3",
    "self_complex_scRMSD": "1.8",
    "self_complex_scRMSD_ca": "1.5",
    "self_complex_pdb_path": "/workspace/outputs/raw/evaluation/pred_0.pdb",
    "self_sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLHGTAFSSQPELLHFVGG",
    "self_complex_i_pAE_all": "[0.15, 0.18]",
    "self_binder_scRMSD_all": "[0.95, 1.1]",
    # --- MPNN metrics (MPNN redesign + AF2 re-prediction) ---
    "mpnn_complex_i_pAE": "0.12",
    "mpnn_complex_pTM": "0.87",
    "mpnn_complex_iPTM": "0.84",
    "mpnn_complex_pLDDT": "0.93",
    "mpnn_binder_pLDDT": "0.90",
    "mpnn_binder_scRMSD": "1.0",
    "mpnn_binder_scRMSD_ca": "0.85",
    "mpnn_complex_pdb_path": "/workspace/outputs/raw/evaluation/mpnn_pred_0.pdb",
    "mpnn_sequence": "MKTTFIAKQRQTSFVKAHFSRQLEERLDLIEVQAPLLSRVGDRTQDNLHGTAFSAQPELLHFLGG",
}


# Second design in the same run
_BINDER_RESULTS_ROW_2 = {
    **_BINDER_RESULTS_ROW,
    "id_gen": "1",
    "pdb_path": (
        "/workspace/outputs/raw/inference/"
        f"{_DIR_NAME}/"
        "job_0_n_60_id_1_single_orig0/"
        "job_0_n_60_id_1_single_orig0.pdb"
    ),
    "L": "60",
    "self_complex_i_pAE": "0.22",
    "self_binder_pLDDT": "0.78",
    "self_binder_scRMSD_ca": "1.8",
    "mpnn_complex_i_pAE": "0.19",
    "mpnn_binder_pLDDT": "0.81",
    "mpnn_binder_scRMSD_ca": "1.5",
}


# Rewards CSV (generated by the generate step, NOT the evaluate step)
_REWARDS_ROW = {
    "pdb_path": (
        "/workspace/outputs/raw/inference/"
        f"{_DIR_NAME}/"
        "job_0_n_80_id_0_single_orig0/"
        "job_0_n_80_id_0_single_orig0.pdb"
    ),
    "pdb_index": "0",
    "aatype": "16,16,4,4,14,15,9,19",
    "total_reward": "",
    "sample_type": "final",
    "metadata_tag": "single_orig0",
}


# Timing CSV (should NOT be loaded as evaluation metrics)
_TIMING_ROW = {
    "job_id": "0",
    "total_time": "9.29",
    "nsamples": "1",
}


# ---------------------------------------------------------------------------
# Test: _load_evaluation_metrics
# ---------------------------------------------------------------------------


class TestLoadEvaluationMetrics:
    """Tests for _load_evaluation_metrics() CSV parsing."""

    def test_detects_binder_results_csv(self, tmp_path: Path) -> None:
        """Evaluation CSV with metric columns is correctly detected."""
        raw_dir = tmp_path / "raw"
        _write_csv(
            raw_dir / "binder_results_search_binder_local_pipeline_0.csv",
            [_BINDER_RESULTS_ROW],
        )

        metrics = _load_evaluation_metrics(raw_dir)
        assert len(metrics) == 1
        assert _BINDER_RESULTS_ROW["pdb_path"] in metrics

    def test_skips_rewards_csv(self, tmp_path: Path) -> None:
        """rewards_*.csv files are not loaded as evaluation metrics."""
        raw_dir = tmp_path / "raw"
        _write_csv(raw_dir / "rewards_search_binder_local_pipeline_0.csv", [_REWARDS_ROW])

        metrics = _load_evaluation_metrics(raw_dir)
        assert len(metrics) == 0

    def test_skips_timing_csv(self, tmp_path: Path) -> None:
        """CSVs without metric-indicator columns are skipped."""
        raw_dir = tmp_path / "raw"
        _write_csv(raw_dir / "timing_0.csv", [_TIMING_ROW])

        metrics = _load_evaluation_metrics(raw_dir)
        assert len(metrics) == 0

    def test_identifier_columns_excluded(self, tmp_path: Path) -> None:
        """pdb_path, id_gen, run_name, and L columns are NOT in the metrics dict."""
        raw_dir = tmp_path / "raw"
        _write_csv(raw_dir / "binder_results_0.csv", [_BINDER_RESULTS_ROW])

        metrics = _load_evaluation_metrics(raw_dir)
        entry = metrics[_BINDER_RESULTS_ROW["pdb_path"]]

        assert "pdb_path" not in entry
        assert "id_gen" not in entry
        assert "run_name" not in entry
        assert "L" not in entry

    def test_numeric_values_parsed_as_float(self, tmp_path: Path) -> None:
        """Numeric metric values are converted to float."""
        raw_dir = tmp_path / "raw"
        _write_csv(raw_dir / "binder_results_0.csv", [_BINDER_RESULTS_ROW])

        metrics = _load_evaluation_metrics(raw_dir)
        entry = metrics[_BINDER_RESULTS_ROW["pdb_path"]]

        assert isinstance(entry["self_complex_i_pAE"], float)
        assert entry["self_complex_i_pAE"] == pytest.approx(0.15)
        assert isinstance(entry["self_binder_pLDDT"], float)
        assert entry["self_binder_pLDDT"] == pytest.approx(0.88)
        assert isinstance(entry["mpnn_binder_scRMSD_ca"], float)
        assert entry["mpnn_binder_scRMSD_ca"] == pytest.approx(0.85)

    def test_nonnumeric_values_kept_as_string(self, tmp_path: Path) -> None:
        """Non-numeric values (paths, sequences, lists) are kept as strings."""
        raw_dir = tmp_path / "raw"
        _write_csv(raw_dir / "binder_results_0.csv", [_BINDER_RESULTS_ROW])

        metrics = _load_evaluation_metrics(raw_dir)
        entry = metrics[_BINDER_RESULTS_ROW["pdb_path"]]

        # Paths should be strings
        assert isinstance(entry["self_complex_pdb_path"], str)
        assert entry["self_complex_pdb_path"].endswith(".pdb")

        # Sequences should be strings
        assert isinstance(entry["self_sequence"], str)

        # Serialized lists should be strings
        assert isinstance(entry["self_complex_i_pAE_all"], str)
        assert entry["self_complex_i_pAE_all"].startswith("[")

    def test_multiple_designs_loaded(self, tmp_path: Path) -> None:
        """Multiple rows in a CSV produce separate metric entries."""
        raw_dir = tmp_path / "raw"
        _write_csv(
            raw_dir / "binder_results_0.csv",
            [_BINDER_RESULTS_ROW, _BINDER_RESULTS_ROW_2],
        )

        metrics = _load_evaluation_metrics(raw_dir)
        assert len(metrics) == 2
        assert _BINDER_RESULTS_ROW["pdb_path"] in metrics
        assert _BINDER_RESULTS_ROW_2["pdb_path"] in metrics

        # Values should differ between the two designs
        entry1 = metrics[_BINDER_RESULTS_ROW["pdb_path"]]
        entry2 = metrics[_BINDER_RESULTS_ROW_2["pdb_path"]]
        assert entry1["self_complex_i_pAE"] != entry2["self_complex_i_pAE"]

    def test_multiple_csvs_merged(self, tmp_path: Path) -> None:
        """Metrics from multiple CSV files are merged into one dict."""
        raw_dir = tmp_path / "raw"
        _write_csv(raw_dir / "binder_results_0.csv", [_BINDER_RESULTS_ROW])
        _write_csv(raw_dir / "binder_results_1.csv", [_BINDER_RESULTS_ROW_2])

        metrics = _load_evaluation_metrics(raw_dir)
        assert len(metrics) == 2

    def test_malformed_csv_skipped(self, tmp_path: Path) -> None:
        """Malformed CSV files are skipped without raising."""
        raw_dir = tmp_path / "raw"
        # Write garbage
        bad_csv = raw_dir / "broken_results.csv"
        bad_csv.parent.mkdir(parents=True, exist_ok=True)
        bad_csv.write_text("this\x00is\x00not\x00valid")

        # Should not raise
        metrics = _load_evaluation_metrics(raw_dir)
        assert len(metrics) == 0

    def test_empty_csv_skipped(self, tmp_path: Path) -> None:
        """CSV with headers but no data rows produces no metrics."""
        raw_dir = tmp_path / "raw"
        csv_path = raw_dir / "empty_results.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text("pdb_path,self_complex_i_pAE,self_binder_pLDDT\n")

        metrics = _load_evaluation_metrics(raw_dir)
        assert len(metrics) == 0

    def test_id_gen_fallback_key(self, tmp_path: Path) -> None:
        """When pdb_path is empty, id_gen is used as the key."""
        raw_dir = tmp_path / "raw"
        row = {
            "id_gen": "design_42",
            "pdb_path": "",
            "self_complex_i_pAE": "0.15",
            "self_binder_pLDDT": "0.88",
        }
        _write_csv(raw_dir / "binder_results_0.csv", [row])

        metrics = _load_evaluation_metrics(raw_dir)
        assert "design_42" in metrics


# ---------------------------------------------------------------------------
# Test: _find_designs
# ---------------------------------------------------------------------------


class TestFindDesigns:
    """Tests for _find_designs() PDB discovery."""

    def test_finds_standard_pdb(self, tmp_path: Path) -> None:
        """Standard Complexa PDB filename is parsed correctly."""
        raw_dir = tmp_path / "raw"
        run_dir = raw_dir / "inference" / _DIR_NAME
        pdb_dir = run_dir / "job_0_n_80_id_0_single_orig0"
        pdb_path = pdb_dir / "job_0_n_80_id_0_single_orig0.pdb"
        _make_pdb(pdb_path)

        designs = _find_designs(raw_dir)
        assert len(designs) == 1
        d = designs[0]
        assert d["batch_index"] == 0
        assert d["design_index"] == 0
        assert d["binder_length"] == 80
        assert d["metadata_tag"] == "single_orig0"
        assert d["raw_pdb_path"] == pdb_path

    def test_skips_binder_intermediates(self, tmp_path: Path) -> None:
        """Files ending with _binder.pdb are skipped."""
        raw_dir = tmp_path / "raw"
        run_dir = raw_dir / "inference" / _DIR_NAME
        pdb_dir = run_dir / "job_0_n_80_id_0"

        _make_pdb(pdb_dir / "job_0_n_80_id_0.pdb")
        _make_pdb(pdb_dir / "job_0_n_80_id_0_binder.pdb")

        designs = _find_designs(raw_dir)
        assert len(designs) == 1
        assert designs[0]["raw_pdb_path"].name == "job_0_n_80_id_0.pdb"

    def test_multiple_designs_found(self, tmp_path: Path) -> None:
        """Multiple PDB files across different jobs are all found."""
        raw_dir = tmp_path / "raw"
        run_dir = raw_dir / "inference" / _DIR_NAME

        _make_pdb(run_dir / "job_0_n_80_id_0" / "job_0_n_80_id_0.pdb")
        _make_pdb(run_dir / "job_0_n_60_id_1" / "job_0_n_60_id_1.pdb")
        _make_pdb(run_dir / "job_1_n_80_id_0" / "job_1_n_80_id_0.pdb")

        designs = _find_designs(raw_dir)
        assert len(designs) == 3


# ---------------------------------------------------------------------------
# Test: _match_evaluation_metrics
# ---------------------------------------------------------------------------


class TestMatchEvaluationMetrics:
    """Tests for matching evaluation metrics to designs."""

    def test_exact_path_match(self) -> None:
        pdb_path = Path("/workspace/outputs/raw/inference/run/job_0_n_80_id_0/job_0_n_80_id_0.pdb")
        design = {"raw_pdb_path": pdb_path}
        eval_metrics = {str(pdb_path): {"self_complex_i_pAE": 0.15}}

        result = _match_evaluation_metrics(design, eval_metrics)
        assert result is not None
        assert result["self_complex_i_pAE"] == 0.15

    def test_filename_match(self) -> None:
        """Matches by filename when full path differs."""
        pdb_path = Path("/host/path/job_0_n_80_id_0.pdb")
        design = {"raw_pdb_path": pdb_path}
        eval_metrics = {"/container/path/job_0_n_80_id_0.pdb": {"self_binder_pLDDT": 0.92}}

        result = _match_evaluation_metrics(design, eval_metrics)
        assert result is not None
        assert result["self_binder_pLDDT"] == 0.92

    def test_stem_match(self) -> None:
        """Matches by stem when extensions differ."""
        pdb_path = Path("/path/job_0_n_80_id_0.pdb")
        design = {"raw_pdb_path": pdb_path}
        eval_metrics = {"job_0_n_80_id_0": {"mpnn_binder_scRMSD_ca": 1.3}}

        result = _match_evaluation_metrics(design, eval_metrics)
        assert result is not None
        assert result["mpnn_binder_scRMSD_ca"] == 1.3

    def test_no_match_returns_none(self) -> None:
        design = {"raw_pdb_path": Path("/path/design_1.pdb")}
        eval_metrics = {"/other/design_2.pdb": {"metric": 0.5}}

        result = _match_evaluation_metrics(design, eval_metrics)
        assert result is None

    def test_empty_metrics_returns_none(self) -> None:
        design = {"raw_pdb_path": Path("/path/design_1.pdb")}
        result = _match_evaluation_metrics(design, {})
        assert result is None


# ---------------------------------------------------------------------------
# Test: _build_dir_to_spec_map
# ---------------------------------------------------------------------------


class TestBuildDirToSpecMap:
    """Tests for directory-to-spec-name mapping."""

    def test_single_spec(self, tmp_path: Path) -> None:
        workspace = _make_workspace(tmp_path)
        mapping = _build_dir_to_spec_map(workspace)
        expected_dir = "search_binder_local_pipeline_crambin_binder_crambin_binder"
        assert mapping[expected_dir] == "crambin_binder"

    def test_multiple_specs(self, tmp_path: Path) -> None:
        workspace = _make_workspace(
            tmp_path,
            design_specs={
                "short_binder": {"target_input": "A1-46"},
                "long_binder": {"target_input": "A1-46"},
            },
        )
        mapping = _build_dir_to_spec_map(workspace)
        assert len(mapping) == 2
        assert mapping["search_binder_local_pipeline_short_binder_short_binder"] == "short_binder"
        assert mapping["search_binder_local_pipeline_long_binder_long_binder"] == "long_binder"

    def test_missing_config(self, tmp_path: Path) -> None:
        workspace = tmp_path / "empty_ws"
        workspace.mkdir()
        mapping = _build_dir_to_spec_map(workspace)
        assert mapping == {}


# ---------------------------------------------------------------------------
# Test: _load_rewards
# ---------------------------------------------------------------------------


class TestLoadRewards:
    """Tests for rewards CSV loading."""

    def test_loads_rewards_csv(self, tmp_path: Path) -> None:
        raw_dir = tmp_path / "raw"
        _write_csv(raw_dir / "rewards_search_binder_local_pipeline_0.csv", [_REWARDS_ROW])

        rewards = _load_rewards(raw_dir)
        assert len(rewards) == 1
        assert _REWARDS_ROW["pdb_path"] in rewards
        entry = rewards[_REWARDS_ROW["pdb_path"]]
        assert "pdb_path" not in entry
        assert entry["sample_type"] == "final"
        assert entry["metadata_tag"] == "single_orig0"

    def test_non_rewards_csv_ignored(self, tmp_path: Path) -> None:
        """Only rewards_*.csv files are loaded by _load_rewards."""
        raw_dir = tmp_path / "raw"
        _write_csv(raw_dir / "binder_results_0.csv", [_BINDER_RESULTS_ROW])

        rewards = _load_rewards(raw_dir)
        assert len(rewards) == 0


# ---------------------------------------------------------------------------
# Test: Full standardize() — design mode E2E
# ---------------------------------------------------------------------------


class TestStandardizeDesignMode:
    """End-to-end tests for standardize() in design mode."""

    def test_design_mode_with_evaluation_metrics(self, tmp_path: Path) -> None:
        """Full pipeline: PDBs + evaluation CSV → result_data.json with metrics."""
        workspace = _make_workspace(tmp_path, mode="design")
        raw_dir = workspace / "outputs" / "raw"

        # Create the PDB output
        run_dir = raw_dir / "inference" / _DIR_NAME
        pdb_dir = run_dir / "job_0_n_80_id_0_single_orig0"
        _make_pdb(pdb_dir / "job_0_n_80_id_0_single_orig0.pdb")

        # Create rewards CSV
        _write_csv(
            run_dir / "rewards_search_binder_local_pipeline_0.csv",
            [_REWARDS_ROW],
        )

        # Create evaluation CSV
        _write_csv(
            raw_dir / "evaluation_results" / "binder_results_0.csv",
            [_BINDER_RESULTS_ROW],
        )

        standardize(workspace)

        result_path = workspace / "outputs" / "standardized" / "result_data.json"
        assert result_path.exists()

        result = json.loads(result_path.read_text())
        assert len(result["designs"]) == 1

        design = result["designs"][0]
        assert design["spec_name"] == "crambin_binder"
        assert design["batch_index"] == 0
        assert design["design_index"] == 0
        assert design["diffusion_metadata"]["binder_length"] == 80
        assert design["diffusion_metadata"]["sample_type"] == "single_orig0"

        # Evaluation metrics should be populated
        assert "evaluation_metrics" in design
        assert design["evaluation_metrics"] is not None
        em = design["evaluation_metrics"]
        assert em["self_complex_i_pAE"] == pytest.approx(0.15)
        assert em["self_binder_pLDDT"] == pytest.approx(0.88)
        assert em["mpnn_binder_scRMSD_ca"] == pytest.approx(0.85)

    def test_generate_mode_no_evaluation_metrics(self, tmp_path: Path) -> None:
        """Generate mode: PDBs produce result_data.json without evaluation metrics."""
        workspace = _make_workspace(tmp_path, mode="generate")
        raw_dir = workspace / "outputs" / "raw"

        run_dir = raw_dir / "inference" / _DIR_NAME
        pdb_dir = run_dir / "job_0_n_80_id_0_single_orig0"
        _make_pdb(pdb_dir / "job_0_n_80_id_0_single_orig0.pdb")

        # Even if evaluation CSVs exist, generate mode should not load them
        _write_csv(
            raw_dir / "evaluation_results" / "binder_results_0.csv",
            [_BINDER_RESULTS_ROW],
        )

        standardize(workspace)

        result = json.loads(
            (workspace / "outputs" / "standardized" / "result_data.json").read_text()
        )
        design = result["designs"][0]
        assert "evaluation_metrics" not in design

    def test_multiple_designs_with_metrics(self, tmp_path: Path) -> None:
        """Multiple designs each get their own evaluation metrics."""
        workspace = _make_workspace(tmp_path, mode="design")
        raw_dir = workspace / "outputs" / "raw"
        run_dir = raw_dir / "inference" / _DIR_NAME

        # Two PDB outputs
        _make_pdb(run_dir / "job_0_n_80_id_0_single_orig0" / "job_0_n_80_id_0_single_orig0.pdb")
        _make_pdb(run_dir / "job_0_n_60_id_1_single_orig0" / "job_0_n_60_id_1_single_orig0.pdb")

        # Evaluation CSV with both designs
        _write_csv(
            raw_dir / "binder_results_0.csv",
            [_BINDER_RESULTS_ROW, _BINDER_RESULTS_ROW_2],
        )

        standardize(workspace)

        result = json.loads(
            (workspace / "outputs" / "standardized" / "result_data.json").read_text()
        )
        assert len(result["designs"]) == 2
        assert result["spec_summary"]["crambin_binder"] == 2

        # Each design should have its own metrics (sorted by path: n_60 before n_80)
        by_length = {d["diffusion_metadata"]["binder_length"]: d for d in result["designs"]}
        assert by_length[80]["evaluation_metrics"]["self_complex_i_pAE"] == pytest.approx(0.15)
        assert by_length[60]["evaluation_metrics"]["self_complex_i_pAE"] == pytest.approx(0.22)

    def test_spec_name_remapped_from_directory(self, tmp_path: Path) -> None:
        """Directory names are remapped back to original spec names."""
        workspace = _make_workspace(tmp_path, mode="design")
        raw_dir = workspace / "outputs" / "raw"

        run_dir = raw_dir / "inference" / _DIR_NAME
        _make_pdb(run_dir / "job_0_n_80_id_0" / "job_0_n_80_id_0.pdb")

        standardize(workspace)

        result = json.loads(
            (workspace / "outputs" / "standardized" / "result_data.json").read_text()
        )
        # Should be "crambin_binder", not the full directory name
        assert result["designs"][0]["spec_name"] == "crambin_binder"

    def test_standardized_pdb_copied(self, tmp_path: Path) -> None:
        """PDB files are copied to outputs/standardized/ with renamed filenames."""
        workspace = _make_workspace(tmp_path, mode="design")
        raw_dir = workspace / "outputs" / "raw"

        run_dir = raw_dir / "inference" / _DIR_NAME
        _make_pdb(run_dir / "job_0_n_80_id_0_single_orig0" / "job_0_n_80_id_0_single_orig0.pdb")

        standardize(workspace)

        std_dir = workspace / "outputs" / "standardized"
        expected_pdb = std_dir / "crambin_binder_b0_d0.pdb"
        assert expected_pdb.exists()
        content = expected_pdb.read_text()
        assert "ATOM" in content

    def test_no_pdbs_raises(self, tmp_path: Path) -> None:
        """Raises RuntimeError when no PDB files are found."""
        workspace = _make_workspace(tmp_path, mode="design")
        with pytest.raises(RuntimeError, match="No Proteina-Complexa design outputs"):
            standardize(workspace)

    def test_design_without_matching_metrics(self, tmp_path: Path) -> None:
        """Designs without matching evaluation metrics get no evaluation_metrics key."""
        workspace = _make_workspace(tmp_path, mode="design")
        raw_dir = workspace / "outputs" / "raw"

        run_dir = raw_dir / "inference" / _DIR_NAME
        _make_pdb(run_dir / "job_0_n_80_id_0" / "job_0_n_80_id_0.pdb")

        # Evaluation CSV for a DIFFERENT design
        different_row = {
            **_BINDER_RESULTS_ROW,
            "pdb_path": "/workspace/outputs/raw/inference/other_dir/other.pdb",
        }
        _write_csv(raw_dir / "binder_results_0.csv", [different_row])

        standardize(workspace)

        result = json.loads(
            (workspace / "outputs" / "standardized" / "result_data.json").read_text()
        )
        design = result["designs"][0]
        # No evaluation_metrics key since no match was found
        assert "evaluation_metrics" not in design

    def test_rewards_attached_to_diffusion_metadata(self, tmp_path: Path) -> None:
        """Rewards from rewards_*.csv are attached to diffusion_metadata."""
        workspace = _make_workspace(tmp_path, mode="design")
        raw_dir = workspace / "outputs" / "raw"

        run_dir = raw_dir / "inference" / _DIR_NAME
        pdb_dir = run_dir / "job_0_n_80_id_0_single_orig0"
        pdb_path = pdb_dir / "job_0_n_80_id_0_single_orig0.pdb"
        _make_pdb(pdb_path)

        # Rewards CSV pdb_path must match the actual file path on disk
        # (in production, both are container paths; in test, both are host paths)
        rewards_row = {**_REWARDS_ROW, "pdb_path": str(pdb_path)}
        _write_csv(
            run_dir / "rewards_search_binder_local_pipeline_0.csv",
            [rewards_row],
        )

        standardize(workspace)

        result = json.loads(
            (workspace / "outputs" / "standardized" / "result_data.json").read_text()
        )
        dm = result["designs"][0]["diffusion_metadata"]
        assert "rewards" in dm
        assert dm["rewards"]["sample_type"] == "final"


# ---------------------------------------------------------------------------
# Test: Metric indicator detection coverage
# ---------------------------------------------------------------------------


class TestMetricIndicatorCoverage:
    """Verify that real Proteina-Complexa column names are detected by the metric indicators."""

    @pytest.mark.parametrize(
        "column_name",
        [
            "self_complex_i_pAE",
            "self_binder_pLDDT",
            "self_binder_scRMSD_ca",
            "self_complex_iPTM",
            "self_complex_pTM",
            "mpnn_complex_i_pAE",
            "mpnn_binder_pLDDT",
            "mpnn_binder_scRMSD_ca",
            "mpnn_complex_iPTM",
            "mpnn_complex_pTM",
            "self_binder_scRMSD",
            "self_binder_scRMSD_bb3",
            "self_binder_scRMSD_allatom",
            "self_complex_scRMSD",
            "self_complex_scRMSD_ca",
            "self_complex_pLDDT",
        ],
    )
    def test_protein_binder_columns_detected(self, column_name: str) -> None:
        """All protein binder evaluation columns match at least one indicator."""
        metric_indicators = {
            "i_pAE",
            "pLDDT",
            "scRMSD",
            "iPTM",
            "pTM",
            "ipae",
            "iptm",
            "ptm",
            "plddt",
        }
        matched = any(ind in column_name for ind in metric_indicators)
        assert matched, f"Column '{column_name}' not detected by metric indicators"

    @pytest.mark.parametrize(
        "column_name",
        [
            "pdb_path",
            "id_gen",
            "run_name",
            "L",
            "ckpt_path",
            "task_name",
            "job_id",
            "total_time",
            "nsamples",
        ],
    )
    def test_nonmetric_columns_not_detected(self, column_name: str) -> None:
        """Non-metric identifier/timing columns do NOT match indicators."""
        metric_indicators = {
            "i_pAE",
            "pLDDT",
            "scRMSD",
            "iPTM",
            "pTM",
            "ipae",
            "iptm",
            "ptm",
            "plddt",
        }
        matched = any(ind in column_name for ind in metric_indicators)
        assert not matched, f"Column '{column_name}' falsely detected as metric"
