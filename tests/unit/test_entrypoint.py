"""Tests for containers/base-entrypoint.sh.

Runs the entrypoint via subprocess with mock hook scripts in a temporary
workspace.  Skips automatically if ``jq`` or ``bc`` are not available.
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

_MISSING: list[str] = []
for _bin in ("jq", "bc", "bash"):
    if shutil.which(_bin) is None:
        _MISSING.append(_bin)

pytestmark = pytest.mark.skipif(
    len(_MISSING) > 0,
    reason=f"Required binaries not found: {', '.join(_MISSING)}",
)

ENTRYPOINT = Path(__file__).resolve().parents[2] / "containers" / "base-entrypoint.sh"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_hook(path: Path, body: str) -> None:
    """Write an executable bash hook script."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/usr/bin/env bash\nset -euo pipefail\n{body}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _build_test_script(
    tmp_path: Path,
    *,
    validate_body: str = 'echo "ok"; exit 0',
    run_body: str = 'echo "ran"; exit 0',
    standardize_body: str = 'echo "standardized"; exit 0',
) -> Path:
    """Create a wrapper script that redirects hook paths to temp dir.

    The real entrypoint hardcodes ``/opt/tool/`` and ``/workspace``.
    We generate a thin wrapper that overrides those paths via sed so
    we can test entirely in userland without root or Docker.
    """
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "config.json").write_text("{}")

    opt_tool = tmp_path / "opt" / "tool"
    _write_hook(opt_tool / "validate_config.sh", validate_body)
    _write_hook(opt_tool / "run.sh", run_body)
    _write_hook(opt_tool / "standardize.sh", standardize_body)

    # Create a patched copy of the entrypoint that uses our temp paths
    patched = tmp_path / "entrypoint.sh"
    original = ENTRYPOINT.read_text()
    patched_text = (
        original.replace(
            'WORKSPACE="/workspace"',
            f'WORKSPACE="{workspace}"',
        )
        .replace(
            "/opt/tool/validate_config.sh",
            str(opt_tool / "validate_config.sh"),
        )
        .replace(
            "/opt/tool/run.sh",
            str(opt_tool / "run.sh"),
        )
        .replace(
            "/opt/tool/standardize.sh",
            str(opt_tool / "standardize.sh"),
        )
    )
    patched.write_text(patched_text)
    patched.chmod(patched.stat().st_mode | stat.S_IEXEC)

    return patched


def _run_entrypoint(script: Path, timeout: float = 30) -> subprocess.CompletedProcess[str]:
    """Execute the patched entrypoint and return the result."""
    return subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PATH": os.environ.get("PATH", "/usr/bin:/bin")},
    )


def _read_result(tmp_path: Path) -> dict:
    """Read and parse the result.json from the workspace."""
    result_path = tmp_path / "workspace" / "result.json"
    assert result_path.exists(), f"result.json not found at {result_path}"
    return json.loads(result_path.read_text())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSuccessfulRun:
    """Test the happy-path three-phase execution."""

    def test_exit_code_zero(self, tmp_path: Path) -> None:
        script = _build_test_script(tmp_path)
        proc = _run_entrypoint(script)
        assert proc.returncode == 0

    def test_result_json_is_valid(self, tmp_path: Path) -> None:
        script = _build_test_script(tmp_path)
        _run_entrypoint(script)
        result = _read_result(tmp_path)
        assert result["status"] == "success"
        assert result["exit_code"] == 0
        assert result["phase"] == "complete"

    def test_result_json_fields(self, tmp_path: Path) -> None:
        script = _build_test_script(tmp_path)
        _run_entrypoint(script)
        result = _read_result(tmp_path)

        assert result["error_type"] is None
        assert result["error_message"] is None
        assert isinstance(result["wall_time_seconds"], (int, float))
        assert result["wall_time_seconds"] >= 0
        assert result["completed"] == 1
        assert result["total"] == 1

    def test_result_json_outputs_structure(self, tmp_path: Path) -> None:
        script = _build_test_script(tmp_path)
        _run_entrypoint(script)
        result = _read_result(tmp_path)

        assert "outputs" in result
        assert isinstance(result["outputs"]["raw_files"], list)
        assert isinstance(result["outputs"]["standardized_files"], list)

    def test_output_files_listed(self, tmp_path: Path) -> None:
        """When hooks produce output files, they appear in result.json."""
        raw_dir = tmp_path / "workspace" / "outputs" / "raw"
        std_dir = tmp_path / "workspace" / "outputs" / "standardized"

        script = _build_test_script(
            tmp_path,
            run_body=(f'mkdir -p "{raw_dir}"\necho "data" > "{raw_dir}/output.pdb"'),
            standardize_body=(f'mkdir -p "{std_dir}"\necho "{{}}" > "{std_dir}/result_data.json"'),
        )
        _run_entrypoint(script)
        result = _read_result(tmp_path)

        raw_files = result["outputs"]["raw_files"]
        std_files = result["outputs"]["standardized_files"]
        assert any("output.pdb" in f for f in raw_files)
        assert any("result_data.json" in f for f in std_files)

    def test_directories_created(self, tmp_path: Path) -> None:
        """Entrypoint creates the standard workspace subdirectories."""
        script = _build_test_script(tmp_path)
        _run_entrypoint(script)
        ws = tmp_path / "workspace"
        assert (ws / "outputs" / "raw").is_dir()
        assert (ws / "outputs" / "standardized").is_dir()
        assert (ws / "logs").is_dir()

    def test_log_files_created(self, tmp_path: Path) -> None:
        script = _build_test_script(tmp_path)
        _run_entrypoint(script)
        logs = tmp_path / "workspace" / "logs"
        assert (logs / "stdout.log").exists()
        assert (logs / "stderr.log").exists()

    def test_phase_file_final_state(self, tmp_path: Path) -> None:
        script = _build_test_script(tmp_path)
        _run_entrypoint(script)
        phase_file = tmp_path / "workspace" / "logs" / "phase.json"
        assert phase_file.exists()
        phase = json.loads(phase_file.read_text())
        assert phase["phase"] == "standardization"


class TestValidationFailure:
    """Phase 1 (validate_config.sh) failures."""

    def test_exit_code_nonzero(self, tmp_path: Path) -> None:
        script = _build_test_script(
            tmp_path,
            validate_body='echo "bad config" >&2; exit 1',
        )
        proc = _run_entrypoint(script)
        assert proc.returncode == 1

    def test_result_json_on_validation_failure(self, tmp_path: Path) -> None:
        script = _build_test_script(
            tmp_path,
            validate_body='echo "missing field X" >&2; exit 1',
        )
        _run_entrypoint(script)
        result = _read_result(tmp_path)

        assert result["status"] == "failed"
        assert result["phase"] == "setup"
        assert result["exit_code"] == 1
        assert result["error_type"] == "runtime"
        assert "Config validation failed" in result["error_message"]

    def test_later_phases_not_run(self, tmp_path: Path) -> None:
        """When validation fails, run.sh and standardize.sh should not execute."""
        marker = tmp_path / "run_was_called"
        script = _build_test_script(
            tmp_path,
            validate_body="exit 1",
            run_body=f'touch "{marker}"; exit 0',
        )
        _run_entrypoint(script)
        assert not marker.exists()


class TestRunFailure:
    """Phase 2 (run.sh) failures."""

    def test_result_json_on_run_failure(self, tmp_path: Path) -> None:
        script = _build_test_script(
            tmp_path,
            run_body="exit 42",
        )
        _run_entrypoint(script)
        result = _read_result(tmp_path)

        assert result["status"] == "failed"
        assert result["phase"] == "execution"
        assert result["error_type"] == "runtime"

    def test_standardize_not_run_after_run_failure(self, tmp_path: Path) -> None:
        marker = tmp_path / "std_was_called"
        script = _build_test_script(
            tmp_path,
            run_body="exit 1",
            standardize_body=f'touch "{marker}"; exit 0',
        )
        _run_entrypoint(script)
        assert not marker.exists()


class TestStandardizeFailure:
    """Phase 3 (standardize.sh) failures."""

    def test_result_json_on_standardize_failure(self, tmp_path: Path) -> None:
        script = _build_test_script(
            tmp_path,
            standardize_body="exit 1",
        )
        _run_entrypoint(script)
        result = _read_result(tmp_path)

        assert result["status"] == "failed"
        assert result["phase"] == "standardization"
        assert result["exit_code"] == 1
        assert result["error_type"] == "runtime"

    def test_exit_code_on_standardize_failure(self, tmp_path: Path) -> None:
        script = _build_test_script(
            tmp_path,
            standardize_body="exit 1",
        )
        proc = _run_entrypoint(script)
        assert proc.returncode == 1


class TestResultJsonValidity:
    """Verify result.json is always well-formed JSON."""

    @pytest.mark.parametrize(
        ("validate", "run", "standardize"),
        [
            ("exit 0", "exit 0", "exit 0"),
            ("exit 1", "exit 0", "exit 0"),
            ("exit 0", "exit 1", "exit 0"),
            ("exit 0", "exit 0", "exit 1"),
        ],
        ids=["all-pass", "validate-fail", "run-fail", "standardize-fail"],
    )
    def test_result_json_always_valid(
        self,
        tmp_path: Path,
        validate: str,
        run: str,
        standardize: str,
    ) -> None:
        script = _build_test_script(
            tmp_path,
            validate_body=validate,
            run_body=run,
            standardize_body=standardize,
        )
        _run_entrypoint(script)
        result = _read_result(tmp_path)

        # Must have all required keys
        for key in (
            "status",
            "exit_code",
            "phase",
            "error_type",
            "error_message",
            "wall_time_seconds",
            "completed",
            "total",
            "outputs",
        ):
            assert key in result, f"Missing key: {key}"

    def test_wall_time_is_positive(self, tmp_path: Path) -> None:
        script = _build_test_script(tmp_path)
        _run_entrypoint(script)
        result = _read_result(tmp_path)
        assert result["wall_time_seconds"] >= 0


class TestPhaseTracking:
    """Verify the phase file is updated at each stage."""

    def test_phase_setup_on_validate_fail(self, tmp_path: Path) -> None:
        script = _build_test_script(tmp_path, validate_body="exit 1")
        _run_entrypoint(script)
        phase = json.loads((tmp_path / "workspace" / "logs" / "phase.json").read_text())
        assert phase["phase"] == "setup"

    def test_phase_execution_on_run_fail(self, tmp_path: Path) -> None:
        script = _build_test_script(tmp_path, run_body="exit 1")
        _run_entrypoint(script)
        phase = json.loads((tmp_path / "workspace" / "logs" / "phase.json").read_text())
        assert phase["phase"] == "execution"

    def test_phase_standardization_on_success(self, tmp_path: Path) -> None:
        script = _build_test_script(tmp_path)
        _run_entrypoint(script)
        phase = json.loads((tmp_path / "workspace" / "logs" / "phase.json").read_text())
        assert phase["phase"] == "standardization"
