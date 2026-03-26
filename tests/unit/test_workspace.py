"""Tests for autobio.core.workspace."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 - used in fixture type hints

import pytest

from autobio.core.workspace import Workspace


class TestWorkspaceCreate:
    def test_creates_all_subdirectories(self, tmp_path: Path) -> None:
        ws = Workspace.create(tmp_path / "ws")
        assert ws.inputs_dir.is_dir()
        assert ws.raw_output_dir.is_dir()
        assert ws.std_output_dir.is_dir()
        assert ws.logs_dir.is_dir()

    def test_temp_workspace_flagged(self) -> None:
        ws = Workspace.create()
        assert ws._is_temp is True
        assert ws.root.exists()
        ws.cleanup()

    def test_explicit_dir_not_flagged_temp(self, tmp_path: Path) -> None:
        ws = Workspace.create(tmp_path / "ws")
        assert ws._is_temp is False

    def test_path_properties(self, tmp_path: Path) -> None:
        ws = Workspace.create(tmp_path / "ws")
        assert ws.config_path == ws.root / "config.json"
        assert ws.result_path == ws.root / "result.json"


class TestWriteConfig:
    def test_round_trip(self, tmp_workspace: Workspace) -> None:
        cfg = {"sequences": {"A": "MKWV"}, "num_models": 3}
        tmp_workspace.write_config(cfg)
        loaded = json.loads(tmp_workspace.config_path.read_text())
        assert loaded == cfg

    def test_overwrites(self, tmp_workspace: Workspace) -> None:
        tmp_workspace.write_config({"a": 1})
        tmp_workspace.write_config({"b": 2})
        loaded = json.loads(tmp_workspace.config_path.read_text())
        assert loaded == {"b": 2}


class TestWriteInputFile:
    def test_write_text(self, tmp_workspace: Workspace) -> None:
        p = tmp_workspace.write_input_file("seq.fasta", ">A\nMKWV\n")
        assert p.exists()
        assert p.read_text() == ">A\nMKWV\n"
        assert p.parent == tmp_workspace.inputs_dir

    def test_write_bytes(self, tmp_workspace: Workspace) -> None:
        data = b"\x00\x01\x02"
        p = tmp_workspace.write_input_file("binary.dat", data)
        assert p.read_bytes() == data


class TestReadResult:
    def test_reads_valid_result(self, tmp_workspace: Workspace) -> None:
        result_data = {
            "status": "success",
            "exit_code": 0,
            "phase": "complete",
            "wall_time_seconds": 5.0,
        }
        tmp_workspace.result_path.write_text(json.dumps(result_data))
        result = tmp_workspace.read_result()
        assert result.status == "success"
        assert result.wall_time_seconds == 5.0

    def test_missing_result_raises(self, tmp_workspace: Workspace) -> None:
        with pytest.raises(FileNotFoundError):
            tmp_workspace.read_result()


class TestCleanup:
    def test_cleanup_removes_temp(self) -> None:
        ws = Workspace.create()
        root = ws.root
        assert root.exists()
        ws.cleanup()
        assert not root.exists()

    def test_cleanup_preserves_explicit_dir(self, tmp_path: Path) -> None:
        ws = Workspace.create(tmp_path / "ws")
        ws.cleanup()
        assert ws.root.exists()
