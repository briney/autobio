"""Unit tests for the migrated freesasa Tool (modes: sasa, bsa)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from autobio.core.catalog import get_tool
from autobio.core.config import AutobioConfig
from autobio.core.result import AutobioError
from autobio.tools import get_runner
from autobio.tools.freesasa import FreeSASARunner

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def _pdb(tmp_path: Path) -> Path:
    p = tmp_path / "complex.pdb"
    p.write_text("ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N\n")
    return p


def _make_runner(mode_name: str) -> FreeSASARunner:
    with patch("autobio.tools.base.ContainerManager"), patch("autobio.tools.base.GPUManager"):
        runner = FreeSASARunner("freesasa", AutobioConfig.resolve())
    runner.current_mode = get_tool("freesasa").modes[mode_name]
    return runner


def _written_config(runner: FreeSASARunner, input_data, tmp_path: Path) -> dict:
    from autobio.core.workspace import Workspace

    ws = Workspace.create(tmp_path / "ws")
    try:
        runner.prepare_workspace(input_data, ws)
        return json.loads((ws.root / "config.json").read_text())
    finally:
        ws.cleanup()


def test_freesasa_registered_as_tool_not_flat() -> None:
    import autobio.tools  # noqa: F401
    from autobio.core.catalog import CATALOG
    from autobio.core.registry import TOOL_REGISTRY
    from autobio.tools import TOOL_RUNNERS

    assert "freesasa" in CATALOG
    assert set(get_tool("freesasa").modes) == {"sasa", "bsa"}
    assert get_tool("freesasa").default_mode == "sasa"
    assert "freesasa" in TOOL_RUNNERS
    assert "freesasa_bsa" not in TOOL_RUNNERS and "freesasa_sasa" not in TOOL_RUNNERS
    assert "freesasa_bsa" not in TOOL_REGISTRY and "freesasa_sasa" not in TOOL_REGISTRY


def test_get_runner_freesasa_resolves_catalog_tool() -> None:
    import autobio.tools  # noqa: F401

    with patch("autobio.tools.base.ContainerManager"), patch("autobio.tools.base.GPUManager"):
        runner = get_runner("freesasa", AutobioConfig.resolve())
    assert runner.tool is not None and runner.tool.name == "freesasa"


def test_get_runner_removed_flat_name_raises() -> None:
    import autobio.tools  # noqa: F401

    with pytest.raises(KeyError, match="freesasa_bsa"):
        get_runner("freesasa_bsa", AutobioConfig.resolve())


def test_sasa_config_unchanged(_pdb: Path, tmp_path: Path) -> None:
    from autobio.schemas.scoring import FreeSASASASAInput

    runner = _make_runner("sasa")
    cfg = _written_config(runner, FreeSASASASAInput(structure_path=_pdb), tmp_path)
    assert cfg["mode"] == "sasa"
    assert cfg["structure_path"] == f"/workspace/inputs/{_pdb.name}"
    assert cfg["algorithm"] == "LeeRichards"
    assert cfg["probe_radius"] == 1.4
    assert cfg["per_residue"] is False
    assert cfg["output_dir"] == "/workspace/outputs/raw"
    assert "partner1" not in cfg and "partner2" not in cfg


def test_bsa_config_unchanged(_pdb: Path, tmp_path: Path) -> None:
    from autobio.schemas.scoring import FreeSASABSAInput

    runner = _make_runner("bsa")
    cfg = _written_config(
        runner,
        FreeSASABSAInput(
            structure_path=_pdb, partner1="A,B", partner2="C", algorithm="ShrakeRupley"
        ),
        tmp_path,
    )
    assert cfg["mode"] == "bsa"
    assert cfg["partner1"] == "A,B" and cfg["partner2"] == "C"
    assert cfg["algorithm"] == "ShrakeRupley"


def test_bsa_overlapping_partners_rejected(_pdb: Path, tmp_path: Path) -> None:
    from autobio.schemas.scoring import FreeSASABSAInput

    runner = _make_runner("bsa")
    with pytest.raises(AutobioError, match="overlap"):
        _written_config(
            runner, FreeSASABSAInput(structure_path=_pdb, partner1="A", partner2="A"), tmp_path
        )


def test_extra_shadowing_typed_field_rejected(_pdb: Path, tmp_path: Path) -> None:
    from autobio.schemas.scoring import FreeSASASASAInput

    runner = _make_runner("sasa")
    with pytest.raises(AutobioError, match="shadow typed input fields"):
        _written_config(
            runner,
            FreeSASASASAInput(structure_path=_pdb, extra={"probe_radius": 2.0}),
            tmp_path,
        )


def test_extra_unknown_key_passed_through(_pdb: Path, tmp_path: Path) -> None:
    from autobio.schemas.scoring import FreeSASASASAInput

    runner = _make_runner("sasa")
    cfg = _written_config(
        runner, FreeSASASASAInput(structure_path=_pdb, extra={"custom_flag": True}), tmp_path
    )
    assert cfg["custom_flag"] is True


def test_info_snapshot_freesasa() -> None:
    import autobio.tools  # noqa: F401
    from autobio.cli.formatters import OutputFormat, format_tool_info_catalog

    parsed = json.loads(format_tool_info_catalog(get_tool("freesasa"), OutputFormat.JSON))
    assert [m["name"] for m in parsed["modes"]] == ["sasa", "bsa"]
    sasa = parsed["modes"][0]
    struct = sasa["input_schema"]["properties"]["structure_path"]
    assert struct["x-autobio"]["widget"] == "file"
    assert "output_schema" in sasa
    bsa = parsed["modes"][1]
    assert "partner1" in bsa["input_schema"]["properties"]
