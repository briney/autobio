"""Tests for autobio CLI subcommands."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 - used in fixture type hints at runtime
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from autobio.cli.main import app
from autobio.core.catalog import CATALOG, Mode, Tool, register
from autobio.core.container import ImageInfo
from autobio.core.registry import ToolCategory
from autobio.core.result import ContainerNotFoundError, RunResult, ToolExecutionError
from autobio.schemas.base import BaseInput, BaseOutput, RunMetadata

runner = CliRunner()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockInput(BaseInput):
    sequences: dict[str, str]


class _MockOutput(BaseOutput):
    scores: list[float]


def _make_catalog_tool(
    name: str = "mock-tool",
    *,
    category: ToolCategory = ToolCategory.STRUCTURE_PREDICTION,
    requires_gpu: bool = True,
    image_tag: str = "mock-tool:1.0",
    mode_name: str = "default",
) -> Tool:
    """Build a mock catalog Tool with a single Mode, for CLI-test fake-tool injection."""
    return Tool(
        name=name,
        display_name=name,
        category=category,
        description="A mock tool for testing.",
        version="1.0",
        image_tag=image_tag,
        requires_gpu=requires_gpu,
        gpu_count=1,
        default_mode=mode_name,
        modes={
            mode_name: Mode(
                name=mode_name,
                display_name="Default",
                description="Mock mode for testing.",
                input_schema=_MockInput,
                output_schema=_MockOutput,
                default_timeout=600,
            ),
        },
    )


def _write_result_json(ws_dir: Path, *, status: str = "success") -> None:
    """Write a minimal result.json into a workspace directory."""
    ws_dir.mkdir(parents=True, exist_ok=True)
    result = RunResult(
        status=status,
        exit_code=0 if status == "success" else 1,
        phase="inference",
        wall_time_seconds=10.0,
    )
    (ws_dir / "result.json").write_text(result.model_dump_json())


# ---------------------------------------------------------------------------
# Catalog fixture — snapshot/restore around each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_catalog():
    """Snapshot, clear, and restore CATALOG around each test."""
    saved = dict(CATALOG)
    CATALOG.clear()
    yield
    CATALOG.clear()
    CATALOG.update(saved)


# ---------------------------------------------------------------------------
# autobio list
# ---------------------------------------------------------------------------


class TestListCommand:
    def test_empty_registry_json(self) -> None:
        result = runner.invoke(app, ["list", "--format", "json"])
        assert result.exit_code == 0
        assert json.loads(result.output) == []

    def test_populated_registry_json(self) -> None:
        register(_make_catalog_tool("mock-tool"))
        result = runner.invoke(app, ["list", "--format", "json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert len(parsed) == 1
        assert parsed[0]["name"] == "mock-tool"

    def test_category_filter(self) -> None:
        register(_make_catalog_tool("sp", category=ToolCategory.STRUCTURE_PREDICTION))
        register(_make_catalog_tool("emb", category=ToolCategory.EMBEDDING))
        result = runner.invoke(app, ["list", "--category", "embedding", "--format", "json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert len(parsed) == 1
        assert parsed[0]["name"] == "emb"

    def test_table_output(self) -> None:
        register(_make_catalog_tool("mock-tool"))
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "mock-tool" in result.output

    def test_empty_table(self) -> None:
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "No tools registered" in result.output


# ---------------------------------------------------------------------------
# autobio info
# ---------------------------------------------------------------------------


class TestInfoCommand:
    def test_success_json(self) -> None:
        register(_make_catalog_tool("mock-tool"))
        result = runner.invoke(app, ["info", "mock-tool", "--format", "json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["name"] == "mock-tool"
        assert "modes" in parsed
        assert "input_schema" in parsed["modes"][0]

    def test_unknown_tool_exits_1(self) -> None:
        result = runner.invoke(app, ["info", "nonexistent"])
        assert result.exit_code == 1

    def test_table_output(self) -> None:
        register(_make_catalog_tool("mock-tool"))
        result = runner.invoke(app, ["info", "mock-tool"])
        assert result.exit_code == 0
        assert "mock-tool" in result.output


# ---------------------------------------------------------------------------
# autobio result
# ---------------------------------------------------------------------------


class TestResultCommand:
    def test_valid_workspace_json(self, tmp_path: Path) -> None:
        ws_dir = tmp_path / "workspace"
        _write_result_json(ws_dir)
        result = runner.invoke(app, ["result", str(ws_dir), "--format", "json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["status"] == "success"

    def test_valid_workspace_table(self, tmp_path: Path) -> None:
        ws_dir = tmp_path / "workspace"
        _write_result_json(ws_dir)
        result = runner.invoke(app, ["result", str(ws_dir)])
        assert result.exit_code == 0
        assert "success" in result.output

    def test_missing_result_json_exits_1(self, tmp_path: Path) -> None:
        ws_dir = tmp_path / "workspace"
        ws_dir.mkdir()
        result = runner.invoke(app, ["result", str(ws_dir)])
        assert result.exit_code == 1

    def test_failure_result(self, tmp_path: Path) -> None:
        ws_dir = tmp_path / "workspace"
        _write_result_json(ws_dir, status="failure")
        result = runner.invoke(app, ["result", str(ws_dir), "--format", "json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["status"] == "failure"


# ---------------------------------------------------------------------------
# autobio run
# ---------------------------------------------------------------------------


class TestRunCommand:
    def test_missing_config_exits_1(self) -> None:
        register(_make_catalog_tool("mock-tool"))
        result = runner.invoke(app, ["run", "mock-tool", "--config", "/nonexistent.json"])
        assert result.exit_code == 1

    def test_unknown_tool_exits_1(self, tmp_path: Path) -> None:
        config_file = tmp_path / "input.json"
        config_file.write_text('{"sequences": {"A": "MGKL"}}')
        result = runner.invoke(app, ["run", "nonexistent", "--config", str(config_file)])
        assert result.exit_code == 1

    def test_invalid_input_exits_1(self, tmp_path: Path) -> None:
        register(_make_catalog_tool("mock-tool"))
        config_file = tmp_path / "input.json"
        # Missing required 'sequences' field
        config_file.write_text("{}")
        with patch("autobio.cli.run.get_runner") as mock_get_runner:
            mock_runner = MagicMock()
            mock_get_runner.return_value = mock_runner
            result = runner.invoke(app, ["run", "mock-tool", "--config", str(config_file)])
        assert result.exit_code == 1

    @patch("autobio.cli.run.get_runner")
    def test_successful_run_json(self, mock_get_runner: MagicMock, tmp_path: Path) -> None:
        register(_make_catalog_tool("mock-tool"))
        config_file = tmp_path / "input.json"
        config_file.write_text('{"sequences": {"A": "MGKL"}}')

        mock_output = MagicMock()
        mock_output.model_dump.return_value = {"scores": [0.9], "status": "ok"}
        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_output
        mock_get_runner.return_value = mock_runner

        result = runner.invoke(
            app, ["run", "mock-tool", "--config", str(config_file), "--format", "json"]
        )
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["scores"] == [0.9]

    @patch("autobio.cli.run.get_runner")
    def test_autobio_error_exits_1(self, mock_get_runner: MagicMock, tmp_path: Path) -> None:
        register(_make_catalog_tool("mock-tool"))
        config_file = tmp_path / "input.json"
        config_file.write_text('{"sequences": {"A": "MGKL"}}')

        mock_runner = MagicMock()
        mock_runner.run.side_effect = ToolExecutionError(
            phase="inference",
            exit_code=1,
            error_message="OOM",
            logs="",
            wall_time=5.0,
        )
        mock_get_runner.return_value = mock_runner

        result = runner.invoke(app, ["run", "mock-tool", "--config", str(config_file)])
        assert result.exit_code == 1

    @patch("autobio.cli.run.get_runner")
    def test_gpu_and_timeout_forwarded(self, mock_get_runner: MagicMock, tmp_path: Path) -> None:
        register(_make_catalog_tool("mock-tool"))
        config_file = tmp_path / "input.json"
        config_file.write_text('{"sequences": {"A": "MGKL"}}')

        mock_output = MagicMock()
        mock_output.model_dump.return_value = {"scores": [0.9]}
        mock_runner = MagicMock()
        mock_runner.run.return_value = mock_output
        mock_get_runner.return_value = mock_runner

        runner.invoke(
            app,
            [
                "run",
                "mock-tool",
                "--config",
                str(config_file),
                "--gpu",
                "0,1",
                "--timeout",
                "120",
                "--format",
                "json",
            ],
        )

        mock_runner.run.assert_called_once()
        call_kwargs = mock_runner.run.call_args
        assert call_kwargs.kwargs["gpu"] == "0,1"
        assert call_kwargs.kwargs["timeout"] == 120


# ---------------------------------------------------------------------------
# autobio images
# ---------------------------------------------------------------------------


class TestImagesCommand:
    @patch("autobio.cli.images.ContainerManager")
    def test_images_json_empty(self, mock_cm_cls: MagicMock) -> None:
        mock_cm_cls.return_value.list_images.return_value = []
        result = runner.invoke(app, ["images", "--format", "json"])
        assert result.exit_code == 0
        assert json.loads(result.output) == []

    @patch("autobio.cli.images.ContainerManager")
    def test_images_json_populated(self, mock_cm_cls: MagicMock) -> None:
        from datetime import UTC, datetime

        mock_cm_cls.return_value.list_images.return_value = [
            ImageInfo(
                uri="ghcr.io/briney/autobio-mock:1.0",
                tag="1.0",
                size=500_000_000,
                created=datetime(2026, 1, 15, 10, 30, tzinfo=UTC),
            )
        ]
        result = runner.invoke(app, ["images", "--format", "json"])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert len(parsed) == 1
        assert parsed[0]["uri"] == "ghcr.io/briney/autobio-mock:1.0"

    @patch("autobio.cli.images.ContainerManager")
    def test_images_table_empty(self, mock_cm_cls: MagicMock) -> None:
        mock_cm_cls.return_value.list_images.return_value = []
        result = runner.invoke(app, ["images"])
        assert result.exit_code == 0
        assert "No autobio images found locally" in result.output


# ---------------------------------------------------------------------------
# autobio pull
# ---------------------------------------------------------------------------


class TestPullCommand:
    def test_no_args_exits_1(self) -> None:
        result = runner.invoke(app, ["pull"])
        assert result.exit_code == 1

    @patch("autobio.cli.images.ContainerManager")
    def test_pull_single_tool(self, mock_cm_cls: MagicMock) -> None:
        register(_make_catalog_tool("mock-tool"))
        result = runner.invoke(app, ["pull", "mock-tool"])
        assert result.exit_code == 0
        mock_cm_cls.return_value.pull_image.assert_called_once()

    def test_pull_unknown_tool_exits_1(self) -> None:
        result = runner.invoke(app, ["pull", "nonexistent"])
        assert result.exit_code == 1

    @patch("autobio.cli.images.ContainerManager")
    def test_pull_all(self, mock_cm_cls: MagicMock) -> None:
        # Distinct image tags so both are pulled (identical-image tools are
        # deduped — see test_pull_resolves_catalog_tools_and_dedupes).
        register(_make_catalog_tool("tool-a", image_tag="tool-a:1.0"))
        register(_make_catalog_tool("tool-b", image_tag="tool-b:1.0"))
        result = runner.invoke(app, ["pull", "--all"])
        assert result.exit_code == 0
        assert mock_cm_cls.return_value.pull_image.call_count == 2

    @patch("autobio.cli.images.ContainerManager")
    def test_pull_failure_exits_1(self, mock_cm_cls: MagicMock) -> None:
        register(_make_catalog_tool("mock-tool"))
        mock_cm_cls.return_value.pull_image.side_effect = ContainerNotFoundError("not found")
        result = runner.invoke(app, ["pull", "mock-tool"])
        assert result.exit_code == 1

    @patch("autobio.cli.images.ContainerManager")
    def test_pull_all_empty_registry(self, mock_cm_cls: MagicMock) -> None:
        result = runner.invoke(app, ["pull", "--all"])
        assert result.exit_code == 0
        assert "No tools registered" in result.output

    @patch("autobio.cli.images.ContainerManager")
    def test_pull_resolves_catalog_tools_and_dedupes(self, mock_cm_cls: MagicMock) -> None:
        """Catalog Tools with per-mode image overrides pull all distinct images.

        Registers a catalog Tool with two modes — one falling back to the
        Tool's own ``image_tag``, one overriding with an image tag that
        collides (post-prefix) with a second catalog Tool's image — to
        exercise both multi-image pulls for a single Tool and cross-Tool URI
        dedup for ``pull --all``.
        """
        shared_image_tag = "mock-tool:1.0"
        register(_make_catalog_tool("other-tool", image_tag=shared_image_tag))

        register(
            Tool(
                name="cat-tool",
                display_name="CatTool",
                category=ToolCategory.SCORING,
                description="Catalog tool for pull test.",
                version="1.0.0",
                image_tag="cat-tool:1.0.0",
                requires_gpu=False,
                gpu_count=0,
                default_mode="a",
                modes={
                    "a": Mode("a", "A", "mode a", _MockInput, _MockOutput, default_timeout=1),
                    "b": Mode(
                        "b",
                        "B",
                        "mode b",
                        _MockInput,
                        _MockOutput,
                        default_timeout=1,
                        image_tag=shared_image_tag,  # collides with other-tool's image
                    ),
                },
            )
        )

        # -- `pull cat-tool` pulls both the Tool's own image and the mode-b
        # override, and nothing else.
        result = runner.invoke(app, ["pull", "cat-tool"])
        assert result.exit_code == 0, result.output
        pulled = {call.args[0] for call in mock_cm_cls.return_value.pull_image.call_args_list}
        assert pulled == {
            "ghcr.io/briney/autobio-cat-tool:1.0.0",
            "ghcr.io/briney/autobio-mock-tool:1.0",
        }

        # -- `pull --all` pulls other-tool's image and the catalog tool's
        # image(s), deduping the URI shared between other-tool and cat-tool's
        # mode-b override so it is only pulled once.
        mock_cm_cls.return_value.pull_image.reset_mock()
        result = runner.invoke(app, ["pull", "--all"])
        assert result.exit_code == 0, result.output
        pulled_list = [call.args[0] for call in mock_cm_cls.return_value.pull_image.call_args_list]
        assert sorted(pulled_list) == [
            "ghcr.io/briney/autobio-cat-tool:1.0.0",
            "ghcr.io/briney/autobio-mock-tool:1.0",
        ]
        assert len(pulled_list) == len(set(pulled_list))  # each unique uri pulled exactly once


# ---------------------------------------------------------------------------
# autobio --help
# ---------------------------------------------------------------------------


class TestHelpOutput:
    def test_main_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "autobio" in result.output.lower() or "Usage" in result.output

    @pytest.mark.parametrize("command", ["list", "info", "run", "result", "pull", "images"])
    def test_subcommand_help(self, command: str) -> None:
        result = runner.invoke(app, [command, "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# autobio run --mode (catalog tools)
# ---------------------------------------------------------------------------


class _RunInput(BaseInput):
    pass


class _RunOutput(BaseOutput):
    pass


def _register_run_tool() -> None:
    if "runtool" in CATALOG:
        return
    register(
        Tool(
            name="runtool",
            display_name="RunTool",
            category=ToolCategory.SCORING,
            description="run demo",
            version="1.0.0",
            image_tag="runtool:1.0.0",
            requires_gpu=False,
            gpu_count=0,
            default_mode="a",
            modes={
                "a": Mode("a", "A", "a", _RunInput, _RunOutput, default_timeout=1),
                "b": Mode("b", "B", "b", _RunInput, _RunOutput, default_timeout=1),
            },
        )
    )


def test_run_forwards_mode_for_catalog_tool(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    _register_run_tool()
    cfg = tmp_path / "config.json"
    cfg.write_text("{}")

    mock_runner = MagicMock()
    mock_output = _RunOutput(
        metadata=RunMetadata(
            tool_name="runtool",
            tool_version="1.0.0",
            image_uri="runtool:1.0.0",
            wall_time_seconds=0.1,
            gpu_ids=None,
            workspace_path=tmp_path,
            timestamp=datetime.now(tz=UTC),
        ),
        raw_output_path=tmp_path,
    )
    mock_runner.run.return_value = mock_output

    with patch("autobio.cli.run.get_runner", return_value=mock_runner):
        result = CliRunner().invoke(
            app, ["run", "runtool", "--mode", "b", "--config", str(cfg), "--gpu", "none"]
        )

    assert result.exit_code == 0, result.output
    assert mock_runner.run.call_args.kwargs["mode"] == "b"


def test_run_unknown_mode_exits_1(tmp_path: Path) -> None:
    _register_run_tool()
    cfg = tmp_path / "config.json"
    cfg.write_text("{}")

    mock_runner = MagicMock()

    with patch("autobio.cli.run.get_runner", return_value=mock_runner):
        result = CliRunner().invoke(app, ["run", "runtool", "--config", str(cfg), "--mode", "nope"])

    assert result.exit_code == 1
    assert "Unknown mode" in result.output
    mock_runner.run.assert_not_called()


# ---------------------------------------------------------------------------
# autobio list / info — real coverage with the migrated freesasa Tool
#
# freesasa is the first real (non-mock) catalog Tool, so these tests exercise
# the actual `list`/`info` CLI paths with a genuine multi-mode Tool alongside
# another catalog Tool, instead of only testing the formatters directly.
# ---------------------------------------------------------------------------


def _register_freesasa() -> None:
    """Re-register the real freesasa Tool into CATALOG after fixture-clearing.

    Importing ``autobio.tools`` registers freesasa once (module import is
    cached), but the autouse ``_clean_catalog`` fixture snapshot-clear-restores
    CATALOG around every test in this file for isolation — so tests that need
    freesasa present must re-register it explicitly, using the module's own
    Tool object (not a reconstruction).
    """
    import autobio.tools  # noqa: F401 - ensures the module (and its schemas) are importable
    from autobio.tools.freesasa import FREESASA_TOOL

    if "freesasa" not in CATALOG:
        register(FREESASA_TOOL)


class TestListInfoWithFreesasa:
    def test_list_json_includes_freesasa_and_another_tool(self) -> None:
        _register_freesasa()
        register(_make_catalog_tool("mock-tool", category=ToolCategory.SCORING))

        result = runner.invoke(app, ["list", "--format", "json"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        names = [row["name"] for row in parsed]
        assert names.count("freesasa") == 1
        assert "mock-tool" in names

        freesasa_row = next(row for row in parsed if row["name"] == "freesasa")
        assert freesasa_row["modes"] == ["sasa", "bsa"]

    def test_list_category_filter_includes_freesasa(self) -> None:
        _register_freesasa()

        result = runner.invoke(app, ["list", "--category", "scoring", "--format", "json"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert any(row["name"] == "freesasa" for row in parsed)

    def test_info_freesasa_json_returns_catalog_payload(self) -> None:
        _register_freesasa()

        result = runner.invoke(app, ["info", "freesasa", "--format", "json"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert [mode["name"] for mode in parsed["modes"]] == ["sasa", "bsa"]
        for mode in parsed["modes"]:
            assert "input_schema" in mode
            assert "output_schema" in mode
