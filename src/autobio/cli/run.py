"""`autobio run` — execute a tool inside its container."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 - Typer evaluates annotations at runtime
from typing import Annotated

import typer

from autobio.cli.formatters import OutputFormat, format_run_result, print_error
from autobio.core.catalog import CATALOG
from autobio.core.catalog import get_tool as get_catalog_tool
from autobio.core.config import AutobioConfig
from autobio.core.result import AutobioError
from autobio.tools import get_runner


def run_cmd(
    tool: Annotated[str, typer.Argument(help="Tool name.")],
    config: Annotated[
        Path,
        typer.Option("--config", help="Path to input config JSON file."),
    ],
    mode: Annotated[
        str | None,
        typer.Option("--mode", help="Mode for a multi-mode tool (defaults to the tool's default)."),
    ] = None,
    gpu: Annotated[
        str,
        typer.Option("--gpu", help="GPU spec: 'auto', 'none', or comma-separated IDs."),
    ] = "auto",
    timeout: Annotated[
        int | None,
        typer.Option("--timeout", help="Maximum wall-clock seconds."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output-dir", help="Persist workspace to this directory."),
    ] = None,
    fmt: Annotated[
        OutputFormat,
        typer.Option("--format", "-f", help="Output format."),
    ] = OutputFormat.TABLE,
) -> None:
    """Run a tool with the given configuration."""
    try:
        config_data = json.loads(config.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print_error(f"Failed to read config file: {exc}")
        raise typer.Exit(code=1) from None

    autobio_config = AutobioConfig.resolve()
    try:
        runner = get_runner(tool, autobio_config)
    except KeyError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from None

    # Choose the input schema (per-mode for catalog tools) and the mode to forward.
    forward_mode: str | None = None
    if tool in CATALOG:
        catalog_tool = get_catalog_tool(tool)
        mode_name = mode if mode is not None else catalog_tool.default_mode
        if mode_name not in catalog_tool.modes:
            available = ", ".join(sorted(catalog_tool.modes))
            print_error(
                f"Unknown mode {mode_name!r} for tool {tool!r}. Available modes: {available}"
            )
            raise typer.Exit(code=1) from None
        input_schema = catalog_tool.modes[mode_name].input_schema
        forward_mode = mode_name
    else:
        if mode is not None:
            print_error(f"Tool {tool!r} does not support --mode.")
            raise typer.Exit(code=1) from None
        assert runner.entry is not None  # legacy branch: name is in TOOL_REGISTRY
        input_schema = runner.entry.input_schema

    try:
        input_data = input_schema.model_validate(config_data)
    except Exception as exc:
        print_error(f"Invalid input: {exc}")
        raise typer.Exit(code=1) from None

    try:
        output = runner.run(
            input_data, gpu=gpu, timeout=timeout, output_dir=output_dir, mode=forward_mode
        )
    except AutobioError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from None

    output_dict = output.model_dump(mode="json")
    typer.echo(format_run_result(output_dict, fmt))
