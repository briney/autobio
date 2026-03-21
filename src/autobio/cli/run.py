"""`autobio run` — execute a tool inside its container."""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003 - Typer evaluates annotations at runtime
from typing import Annotated

import typer

from autobio.cli.formatters import OutputFormat, format_run_result, print_error
from autobio.core.config import AutobioConfig
from autobio.core.result import AutobioError
from autobio.tools import get_runner


def run_cmd(
    tool: Annotated[str, typer.Argument(help="Tool name.")],
    config: Annotated[
        Path,
        typer.Option("--config", help="Path to input config JSON file."),
    ],
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
    # Load input config
    try:
        config_data = json.loads(config.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print_error(f"Failed to read config file: {exc}")
        raise typer.Exit(code=1) from None

    # Resolve runner
    autobio_config = AutobioConfig.resolve()
    try:
        runner = get_runner(tool, autobio_config)
    except KeyError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from None

    # Validate input against the tool's schema
    try:
        input_data = runner.entry.input_schema.model_validate(config_data)
    except Exception as exc:
        print_error(f"Invalid input: {exc}")
        raise typer.Exit(code=1) from None

    # Execute
    try:
        output = runner.run(input_data, gpu=gpu, timeout=timeout, output_dir=output_dir)
    except AutobioError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from None

    output_dict = output.model_dump(mode="json")
    typer.echo(format_run_result(output_dict, fmt))
