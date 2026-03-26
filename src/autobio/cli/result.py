"""`autobio result` — inspect a previous run from its workspace."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - Typer evaluates annotations at runtime
from typing import Annotated

import typer

from autobio.cli.formatters import OutputFormat, format_workspace_result, print_error
from autobio.core.workspace import Workspace


def result_cmd(
    workspace_dir: Annotated[Path, typer.Argument(help="Path to workspace directory.")],
    fmt: Annotated[
        OutputFormat,
        typer.Option("--format", "-f", help="Output format."),
    ] = OutputFormat.TABLE,
) -> None:
    """Read and display result.json from a workspace directory."""
    workspace = Workspace(workspace_dir)

    if not workspace.result_path.exists():
        print_error(f"No result.json found in {workspace_dir}")
        raise typer.Exit(code=1)

    try:
        run_result = workspace.read_result()
    except Exception as exc:
        print_error(f"Failed to parse result.json: {exc}")
        raise typer.Exit(code=1) from None

    typer.echo(format_workspace_result(run_result, fmt))
