"""`autobio info` — show details for a single tool."""

from __future__ import annotations

from typing import Annotated

import typer

from autobio.cli.formatters import OutputFormat, format_tool_info, print_error
from autobio.core.registry import get_tool


def info_cmd(
    tool: Annotated[str, typer.Argument(help="Tool name.")],
    fmt: Annotated[
        OutputFormat,
        typer.Option("--format", "-f", help="Output format."),
    ] = OutputFormat.TABLE,
) -> None:
    """Show detailed information about a tool."""
    try:
        entry = get_tool(tool)
    except KeyError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from None
    typer.echo(format_tool_info(tool, entry, fmt))
