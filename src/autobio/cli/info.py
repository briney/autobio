"""`autobio info` — show details for a single tool."""

from __future__ import annotations

from typing import Annotated

import typer

from autobio.cli.formatters import (
    OutputFormat,
    format_tool_info,
    format_tool_info_catalog,
    print_error,
)
from autobio.core.catalog import CATALOG
from autobio.core.catalog import get_tool as get_catalog_tool
from autobio.core.registry import TOOL_REGISTRY
from autobio.core.registry import get_tool as get_registry_tool


def info_cmd(
    tool: Annotated[str, typer.Argument(help="Tool name.")],
    fmt: Annotated[
        OutputFormat,
        typer.Option("--format", "-f", help="Output format."),
    ] = OutputFormat.TABLE,
) -> None:
    """Show detailed information about a tool."""
    if tool in CATALOG:
        typer.echo(format_tool_info_catalog(get_catalog_tool(tool), fmt))
        return
    if tool in TOOL_REGISTRY:
        typer.echo(format_tool_info(tool, get_registry_tool(tool), fmt))
        return
    available = ", ".join(sorted(set(CATALOG) | set(TOOL_REGISTRY))) or "(none)"
    print_error(f"Unknown tool {tool!r}. Available tools: {available}")
    raise typer.Exit(code=1)
