"""`autobio list` — display registered tools."""

from __future__ import annotations

from typing import Annotated

import typer

from autobio.cli.formatters import OutputFormat, format_tool_list
from autobio.core.catalog import list_tools as list_catalog_tools
from autobio.core.registry import (
    ToolCategory,  # noqa: TC001 - Typer evaluates annotations at runtime
)


def list_tools_cmd(
    category: Annotated[
        ToolCategory | None,
        typer.Option("--category", "-c", help="Filter by tool category."),
    ] = None,
    fmt: Annotated[
        OutputFormat,
        typer.Option("--format", "-f", help="Output format."),
    ] = OutputFormat.TABLE,
) -> None:
    """List available tools."""
    tools = list_catalog_tools(category=category)
    typer.echo(format_tool_list(tools, fmt))
