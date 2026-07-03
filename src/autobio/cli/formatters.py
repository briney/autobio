"""Output formatting for the autobio CLI."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

from autobio.core.catalog import tool_categories

if TYPE_CHECKING:
    from autobio.core.catalog import Tool
    from autobio.core.container import ImageInfo
    from autobio.core.result import RunResult


class OutputFormat(StrEnum):
    """Supported CLI output formats."""

    JSON = "json"
    TABLE = "table"


_console = Console()
_err_console = Console(stderr=True)


def format_tool_list(
    tools: dict[str, Tool],
    fmt: OutputFormat = OutputFormat.TABLE,
) -> str:
    """Format a mapping of catalog tool names to Tools for display, sorted by name.

    Args:
        tools: Catalog tool name → :class:`~autobio.core.catalog.Tool`.
        fmt: Output format.

    Returns:
        Formatted string.
    """
    if fmt == OutputFormat.JSON:
        rows = [
            {
                "name": name,
                "display_name": tool.display_name,
                "category": tool.category.value,
                "categories": [c.value for c in tool_categories(tool)],
                "gpu": tool.requires_gpu,
                "version": tool.version,
                "description": tool.description,
                "modes": list(tool.modes),
                "keywords": list(tool.keywords),
            }
            for name, tool in sorted(tools.items())
        ]
        return json.dumps(rows, indent=2)

    if not tools:
        return "No tools registered."

    table = Table(title="Available Tools")
    table.add_column("Name", style="cyan")
    table.add_column("Category")
    table.add_column("GPU")
    table.add_column("Version")
    table.add_column("Modes")
    table.add_column("Description")

    for name, tool in sorted(tools.items()):
        table.add_row(
            name,
            tool.category.value,
            "yes" if tool.requires_gpu else "no",
            tool.version,
            ", ".join(tool.modes),
            tool.description,
        )

    return _render_table(table)


def format_tool_info_catalog(tool: Tool, fmt: OutputFormat = OutputFormat.TABLE) -> str:
    """Format detailed info for a catalog Tool (with its Modes).

    Args:
        tool: The catalog :class:`~autobio.core.catalog.Tool`.
        fmt: Output format.

    Returns:
        Formatted string.
    """
    if fmt == OutputFormat.JSON:
        modes = [
            {
                "name": mode.name,
                "display_name": mode.display_name,
                "description": mode.description,
                "category": (mode.category or tool.category).value,
                "default_timeout": mode.default_timeout,
                "supports_batch": mode.supports_batch,
                "notes": list(mode.notes),
                "input_schema": mode.input_schema.model_json_schema(),
                "output_schema": mode.output_schema.model_json_schema(),
            }
            for mode in tool.modes.values()
        ]
        data = {
            "name": tool.name,
            "display_name": tool.display_name,
            "category": tool.category.value,
            "categories": [c.value for c in tool_categories(tool)],
            "version": tool.version,
            "image_tag": tool.image_tag,
            "requires_gpu": tool.requires_gpu,
            "gpu_count": tool.gpu_count,
            "description": tool.description,
            "keywords": list(tool.keywords),
            "notes": list(tool.notes),
            "default_mode": tool.default_mode,
            "modes": modes,
        }
        return json.dumps(data, indent=2)

    table = Table(title=f"Tool: {tool.name}", show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Display Name", tool.display_name)
    table.add_row("Category", tool.category.value)
    table.add_row("Categories", ", ".join(c.value for c in tool_categories(tool)))
    table.add_row("Image", tool.image_tag)
    table.add_row("GPU Required", "yes" if tool.requires_gpu else "no")
    table.add_row("GPU Count", str(tool.gpu_count))
    table.add_row("Version", tool.version)
    table.add_row("Description", tool.description)
    if tool.keywords:
        table.add_row("Keywords", ", ".join(tool.keywords))
    if tool.notes:
        table.add_row("Notes", "\n".join(f"- {n}" for n in tool.notes))
    table.add_row("Default Mode", tool.default_mode)
    for mode in tool.modes.values():
        category = (mode.category or tool.category).value
        table.add_row(
            f"Mode: {mode.name}",
            f"{mode.display_name} — {mode.description} "
            f"(category={category}, timeout={mode.default_timeout}s)",
        )
        if mode.notes:
            table.add_row("", "\n".join(f"- {n}" for n in mode.notes))
    return _render_table(table)


def format_run_result(
    output_data: dict[str, object],
    fmt: OutputFormat = OutputFormat.TABLE,
) -> str:
    """Format a tool run output for display.

    Args:
        output_data: Serialised output model dict.
        fmt: Output format.

    Returns:
        Formatted string.
    """
    if fmt == OutputFormat.JSON:
        return json.dumps(output_data, indent=2, default=str)

    table = Table(title="Run Result")
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    for key, value in output_data.items():
        table.add_row(key, _format_value(value))

    return _render_table(table)


def format_workspace_result(result: RunResult, fmt: OutputFormat = OutputFormat.TABLE) -> str:
    """Format a workspace's result.json for display.

    Args:
        result: Parsed :class:`RunResult`.
        fmt: Output format.

    Returns:
        Formatted string.
    """
    if fmt == OutputFormat.JSON:
        return result.model_dump_json(indent=2)

    table = Table(title="Workspace Result")
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    table.add_row("Status", result.status)
    table.add_row("Exit Code", str(result.exit_code))
    table.add_row("Phase", result.phase)
    table.add_row("Wall Time", f"{result.wall_time_seconds:.1f}s")
    table.add_row("Progress", f"{result.completed}/{result.total}")

    if result.error_type:
        table.add_row("Error Type", result.error_type)
    if result.error_message:
        table.add_row("Error Message", result.error_message)
    if result.gpu_ids:
        table.add_row("GPUs", ", ".join(str(g) for g in result.gpu_ids))

    return _render_table(table)


def format_image_list(images: list[ImageInfo], fmt: OutputFormat = OutputFormat.TABLE) -> str:
    """Format a list of locally cached container images.

    Args:
        images: Image metadata list.
        fmt: Output format.

    Returns:
        Formatted string.
    """
    if fmt == OutputFormat.JSON:
        rows = [
            {
                "uri": img.uri,
                "tag": img.tag,
                "size_mb": round(img.size / 1_000_000, 1),
                "created": img.created.isoformat(),
            }
            for img in images
        ]
        return json.dumps(rows, indent=2)

    if not images:
        return "No autobio images found locally."

    table = Table(title="Local Images")
    table.add_column("URI", style="cyan")
    table.add_column("Tag")
    table.add_column("Size (MB)", justify="right")
    table.add_column("Created")

    for img in images:
        table.add_row(
            img.uri,
            img.tag,
            f"{img.size / 1_000_000:.1f}",
            img.created.strftime("%Y-%m-%d %H:%M"),
        )

    return _render_table(table)


def print_error(message: str) -> None:
    """Print an error message to stderr."""
    _err_console.print(f"[bold red]Error:[/bold red] {message}")


def _render_table(table: Table) -> str:
    """Render a Rich table to a plain string."""
    console = Console(file=None, force_terminal=False, width=120)
    with console.capture() as capture:
        console.print(table)
    return capture.get()


def _format_value(value: object) -> str:
    """Convert a value to a display string, pretty-printing nested structures."""
    if isinstance(value, dict | list):
        return json.dumps(value, indent=2, default=str)
    return str(value)
