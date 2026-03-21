"""Output formatting for the autobio CLI."""

from __future__ import annotations

import json
from enum import StrEnum
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from autobio.core.container import ImageInfo
    from autobio.core.registry import ToolEntry
    from autobio.core.result import RunResult


class OutputFormat(StrEnum):
    """Supported CLI output formats."""

    JSON = "json"
    TABLE = "table"


_console = Console()
_err_console = Console(stderr=True)


def format_tool_list(
    tools: dict[str, ToolEntry],
    fmt: OutputFormat = OutputFormat.TABLE,
) -> str:
    """Format a mapping of tool names to entries for display.

    Args:
        tools: Tool name to :class:`ToolEntry` mapping.
        fmt: Output format.

    Returns:
        Formatted string.
    """
    if fmt == OutputFormat.JSON:
        rows = [
            {
                "name": name,
                "category": entry.category.value,
                "gpu": entry.requires_gpu,
                "version": entry.version,
                "description": entry.description,
            }
            for name, entry in sorted(tools.items())
        ]
        return json.dumps(rows, indent=2)

    if not tools:
        return "No tools registered."

    table = Table(title="Available Tools")
    table.add_column("Name", style="cyan")
    table.add_column("Category")
    table.add_column("GPU")
    table.add_column("Version")
    table.add_column("Description")

    for name, entry in sorted(tools.items()):
        table.add_row(
            name,
            entry.category.value,
            "yes" if entry.requires_gpu else "no",
            entry.version,
            entry.description,
        )

    return _render_table(table)


def format_tool_info(name: str, entry: ToolEntry, fmt: OutputFormat = OutputFormat.TABLE) -> str:
    """Format detailed information about a single tool.

    Args:
        name: Tool name.
        entry: Tool metadata.
        fmt: Output format.

    Returns:
        Formatted string.
    """
    input_schema = entry.input_schema.model_json_schema()

    if fmt == OutputFormat.JSON:
        return json.dumps(
            {
                "name": name,
                "category": entry.category.value,
                "image_tag": entry.image_tag,
                "requires_gpu": entry.requires_gpu,
                "gpu_count": entry.gpu_count,
                "default_timeout": entry.default_timeout,
                "supports_batch": entry.supports_batch,
                "version": entry.version,
                "description": entry.description,
                "input_schema": input_schema,
            },
            indent=2,
        )

    table = Table(title=f"Tool: {name}", show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    table.add_row("Category", entry.category.value)
    table.add_row("Image", entry.image_tag)
    table.add_row("GPU Required", "yes" if entry.requires_gpu else "no")
    table.add_row("GPU Count", str(entry.gpu_count))
    table.add_row("Timeout", f"{entry.default_timeout}s")
    table.add_row("Batch Support", "yes" if entry.supports_batch else "no")
    table.add_row("Version", entry.version)
    table.add_row("Description", entry.description)
    table.add_row("Input Schema", json.dumps(input_schema, indent=2))

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
