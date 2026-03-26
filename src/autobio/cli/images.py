"""`autobio images` and `autobio pull` — manage container images."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from autobio.cli.formatters import OutputFormat, format_image_list, print_error
from autobio.core.config import AutobioConfig
from autobio.core.container import ContainerManager
from autobio.core.registry import TOOL_REGISTRY, get_tool
from autobio.core.result import ContainerNotFoundError

_console = Console()


def pull_cmd(
    tool: Annotated[
        str | None,
        typer.Argument(help="Tool name to pull, or omit with --all."),
    ] = None,
    all_tools: Annotated[
        bool,
        typer.Option("--all", help="Pull images for all registered tools."),
    ] = False,
) -> None:
    """Pull container image(s) for registered tools."""
    if not tool and not all_tools:
        print_error("Provide a tool name or use --all.")
        raise typer.Exit(code=1)

    config = AutobioConfig.resolve()
    manager = ContainerManager(config)

    if all_tools:
        entries = list(TOOL_REGISTRY.items())
        if not entries:
            typer.echo("No tools registered.")
            return
        for name, entry in entries:
            uri = f"{config.image_prefix}{entry.image_tag}"
            _pull_with_status(manager, name, uri)
        return

    # Single tool
    try:
        entry = get_tool(tool)  # type: ignore[arg-type]
    except KeyError as exc:
        print_error(str(exc))
        raise typer.Exit(code=1) from None

    uri = f"{config.image_prefix}{entry.image_tag}"
    _pull_with_status(manager, tool, uri)  # type: ignore[arg-type]


def images_cmd(
    fmt: Annotated[
        OutputFormat,
        typer.Option("--format", "-f", help="Output format."),
    ] = OutputFormat.TABLE,
) -> None:
    """List locally cached autobio container images."""
    config = AutobioConfig.resolve()
    manager = ContainerManager(config)
    images = manager.list_images(config.image_prefix)
    typer.echo(format_image_list(images, fmt))


def _pull_with_status(manager: ContainerManager, tool_name: str, uri: str) -> None:
    """Pull an image with console status feedback."""
    with _console.status(f"Pulling {tool_name} ({uri})..."):
        try:
            manager.pull_image(uri)
        except ContainerNotFoundError as exc:
            print_error(str(exc))
            raise typer.Exit(code=1) from None
    _console.print(f"  [green]Pulled[/green] {tool_name} ({uri})")
