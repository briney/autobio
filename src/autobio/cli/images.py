"""`autobio images` and `autobio pull` — manage container images."""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

from autobio.cli.formatters import OutputFormat, format_image_list, print_error
from autobio.core import catalog
from autobio.core.config import AutobioConfig
from autobio.core.container import ContainerManager
from autobio.core.result import ContainerNotFoundError

_console = Console()


def _catalog_image_uris(tool: catalog.Tool, config: AutobioConfig) -> set[str]:
    """Return the prefixed image URIs a catalog Tool pulls from.

    This is the Tool's own ``image_tag`` plus any per-mode ``image_tag``
    overrides (used by engines whose modes ship as separate container images,
    e.g. future rosetta/openmm modes). ``freesasa``/``esm1b``/``esm2`` have no
    mode overrides, so this is just the Tool's single image.

    Args:
        tool: The catalog Tool to resolve image URIs for.
        config: Active autobio config (supplies the image prefix).

    Returns:
        A set of fully-prefixed image URIs (deduplicated).
    """
    tags = {tool.image_tag} | {m.image_tag for m in tool.modes.values() if m.image_tag}
    return {f"{config.image_prefix}{tag}" for tag in tags}


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
        if not catalog.CATALOG:
            typer.echo("No tools registered.")
            return
        # Map uri -> representative tool name, so shared images (e.g. esm1b
        # and esm2 both resolving to esm:1.0.0) are pulled exactly once.
        uris: dict[str, str] = {}
        for name, cat_tool in catalog.CATALOG.items():
            for uri in _catalog_image_uris(cat_tool, config):
                uris.setdefault(uri, name)
        for uri, name in uris.items():
            _pull_with_status(manager, name, uri)
        return

    # Single tool — argparse guarantees `tool` is a non-empty str here, since
    # the only way to reach this branch with `tool is None` would have exited
    # via the guard above (`not tool and not all_tools`).
    assert tool is not None
    if tool in catalog.CATALOG:
        cat_tool = catalog.get_tool(tool)
        for uri in sorted(_catalog_image_uris(cat_tool, config)):
            _pull_with_status(manager, tool, uri)
        return

    available = ", ".join(sorted(catalog.CATALOG)) or "(none)"
    print_error(f"Unknown tool {tool!r}. Available tools: {available}")
    raise typer.Exit(code=1)


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
