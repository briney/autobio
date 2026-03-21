"""Autobio CLI entry point."""

from __future__ import annotations

import typer

from autobio.cli.images import images_cmd, pull_cmd
from autobio.cli.info import info_cmd
from autobio.cli.list import list_tools_cmd
from autobio.cli.result import result_cmd
from autobio.cli.run import run_cmd

app = typer.Typer(
    name="autobio",
    help="Unified CLI for computational biology tools.",
    no_args_is_help=True,
)

app.command("list")(list_tools_cmd)
app.command("info")(info_cmd)
app.command("run")(run_cmd)
app.command("result")(result_cmd)
app.command("pull")(pull_cmd)
app.command("images")(images_cmd)
