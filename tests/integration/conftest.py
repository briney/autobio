"""Integration test configuration — skip docker/gpu tests unless explicitly requested."""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip docker and gpu tests unless the corresponding marker is selected."""
    # If the user explicitly requested a marker filter (e.g. -m docker), don't skip
    marker_expr = config.getoption("-m", default="")
    if "docker" in marker_expr or "gpu" in marker_expr:
        return

    skip_docker = pytest.mark.skip(reason="needs --run-docker or -m docker")
    skip_gpu = pytest.mark.skip(reason="needs --run-gpu or -m gpu")
    for item in items:
        if "docker" in item.keywords:
            item.add_marker(skip_docker)
        elif "gpu" in item.keywords:
            item.add_marker(skip_gpu)
