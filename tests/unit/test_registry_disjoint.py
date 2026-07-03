"""Guard: the catalog and the legacy flat registry must never share a name.

A tool name in both ``CATALOG`` and ``TOOL_REGISTRY`` would be listed twice by
``autobio list`` and makes runner/metadata lookup ambiguous — it signals a
half-finished migration (a flat entry that was not deleted). This test fails
loudly if that ever happens. It deliberately has no registry-clearing fixture:
it must observe the real, fully-populated registries.
"""

from __future__ import annotations


def test_catalog_and_flat_registry_are_disjoint() -> None:
    import autobio.tools  # noqa: F401 - importing populates both registries
    from autobio.core.catalog import CATALOG
    from autobio.core.registry import TOOL_REGISTRY

    overlap = set(CATALOG) & set(TOOL_REGISTRY)
    assert not overlap, f"Tools registered in both CATALOG and TOOL_REGISTRY: {sorted(overlap)}"
