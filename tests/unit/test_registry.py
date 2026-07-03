"""Tests for autobio.core.registry."""

from __future__ import annotations

from autobio.core.registry import ToolCategory

# ---------------------------------------------------------------------------
# ToolCategory
# ---------------------------------------------------------------------------


class TestToolCategory:
    def test_values(self) -> None:
        assert ToolCategory.STRUCTURE_PREDICTION == "structure-prediction"
        assert ToolCategory.EMBEDDING == "embedding"
        assert ToolCategory.INVERSE_FOLDING == "inverse-folding"
        assert ToolCategory.SCORING == "scoring"
        assert ToolCategory.STRUCTURE_DESIGN == "structure-design"
        assert ToolCategory.SIMULATION == "simulation"

    def test_is_str(self) -> None:
        for member in ToolCategory:
            assert isinstance(member, str)

    def test_member_count(self) -> None:
        assert len(ToolCategory) == 6
