"""Tests for containers/rosetta-base/parse_scorefile.py — Rosetta score file parsing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Import the shared parser from the container directory
_CONTAINER_DIR = str(Path(__file__).resolve().parent.parent.parent / "containers" / "rosetta-base")
if _CONTAINER_DIR not in sys.path:
    sys.path.insert(0, _CONTAINER_DIR)

from parse_scorefile import extract_scored_structure, parse_score_file  # noqa: E402

# ---------------------------------------------------------------------------
# Sample score file content
# ---------------------------------------------------------------------------

_SIMPLE_SCORE_FILE = """\
SEQUENCE:
SCORE:     total_score     fa_atr     fa_rep     fa_sol  description
SCORE:       -198.432    -320.120     45.670    189.320  input_0001
"""

_MULTI_STRUCTURE_SCORE_FILE = """\
SEQUENCE:
SCORE:     total_score     fa_atr     fa_rep     fa_sol  description
SCORE:       -198.432    -320.120     45.670    189.320  relaxed_0001
SCORE:       -205.100    -325.000     40.000    180.000  relaxed_0002
SCORE:       -201.780    -322.500     42.800    185.100  relaxed_0003
"""

_FULL_SCORE_FILE = (  # noqa: E501 — Rosetta score files have wide header lines
    "SEQUENCE:\n"
    "SCORE:     total_score     fa_atr     fa_rep     fa_sol     fa_intra_rep"
    "     fa_elec     pro_close     hbond_sr_bb     hbond_lr_bb     hbond_bb_sc"
    "     hbond_sc     dslf_fa13     omega     fa_dun     p_aa_pp"
    "     yhh_planarity     ref     rama_prepro     lk_ball_wtd  description\n"
    "SCORE:       -198.432    -320.120     45.670    189.320     1.230"
    "     -55.400     0.450     -12.300     -8.900     -5.600"
    "     -3.200     0.000     2.100     150.800     -22.500"
    "     0.030     -41.920     -7.550     -110.550  1ubq_0001\n"
)

_EMPTY_SCORE_FILE = """\
SEQUENCE:
SCORE:     total_score     fa_atr     fa_rep     fa_sol  description
"""

_SCORE_FILE_WITH_BLANK_LINES = """\

SEQUENCE:

SCORE:     total_score     fa_atr  description
SCORE:       -100.000     -80.000  test_0001

"""


# ---------------------------------------------------------------------------
# Tests: parse_score_file
# ---------------------------------------------------------------------------


class TestParseScoreFile:
    """Tests for the score file parser."""

    def test_simple_score_file(self, tmp_path: Path) -> None:
        """Parse a single-structure score file."""
        sc = tmp_path / "score.sc"
        sc.write_text(_SIMPLE_SCORE_FILE)

        results = parse_score_file(sc)
        assert len(results) == 1

        row = results[0]
        assert row["total_score"] == pytest.approx(-198.432)
        assert row["fa_atr"] == pytest.approx(-320.120)
        assert row["fa_rep"] == pytest.approx(45.670)
        assert row["fa_sol"] == pytest.approx(189.320)
        assert row["description"] == "input_0001"

    def test_multi_structure_score_file(self, tmp_path: Path) -> None:
        """Parse a score file with multiple structures."""
        sc = tmp_path / "score.sc"
        sc.write_text(_MULTI_STRUCTURE_SCORE_FILE)

        results = parse_score_file(sc)
        assert len(results) == 3
        assert results[0]["description"] == "relaxed_0001"
        assert results[1]["description"] == "relaxed_0002"
        assert results[2]["description"] == "relaxed_0003"
        assert results[1]["total_score"] == pytest.approx(-205.100)

    def test_full_energy_terms(self, tmp_path: Path) -> None:
        """Parse a score file with all standard Rosetta energy terms."""
        sc = tmp_path / "score.sc"
        sc.write_text(_FULL_SCORE_FILE)

        results = parse_score_file(sc)
        assert len(results) == 1
        row = results[0]
        assert row["hbond_sr_bb"] == pytest.approx(-12.300)
        assert row["lk_ball_wtd"] == pytest.approx(-110.550)
        assert row["description"] == "1ubq_0001"

    def test_empty_score_file(self, tmp_path: Path) -> None:
        """Score file with header only produces no results."""
        sc = tmp_path / "score.sc"
        sc.write_text(_EMPTY_SCORE_FILE)

        results = parse_score_file(sc)
        assert results == []

    def test_blank_lines_handled(self, tmp_path: Path) -> None:
        """Score file with blank lines is parsed correctly."""
        sc = tmp_path / "score.sc"
        sc.write_text(_SCORE_FILE_WITH_BLANK_LINES)

        results = parse_score_file(sc)
        assert len(results) == 1
        assert results[0]["total_score"] == pytest.approx(-100.0)


# ---------------------------------------------------------------------------
# Tests: extract_scored_structure
# ---------------------------------------------------------------------------


class TestExtractScoredStructure:
    """Tests for converting parsed rows to autobio schema format."""

    def test_basic_extraction(self) -> None:
        row = {
            "total_score": -198.432,
            "fa_atr": -320.12,
            "fa_rep": 45.67,
            "description": "input_0001",
        }
        result = extract_scored_structure(row)

        assert result["total_score"] == pytest.approx(-198.432)
        assert result["units"] == "REU"
        assert result["description"] == "input_0001"
        assert result["score_breakdown"]["fa_atr"] == pytest.approx(-320.12)
        assert result["score_breakdown"]["fa_rep"] == pytest.approx(45.67)
        # total_score and description should not be in breakdown
        assert "total_score" not in result["score_breakdown"]
        assert "description" not in result["score_breakdown"]

    def test_missing_total_score_defaults_zero(self) -> None:
        row = {"fa_atr": -100.0, "description": "test"}
        result = extract_scored_structure(row)
        assert result["total_score"] == 0.0

    def test_end_to_end_with_parse(self, tmp_path: Path) -> None:
        """Full pipeline: parse score file then extract structured output."""
        sc = tmp_path / "score.sc"
        sc.write_text(_SIMPLE_SCORE_FILE)

        rows = parse_score_file(sc)
        results = [extract_scored_structure(r) for r in rows]

        assert len(results) == 1
        assert results[0]["total_score"] == pytest.approx(-198.432)
        assert results[0]["units"] == "REU"
        assert "fa_atr" in results[0]["score_breakdown"]
