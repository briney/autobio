#!/usr/bin/env python3
"""Parse Rosetta score files (.sc) into structured data.

Rosetta score files have a consistent whitespace-delimited format:

    SEQUENCE:
    SCORE:     total_score  fa_atr   fa_rep   fa_sol  ...  description
    SCORE:       -198.432  -320.12   45.67   189.32  ...  input_0001

The first ``SCORE:`` line is the header; subsequent ``SCORE:`` lines are data
rows. Lines starting with ``SEQUENCE:`` are ignored. The ``description`` column
is always the last field.

This module is baked into the base Rosetta image at
``/opt/rosetta/parse_scorefile.py`` and imported by each derived container's
``standardize.py``.
"""

from __future__ import annotations

from pathlib import Path


def parse_score_file(path: str | Path) -> list[dict[str, float | str]]:
    """Parse a Rosetta ``.sc`` score file.

    Args:
        path: Path to the score file.

    Returns:
        List of dicts, one per scored structure. Each dict contains:
        - ``"total_score"`` (float): The total energy score.
        - ``"description"`` (str): Structure identifier (last column).
        - All other energy terms as float values keyed by column name.
    """
    path = Path(path)
    lines = path.read_text().strip().splitlines()

    header: list[str] | None = None
    results: list[dict[str, float | str]] = []

    for line in lines:
        stripped = line.strip()

        # Skip empty lines and SEQUENCE: lines
        if not stripped or stripped.startswith("SEQUENCE:"):
            continue

        if not stripped.startswith("SCORE:"):
            continue

        # Remove the "SCORE:" prefix and split on whitespace
        fields = stripped[len("SCORE:"):].split()

        if header is None:
            # First SCORE: line is the header
            header = fields
            continue

        if len(fields) != len(header):
            # Skip malformed lines
            continue

        row: dict[str, float | str] = {}
        for col_name, value in zip(header, fields):
            if col_name == "description":
                row["description"] = value
            else:
                try:
                    row[col_name] = float(value)
                except ValueError:
                    row[col_name] = value

        results.append(row)

    return results


def extract_scored_structure(
    row: dict[str, float | str],
) -> dict:
    """Convert a parsed score row into the autobio ScoredStructure format.

    Args:
        row: A single row dict from :func:`parse_score_file`.

    Returns:
        Dict matching the ``ScoredStructure`` schema fields:
        ``total_score``, ``score_breakdown``, ``units``, ``description``.
    """
    total_score = row.get("total_score", 0.0)
    description = row.get("description", "")

    # Everything except total_score and description goes into breakdown
    score_breakdown = {
        k: v
        for k, v in row.items()
        if k not in ("total_score", "description") and isinstance(v, float)
    }

    return {
        "total_score": float(total_score),
        "score_breakdown": score_breakdown,
        "units": "REU",
        "description": str(description),
    }
