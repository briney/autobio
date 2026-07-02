"""Tests for the x-autobio UI-hint helper."""

from __future__ import annotations

from pydantic import BaseModel, Field

from autobio.schemas.hints import Tier, Widget, ui


def test_ui_emits_only_provided_keys_under_namespace():
    assert ui(tier=Tier.PRIMARY, widget=Widget.SELECT) == {
        "x-autobio": {"tier": "primary", "widget": "select"}
    }


def test_ui_coerces_enums_and_passes_scalars():
    hint = ui(tier=Tier.ADVANCED, order=2, unit="Å", step=0.5, flavor="antibody")
    assert hint == {
        "x-autobio": {
            "tier": "advanced",
            "order": 2,
            "unit": "Å",
            "step": 0.5,
            "flavor": "antibody",
        }
    }


def test_ui_omits_none_values():
    assert ui(widget=Widget.TOGGLE) == {"x-autobio": {"widget": "toggle"}}


def test_hint_surfaces_in_model_json_schema():
    class M(BaseModel):
        per_position: bool = Field(
            default=False, json_schema_extra=ui(tier=Tier.PRIMARY, widget=Widget.TOGGLE)
        )

    schema = M.model_json_schema()
    assert schema["properties"]["per_position"]["x-autobio"] == {
        "tier": "primary",
        "widget": "toggle",
    }
