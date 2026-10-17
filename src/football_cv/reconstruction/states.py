"""Player-state and position-provenance vocabularies (stored as plain strings in parquet/CSV).

A state says *why a player has no observation*; it is assigned only when its evidence
requirement is met (see ``docs/implementation_plan.md``). A source says *how a published
coordinate was obtained*. The two are independent: an OCCLUDED player may carry an
INTERPOLATED position, or none.
"""
from __future__ import annotations

from enum import Enum


class PlayerState(str, Enum):
    VISIBLE_ON_PITCH = "VISIBLE_ON_PITCH"
    OCCLUDED = "OCCLUDED"
    OUTSIDE_CAMERA_VIEW = "OUTSIDE_CAMERA_VIEW"
    OFF_PITCH_INJURY_TREATMENT = "OFF_PITCH_INJURY_TREATMENT"
    RETURNING_TO_PLAY = "RETURNING_TO_PLAY"
    SUBSTITUTED_CONFIRMED = "SUBSTITUTED_CONFIRMED"
    UNKNOWN_MISSING = "UNKNOWN_MISSING"


class PositionSource(str, Enum):
    OBSERVED = "OBSERVED"
    INTERPOLATED = "INTERPOLATED"
    PREDICTED = "PREDICTED"
    MANUALLY_ANNOTATED = "MANUALLY_ANNOTATED"


# These can never be inferred from a gap: they need a human annotation (or an external event feed).
ANNOTATION_ONLY_STATES = frozenset(
    {
        PlayerState.OFF_PITCH_INJURY_TREATMENT,
        PlayerState.RETURNING_TO_PLAY,
        PlayerState.SUBSTITUTED_CONFIRMED,
    }
)
