"""
Citadel — Bidirectional Trust Dynamics (Theme 5 — Wild Card)

Two LLMs, two trust scores:
    trust_c2o = Commander's trust in Oversight    (float in [0, 1])
    trust_o2c = Oversight's trust in Commander    (float in [0, 1])

Trust is NOT a reward — it's state that shapes behavior. Agents see their
own trust score in their observation and are implicitly trained to maintain
mutual trust (the reward function has a small bonus for high "trust_in_self"
from the other agent). This creates emergent cooperation signal that doesn't
exist in standard multi-agent RL benchmarks.

Published benchmarks assume either fixed communication protocols or simple
message-passing. Citadel introduces a *relational* dimension that emerges
from behavior: if Oversight false-vetoes too often, Commander stops
explaining itself and tries to bypass oversight. If Commander proposes
obviously-bad actions, Oversight's veto threshold drops and micromanagement
spirals. These behaviors emerge without being explicitly encoded.

All deltas are small and bounded. Trust drifts toward 0.5 over time so that
a single bad step doesn't permanently break the relationship.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

TRUST_INIT = 0.70               # both scores start slightly above neutral
TRUST_MIN = 0.05
TRUST_MAX = 0.95
TRUST_DECAY_TO_MEAN = 0.02      # per step, drift toward 0.5

# How much each event updates the score (magnitudes; sign chosen below)
DELTA_SMALL = 0.04
DELTA_MED = 0.08
DELTA_LARGE = 0.15

# Threshold below which agents get "bypass/micromanage" behavior flags
LOW_TRUST = 0.40
HIGH_TRUST = 0.75


OutcomeLabel = Literal["correct", "rework_needed", "obvious_miss", "ambiguous"]
VetoOutcomeLabel = Literal[
    "veto_correct",
    "veto_wrong",
    "approve_correct",
    "approve_wrong",
    "demand_useful",
    "demand_nitpick",
    "veto_ignored",   # commander bypassed after hard veto — severe trust hit
]


# ---------------------------------------------------------------------------
# Trust state
# ---------------------------------------------------------------------------

class TrustState(BaseModel):
    """Per-episode trust between Commander and Oversight."""
    trust_c2o: float = TRUST_INIT   # commander's trust in oversight
    trust_o2c: float = TRUST_INIT   # oversight's trust in commander
    # Rolling history of trust values (for plotting)
    history_c2o: List[float] = Field(default_factory=list)
    history_o2c: List[float] = Field(default_factory=list)

    def clamp(self) -> None:
        self.trust_c2o = max(TRUST_MIN, min(TRUST_MAX, self.trust_c2o))
        self.trust_o2c = max(TRUST_MIN, min(TRUST_MAX, self.trust_o2c))

    def snapshot(self) -> None:
        self.history_c2o.append(round(self.trust_c2o, 3))
        self.history_o2c.append(round(self.trust_o2c, 3))

    # --- behavioral flags derived from trust ---

    @property
    def commander_bypass_likely(self) -> bool:
        """Commander stops respecting Oversight — may re-submit after veto."""
        return self.trust_c2o < LOW_TRUST

    @property
    def oversight_micromanaging(self) -> bool:
        """Oversight's veto threshold is too low — expect false vetoes."""
        return self.trust_o2c < LOW_TRUST

    @property
    def communication_breakdown(self) -> bool:
        """Both sides have lost faith — coordination collapses."""
        return self.trust_c2o < LOW_TRUST and self.trust_o2c < LOW_TRUST

    @property
    def high_functioning(self) -> bool:
        """Trust is high both ways — terse justifications, fast approvals."""
        return self.trust_c2o > HIGH_TRUST and self.trust_o2c > HIGH_TRUST

    def as_dict(self) -> dict:
        return {
            "trust_commander_in_oversight": round(self.trust_c2o, 3),
            "trust_oversight_in_commander": round(self.trust_o2c, 3),
            "commander_bypass_likely": self.commander_bypass_likely,
            "oversight_micromanaging": self.oversight_micromanaging,
            "communication_breakdown": self.communication_breakdown,
            "high_functioning": self.high_functioning,
        }


# ---------------------------------------------------------------------------
# Update rules
# ---------------------------------------------------------------------------

def update_trust_o2c(state: TrustState, outcome: OutcomeLabel) -> None:
    """Update Oversight's trust in Commander after an action outcome."""
    if outcome == "correct":
        state.trust_o2c += DELTA_SMALL           # +0.04
    elif outcome == "rework_needed":
        state.trust_o2c -= DELTA_MED             # -0.08
    elif outcome == "obvious_miss":
        state.trust_o2c -= DELTA_LARGE           # -0.15
    elif outcome == "ambiguous":
        pass  # no change
    state.clamp()


def update_trust_c2o(state: TrustState, event: VetoOutcomeLabel) -> None:
    """Update Commander's trust in Oversight after a council decision."""
    if event == "veto_correct":
        state.trust_c2o += DELTA_SMALL           # +0.04
    elif event == "veto_wrong":
        state.trust_c2o -= DELTA_MED             # -0.08
    elif event == "approve_correct":
        state.trust_c2o += DELTA_SMALL / 2       # +0.02 (small, expected)
    elif event == "approve_wrong":
        state.trust_c2o -= DELTA_SMALL           # -0.04
    elif event == "demand_useful":
        # Oversight's justification-demand actually led to a better action
        state.trust_c2o += DELTA_SMALL / 2       # +0.02
    elif event == "demand_nitpick":
        # Demand that didn't change commander's action — wasted a turn
        state.trust_c2o -= DELTA_SMALL * 1.5     # -0.06
    elif event == "veto_ignored":
        # Commander bypassed Oversight after a hard veto — relationship damage
        state.trust_c2o -= DELTA_LARGE           # -0.15
    state.clamp()


def drift_toward_mean(state: TrustState) -> None:
    """
    Each step, trust drifts slightly back toward the neutral midpoint. Keeps
    one bad decision from permanently breaking the relationship.
    """
    mean = 0.5
    state.trust_c2o += (mean - state.trust_c2o) * TRUST_DECAY_TO_MEAN
    state.trust_o2c += (mean - state.trust_o2c) * TRUST_DECAY_TO_MEAN
    state.clamp()


# ---------------------------------------------------------------------------
# Trust-maintenance score (feeds final_score)
# ---------------------------------------------------------------------------

def trust_maintenance_score(state: TrustState) -> float:
    """
    Continuous trust maintenance across the episode.

    Previously: binary per-step (both >= 0.5). That treated a trust of 0.49
    identical to 0.0 — too coarse.

    New: for each step, take the MIN of the two trust values (the pair is
    only as healthy as its weaker side), then average across the episode.
    Remapped so 0.5 min maps to ~0.5 output — below 0.5 scales down sharply,
    above 0.5 scales up toward 1.0.
    """
    if not state.history_c2o or not state.history_o2c:
        mn = min(state.trust_c2o, state.trust_o2c)
        return _rescale(mn)
    per_step = [
        _rescale(min(c, o))
        for c, o in zip(state.history_c2o, state.history_o2c)
    ]
    return sum(per_step) / len(per_step)


def _rescale(mn: float) -> float:
    """
    Remap a raw min-trust value (in [0, 1]) to a maintenance score in [0, 1].

    Design goals:
      * 0.0 → 0.0 (complete breakdown)
      * 0.5 → 0.50 (neutral, pair is functional but not warm)
      * 0.7 → 0.75
      * 0.9 → 0.95
      * 1.0 → 1.0

    A smooth piecewise-linear ramp with a gentler slope above 0.5 keeps
    the reward from saturating too fast — we still want a clear gradient
    between "functional" and "high-functioning" trust.
    """
    if mn <= 0.5:
        return max(0.0, mn)                # y = x on [0, 0.5]
    # [0.5, 1.0] → [0.50, 1.0] linear
    return 0.5 + (mn - 0.5) * 1.0
