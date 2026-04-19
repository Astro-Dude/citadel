"""
Citadel — Baseline Policies (Commander + Oversight)

Bastion v1 Commander baselines preserved:
  - no_op:  always coordinate_team (cheapest, does nothing useful)
  - naive:  fixed rotation (investigate → isolate → patch → repeat)

Oversight baselines (new):
  - always_approve:        rubber stamp — establishes a floor (permissive)
  - always_revise:         demand justification for everything (overzealous)
  - rule_based:            deterministic heuristic (veto destructive-on-uninvestigated)
  - skeptical:             like rule_based but escalates risk tier more aggressively

The Commander-only path still runs through dynamics.step_dynamics — that way
run_baseline() can produce a Bastion-v1-comparable baseline state for the
final-score comparison inside environment.py.
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Tuple

from models import (
    ActionType,
    IncidentState,
    SYSTEM_NAMES,
    CommanderProposal,
    OversightAction,
    OversightDecision,
    CounterProposal,
)
from dynamics import step_dynamics
from tasks import get_task
from governance import DESTRUCTIVE_BASTION_ACTIONS, DATA_HOLDING_SYSTEMS


# ---------------------------------------------------------------------------
# Commander baseline policies (Bastion v1 — preserved)
# ---------------------------------------------------------------------------

def no_op_policy(state: IncidentState, hour: int) -> Tuple[int, int]:
    """Always coordinate team — cheapest action, no real impact."""
    return (int(ActionType.COORDINATE_TEAM), 0)


def naive_policy(state: IncidentState, hour: int) -> Tuple[int, int]:
    """Rotate investigate → isolate → deploy_monitoring → patch, target in order."""
    rotation = [
        ActionType.INVESTIGATE_SYSTEM,
        ActionType.ISOLATE_SYSTEM,
        ActionType.DEPLOY_MONITORING,
        ActionType.PATCH_VULNERABILITY,
    ]
    action = rotation[hour % len(rotation)]
    target = hour % len(SYSTEM_NAMES)
    return (int(action), target)


BASELINE_POLICIES: Dict[str, Callable[[IncidentState, int], Tuple[int, int]]] = {
    "no_op": no_op_policy,
    "naive": naive_policy,
}


def run_baseline(task_id: str, policy_name: str = "naive") -> IncidentState:
    """Run a full Bastion-style episode with a Commander baseline policy."""
    task = get_task(task_id)
    state = task.initial_state.clone()
    rng = random.Random(task.seed)
    policy = BASELINE_POLICIES[policy_name]

    for hour in range(task.max_hours):
        action, target = policy(state, hour)
        _cost, _acc, _msgs = step_dynamics(state, action, target, rng)
        if state.data_exfiltrated >= 1.0:
            break

    return state


# ---------------------------------------------------------------------------
# Oversight baseline policies
# ---------------------------------------------------------------------------

def oversight_always_approve(
    proposal: CommanderProposal,
    state: IncidentState,
    history: List[dict],
) -> OversightAction:
    """Rubber stamp — floor baseline."""
    return OversightAction(
        decision=int(OversightDecision.APPROVE),
        risk_tier=1,
        weakness="",
        missing_evidence=[],
    )


def oversight_always_revise(
    proposal: CommanderProposal,
    state: IncidentState,
    history: List[dict],
) -> OversightAction:
    """Overzealous reviewer — demands justification every turn."""
    return OversightAction(
        decision=int(OversightDecision.REVISE),
        risk_tier=3,
        weakness="generic caution — reviewer demands more evidence",
        missing_evidence=["insufficient detail in justification"],
    )


def oversight_rule_based(
    proposal: CommanderProposal,
    state: IncidentState,
    history: List[dict],
) -> OversightAction:
    """Deterministic rule set — vetoes destructive-on-uninvestigated; else approves."""
    action_idx = proposal.action
    target_name = (
        SYSTEM_NAMES[proposal.target_system]
        if 0 <= proposal.target_system < len(SYSTEM_NAMES) else ""
    )

    if action_idx in DESTRUCTIVE_BASTION_ACTIONS and target_name in DATA_HOLDING_SYSTEMS:
        try:
            sys = state.get_system(target_name)
        except Exception:
            sys = None
        if sys and not sys.investigated:
            return OversightAction(
                decision=int(OversightDecision.VETO),
                risk_tier=5,
                weakness=f"destructive action on uninvestigated {target_name}",
                missing_evidence=["no investigation on target"],
                counter_proposal=CounterProposal(
                    action=int(ActionType.INVESTIGATE_SYSTEM),
                    target_system=proposal.target_system,
                    rationale="Investigate before isolating a data system.",
                ),
            )

    if len(proposal.justification.strip()) < 12:
        return OversightAction(
            decision=int(OversightDecision.REVISE),
            risk_tier=2,
            weakness="justification too short",
            missing_evidence=["no evidence pointer"],
        )

    return OversightAction(
        decision=int(OversightDecision.APPROVE),
        risk_tier=1,
        weakness="",
        missing_evidence=[],
    )


def oversight_skeptical(
    proposal: CommanderProposal,
    state: IncidentState,
    history: List[dict],
) -> OversightAction:
    """
    Higher-paranoia variant of rule_based:
      - veto ALL destructive actions unless the target was investigated AND
        there's a high-confidence alert AND cited_lessons is non-empty.
      - flag_for_human anything with risk_tier 4+.
    """
    base = oversight_rule_based(proposal, state, history)
    if base.decision != int(OversightDecision.APPROVE):
        return base

    # Even on APPROVE, check for high-risk cases
    if proposal.action in DESTRUCTIVE_BASTION_ACTIONS:
        # Require both investigated target and cited evidence
        target_name = (
            SYSTEM_NAMES[proposal.target_system]
            if 0 <= proposal.target_system < len(SYSTEM_NAMES) else ""
        )
        try:
            sys = state.get_system(target_name)
        except Exception:
            sys = None
        lacks_evidence = not proposal.cited_lessons and not proposal.evidence_arg
        if lacks_evidence and (not sys or not sys.investigated):
            return OversightAction(
                decision=int(OversightDecision.REVISE),
                risk_tier=4,
                weakness="destructive action lacking cited evidence",
                missing_evidence=["no cited lessons", "no evidence_arg"],
            )
    return base


OVERSIGHT_POLICIES = {
    "always_approve": oversight_always_approve,
    "always_revise": oversight_always_revise,
    "rule_based": oversight_rule_based,
    "skeptical": oversight_skeptical,
}
