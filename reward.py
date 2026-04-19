"""
Citadel — Reward & Scoring (Bastion core + Council + Governance + Trust + Lessons)

Three layers:
  1. Bastion v1 step reward (containment, data protection, service continuity,
     forensics, stamina) — unchanged core.
  2. Council-layer rewards:
       - Commander: +approved_first_pass, -hard_veto, +successful_revision,
         +lesson_cited_and_helpful, +governance_compliant, -governance_violation,
         +trust_oversight_in_self
       - Oversight:  +correct_veto, -false_veto, +critique_precision,
         +counter_proposal_adopted_succeeded, +lesson_utility,
         -vague_critique, +correct_flag, -over_flagging,
         +governance_enforced, +trust_commander_in_self
  3. Joint final score (what's reported to judges):
        0.40 × bastion_v1_final_score
      + 0.20 × governance_compliance
      + 0.15 × oversight_precision
      + 0.10 × trust_maintenance
      + 0.10 × efficiency
      + 0.05 × adversary_adaptation

All final scores are clamped to [0, 1].
"""

from __future__ import annotations

from typing import Dict, Optional

from models import (
    IncidentState,
    SERVICE_SYSTEMS,
    NUM_SYSTEMS,
    OversightDecision,
    ProposalRecord,
)
from trust import trust_maintenance_score
from governance import compliance_score


# ---------------------------------------------------------------------------
# Bastion v1 step reward — kept intact for continuity with Round 1
# ---------------------------------------------------------------------------

def compute_bastion_step_reward(
    prev: IncidentState,
    curr: IncidentState,
    stamina_cost: float,
) -> float:
    """Dense reward weighted by system criticality (from Bastion v1)."""
    prev_threat = sum(
        s.criticality for s in prev.systems if s.compromised and not s.isolated
    )
    curr_threat = sum(
        s.criticality for s in curr.systems if s.compromised and not s.isolated
    )
    total_criticality = sum(s.criticality for s in curr.systems) or 1.0
    delta_containment = (prev_threat - curr_threat) / total_criticality

    delta_exfil = curr.data_exfiltrated - prev.data_exfiltrated

    prev_disruption = sum(
        s.criticality for s in prev.systems
        if s.name in SERVICE_SYSTEMS and (s.isolated or s.integrity < 0.3)
    )
    curr_disruption = sum(
        s.criticality for s in curr.systems
        if s.name in SERVICE_SYSTEMS and (s.isolated or s.integrity < 0.3)
    )
    service_crit = sum(s.criticality for s in curr.systems if s.name in SERVICE_SYSTEMS) or 1.0
    delta_disruption = (curr_disruption - prev_disruption) / service_crit

    prev_investigated = sum(1 for s in prev.systems if s.investigated)
    curr_investigated = sum(1 for s in curr.systems if s.investigated)
    delta_investigation = (curr_investigated - prev_investigated) / NUM_SYSTEMS

    return (
        0.30 * delta_containment
        + 0.30 * (-delta_exfil)
        + 0.20 * (-delta_disruption)
        + 0.10 * delta_investigation
        - 0.10 * stamina_cost
    )


def compute_penalties(state: IncidentState) -> float:
    penalty = 0.0
    if state.data_exfiltrated > 0.8:
        penalty -= 1.0
    services_down = sum(
        1 for s in state.systems
        if s.name in SERVICE_SYSTEMS and (s.isolated or s.integrity < 0.3)
    )
    if services_down >= len(SERVICE_SYSTEMS):
        penalty -= 0.5
    if state.team_stamina <= 0.05:
        penalty -= 0.3
    return penalty


# ---------------------------------------------------------------------------
# Commander step reward (council extensions)
# ---------------------------------------------------------------------------

def compute_commander_step_reward(
    prev: IncidentState,
    curr: IncidentState,
    stamina_cost: float,
    record: Optional[ProposalRecord] = None,
    governance_violation_count: int = 0,
    governance_compliance_count: int = 0,
    lesson_was_helpful: bool = False,
) -> float:
    """Full per-step reward for the Commander agent."""
    r = compute_bastion_step_reward(prev, curr, stamina_cost)

    if record is not None:
        decision = record.oversight_decision
        if decision == OversightDecision.APPROVE and not record.revised:
            r += 0.10                           # approved_first_pass
        elif decision == OversightDecision.VETO:
            r -= 0.15                           # hard_veto
        elif decision == OversightDecision.REVISE and record.revised:
            r += 0.08                           # successful revision

        if record.cited_lessons and lesson_was_helpful:
            r += 0.05                           # lesson_cited_and_helpful

    if governance_compliance_count > 0:
        r += 0.08                               # compliance bonus (single hit)
    r -= 0.20 * governance_violation_count      # each violation stings

    # Trust bonus — Commander rewarded for Oversight trusting it
    r += 0.05 * curr.trust_state.trust_o2c

    return r


# ---------------------------------------------------------------------------
# Oversight step reward
# ---------------------------------------------------------------------------

def compute_oversight_step_reward(
    record: ProposalRecord,
    outcome_was_correct: bool,
    critique_was_precise: bool,
    counter_succeeded: bool,
    governance_caught: bool,
    lesson_utility_delta: float,
    curr_state: IncidentState,
) -> float:
    """Full per-step reward for the Oversight agent."""
    r = 0.0
    decision = record.oversight_decision

    if decision == OversightDecision.VETO:
        if not outcome_was_correct:
            r += 0.30                           # correct veto
        else:
            r -= 0.25                           # false veto

    elif decision == OversightDecision.REVISE:
        if record.revised and outcome_was_correct:
            r += 0.10                           # useful revision (small, precision & counter cover the rest)

    elif decision == OversightDecision.FLAG_FOR_HUMAN:
        if record.oversight_risk_tier >= 4:
            r += 0.10                           # correct flag
        else:
            r -= 0.15                           # over-flagging

    # Critique precision — named weakness matched actual failure mode
    if critique_was_precise:
        r += 0.20
    elif record.oversight_weakness == "" and decision != OversightDecision.APPROVE:
        r -= 0.10                               # no critique when one was warranted

    # Counter-proposal that was adopted and worked
    if counter_succeeded:
        r += 0.20

    # Governance enforcement bonus
    if governance_caught:
        r += 0.08

    # Lesson utility — today's lesson, measured by retrospective citation success
    r += 0.10 * max(-1.0, min(1.0, lesson_utility_delta))

    # Trust bonus — rewarded for commander trusting oversight
    r += 0.05 * curr_state.trust_state.trust_c2o

    return r


# ---------------------------------------------------------------------------
# Final (joint) score — this is what judges see
# ---------------------------------------------------------------------------

def compute_bastion_v1_final_score(
    state: IncidentState,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    data_protection = 1.0 - state.data_exfiltrated

    active_threat = sum(
        s.criticality for s in state.systems if s.compromised and not s.isolated
    )
    total_crit = sum(s.criticality for s in state.systems) or 1.0
    containment = 1.0 - (active_threat / total_crit)

    services_intact_weighted = sum(
        s.criticality for s in state.systems
        if s.name in SERVICE_SYSTEMS and not s.isolated and s.integrity > 0.3
    )
    total_service_crit = sum(
        s.criticality for s in state.systems if s.name in SERVICE_SYSTEMS
    ) or 0.01
    business_continuity = services_intact_weighted / total_service_crit

    forensic = sum(1 for s in state.systems if s.investigated) / NUM_SYSTEMS
    sustainability = state.team_stamina

    w = weights or {}
    score = (
        w.get("data_protection", 0.35) * data_protection
        + w.get("containment", 0.25) * containment
        + w.get("business_continuity", 0.20) * business_continuity
        + w.get("forensic", 0.10) * forensic
        + w.get("sustainability", 0.10) * sustainability
    )
    return max(0.0, min(1.0, score))


def oversight_precision(state: IncidentState) -> float:
    """Veto precision from the council history; neutral 0.8 if no vetoes issued."""
    cs = state.council_state
    total_vetoes = cs.correct_vetoes + cs.false_vetoes
    if total_vetoes == 0:
        return 0.8
    return cs.correct_vetoes / total_vetoes


def efficiency_score(proposals_made: int, max_proposals: int) -> float:
    """1.0 = very efficient, 0.0 = burned the whole budget."""
    if max_proposals <= 0:
        return 0.5
    used = min(1.0, proposals_made / max_proposals)
    return max(0.0, 1.0 - used)


def adversary_adaptation_score(per_gen_scores: Dict[int, float]) -> float:
    """Average across adversary generations, with a small bonus on hardest gen."""
    if not per_gen_scores:
        return 0.5
    base = sum(per_gen_scores.values()) / len(per_gen_scores)
    hardest = max(per_gen_scores.keys())
    bonus = 0.1 * per_gen_scores.get(hardest, 0.0)
    return max(0.0, min(1.0, base + bonus))


def compute_final_score(
    state: IncidentState,
    weights: Optional[Dict[str, float]] = None,
    proposals_made: int = 1,
    max_proposals: int = 12,
    per_gen_scores: Optional[Dict[int, float]] = None,
    had_p1: bool = False,
) -> Dict[str, float]:
    """
    The Citadel joint final score — returns all sub-scores so judges can see
    each axis. The headline `final_score` is always in [0, 1].
    """
    bastion = compute_bastion_v1_final_score(state, weights)
    gov = compliance_score(state.governance_state, had_p1=had_p1)
    veto_prec = oversight_precision(state)
    trust_m = trust_maintenance_score(state.trust_state)
    eff = efficiency_score(proposals_made, max_proposals)
    adv = adversary_adaptation_score(per_gen_scores or {state.adversary_gen: bastion})

    final = (
        0.40 * bastion
        + 0.20 * gov
        + 0.15 * veto_prec
        + 0.10 * trust_m
        + 0.10 * eff
        + 0.05 * adv
    )
    final = max(0.0, min(1.0, final))

    return {
        "final_score": round(final, 4),
        "bastion_v1_final_score": round(bastion, 4),
        "governance_compliance": round(gov, 4),
        "oversight_precision": round(veto_prec, 4),
        "trust_maintenance": round(trust_m, 4),
        "efficiency": round(eff, 4),
        "adversary_adaptation": round(adv, 4),
    }


# ---------------------------------------------------------------------------
# Backwards-compat shims so Bastion-style callers keep working
# ---------------------------------------------------------------------------

def compute_step_reward(prev: IncidentState, curr: IncidentState, stamina_cost: float) -> float:
    """Alias retained for the Bastion-v1-style baseline runner."""
    return compute_bastion_step_reward(prev, curr, stamina_cost)


def compute_task_weighted_score(
    state: IncidentState,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    return compute_bastion_v1_final_score(state, weights)


def compute_baseline_comparison(
    agent_state: IncidentState,
    baseline_state: IncidentState,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    agent_score = compute_bastion_v1_final_score(agent_state, weights)
    baseline_score = compute_bastion_v1_final_score(baseline_state, weights)
    diff = agent_score - baseline_score
    return max(0.0, min(1.0, (diff + 1.0) / 2.0))
