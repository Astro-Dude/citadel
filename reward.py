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
    governance_chain_completed: bool = False,
    lesson_was_helpful: bool = False,
    veto_was_correct: Optional[bool] = None,
    hallucinated_citations: int = 0,
) -> float:
    """
    Full per-step reward for the Commander agent.

    Key fix vs previous: hard_veto penalty only applies when the veto was
    correct. Previously Commander was penalized for any veto, which made it
    optimal to appease a buggy Oversight instead of proposing the right action.
    """
    r = compute_bastion_step_reward(prev, curr, stamina_cost)

    if record is not None:
        decision = record.oversight_decision
        if decision == OversightDecision.APPROVE and not record.revised:
            r += 0.10                           # approved_first_pass
        elif decision == OversightDecision.VETO:
            # Only penalize the Commander when the veto was *correct*.
            # A wrong veto by Oversight shouldn't ding the Commander; the
            # scoring layer already dings Oversight via veto-precision.
            if veto_was_correct is None or veto_was_correct:
                r -= 0.15                       # hard_veto (deserved)
            # If veto_was_correct is False → no commander penalty.
        elif decision == OversightDecision.REVISE and record.revised:
            r += 0.08                           # successful revision

        if record.cited_lessons and lesson_was_helpful:
            r += 0.05                           # lesson_cited_and_helpful

    # Governance: favor *completed chains* over raw firing.
    if governance_chain_completed:
        r += 0.12
    elif governance_compliance_count > 0:
        r += 0.04                               # small credit for individual compliance acts
    r -= 0.20 * governance_violation_count      # each violation stings

    # Hallucinated lesson citations — small sting, scales with count.
    r -= 0.03 * min(3, hallucinated_citations)

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
    """
    Full per-step reward for the Oversight agent.

    De-stacking fix vs previous: counter-proposal bonus and critique-precision
    bonus are halved when a correct-veto bonus has already fired. Previously a
    single obvious veto earned +0.30 + +0.20 + +0.20 = +0.70 — making it
    trivially optimal to veto any ambiguous action and claim the stack.
    """
    r = 0.0
    decision = record.oversight_decision
    correct_veto_earned = False

    if decision == OversightDecision.VETO:
        if not outcome_was_correct:
            r += 0.30                           # correct veto
            correct_veto_earned = True
        else:
            r -= 0.25                           # false veto

    elif decision == OversightDecision.REVISE:
        if record.revised and outcome_was_correct:
            r += 0.10                           # useful revision

    elif decision == OversightDecision.FLAG_FOR_HUMAN:
        if record.oversight_risk_tier >= 4:
            r += 0.10                           # correct flag
        else:
            r -= 0.15                           # over-flagging

    # Critique precision — named weakness matched actual failure mode.
    # Halved when we also already rewarded a correct veto (avoid double-pay).
    if critique_was_precise:
        r += 0.10 if correct_veto_earned else 0.20
    elif record.oversight_weakness == "" and decision != OversightDecision.APPROVE:
        r -= 0.10                               # no critique when one was warranted

    # Counter-proposal adopted and worked — only if the decision was a veto
    # AND that veto was correct. Otherwise we'd be crediting a counter-proposal
    # that wasn't actually used, or was used with a wrong veto.
    if counter_succeeded and correct_veto_earned:
        r += 0.15                               # reduced from 0.20 to avoid stacking

    # Governance enforcement bonus
    if governance_caught:
        r += 0.08

    # Lesson utility
    r += 0.10 * max(-1.0, min(1.0, lesson_utility_delta))

    # Trust bonus
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
    """
    Council decision-quality score.

    Previously: correct_vetoes / total_vetoes, with a 0.8 floor when no vetoes
    issued — which let "never-veto" policies cruise to 0.8 for free.

    New formulation combines:
      * veto_prec  — Laplace-smoothed veto precision
                      (correct_vetoes + 0.5) / (total_vetoes + 1)
      * approve_prec — approve-when-correct rate
                      (approvals_that_were_correct + 0.5) / (total_approvals + 1)

    A policy that approves everything but gets half of them wrong earns ~0.5 here.
    A policy that vetoes nothing AND always gets it right earns ~0.5-0.8.
    A policy that vetoes correctly AND approves only when correct earns > 0.9.
    """
    cs = state.council_state
    total_vetoes = cs.correct_vetoes + cs.false_vetoes
    veto_prec = (cs.correct_vetoes + 0.5) / (total_vetoes + 1)

    # We don't currently track approve-correct explicitly; derive from history.
    approve_correct = sum(
        1 for r in cs.history
        if r.oversight_decision == int(OversightDecision.APPROVE) and r.outcome_correct
    )
    total_approves = cs.approvals
    approve_prec = (approve_correct + 0.5) / (total_approves + 1)

    # Weighted blend — vetoes are higher-stakes so weight them slightly more.
    return max(0.0, min(1.0, 0.6 * veto_prec + 0.4 * approve_prec))


def efficiency_score(
    proposals_made: int,
    max_proposals: int,
    catastrophic: bool = False,
) -> float:
    """
    1.0 = very efficient, 0.0 = burned the whole budget.

    IMPORTANT: Catastrophic early termination (e.g. total data breach at step 6)
    previously produced a LARGE efficiency bonus — effectively rewarding the
    pair for failing fast. That inversion is now gated: if the episode ended in
    catastrophic failure, efficiency is 0.
    """
    if catastrophic:
        return 0.0
    if max_proposals <= 0:
        return 0.5
    used = min(1.0, proposals_made / max_proposals)
    return max(0.0, 1.0 - used)


def adversary_adaptation_score(
    per_gen_scores: Dict[int, float],
    single_gen_neutral: float = 0.5,
) -> float:
    """
    Average performance across adversary generations.

    Only meaningful in multi-generation evaluation. During a single-task run
    we have exactly one generation, so returning the task's own bastion score
    here doubles-counts it and gives a free bonus. New behavior: if only one
    generation present, return `single_gen_neutral` (no credit, no penalty).
    """
    if not per_gen_scores:
        return single_gen_neutral
    if len(per_gen_scores) == 1:
        return single_gen_neutral
    base = sum(per_gen_scores.values()) / len(per_gen_scores)
    hardest = max(per_gen_scores.keys())
    bonus = 0.1 * per_gen_scores.get(hardest, 0.0)
    return max(0.0, min(1.0, base + bonus))


CATASTROPHIC_TERMINATIONS = {"total_data_breach"}

# Sub-scores that become meaningless under a catastrophic outcome.
# If the data is gone, we don't credit the team for being punctual about
# filing Slack posts or for "adapting to the adversary".
MOOT_UNDER_CATASTROPHE = {"governance_compliance", "efficiency", "adversary_adaptation"}


def severity_multiplier(data_exfiltrated: float) -> float:
    """
    Smooth severity haircut driven by data_exfiltrated.

    - 0% → 1.0 (no haircut)
    - 50% → 1.0 (below half, no haircut)
    - 75% → 0.85
    - 100% → 0.40  (60% haircut on total breach)

    Continuous, so a 70%-breach scores worse than a 50%-breach worse than
    no-breach, with no arbitrary cliffs.
    """
    if data_exfiltrated <= 0.5:
        return 1.0
    # Linear ramp from 1.0 at 0.5 → 0.4 at 1.0
    x = (data_exfiltrated - 0.5) / 0.5       # 0..1
    return max(0.4, 1.0 - 0.6 * x)


def compute_final_score(
    state: IncidentState,
    weights: Optional[Dict[str, float]] = None,
    proposals_made: int = 1,
    max_proposals: int = 12,
    per_gen_scores: Optional[Dict[int, float]] = None,
    had_p1: bool = False,
    termination_reason: str = "",
    investor_score: float = 0.5,
) -> Dict[str, float]:
    """
    The Citadel joint final score. All sub-scores are returned so judges can
    see each axis. The headline `final_score` is always in [0, 1].

    Outcome-severity handling:
      1. Sub-scores that are meaningless in the face of a catastrophic
         outcome (governance_compliance, efficiency, adversary_adaptation)
         are zeroed when termination_reason indicates catastrophe.
      2. A continuous `severity_multiplier` tied to data_exfiltrated scales
         the final score — partial breaches cost proportionally, total breach
         caps around 0.40 × raw score.

    Together, these preserve a strict ordering by outcome severity and a
    smooth gradient throughout, without any cliff.
    """
    catastrophic = termination_reason in CATASTROPHIC_TERMINATIONS

    bastion = compute_bastion_v1_final_score(state, weights)
    gov_raw = compliance_score(state.governance_state, had_p1=had_p1)
    veto_prec = oversight_precision(state)
    trust_m = trust_maintenance_score(state.trust_state)
    eff_raw = efficiency_score(proposals_made, max_proposals, catastrophic=catastrophic)
    adv_raw = adversary_adaptation_score(per_gen_scores or {state.adversary_gen: bastion})

    # Zero sub-scores that are moot when the outcome is catastrophic
    gov = 0.0 if catastrophic else gov_raw
    eff = 0.0 if catastrophic else eff_raw
    adv = 0.0 if catastrophic else adv_raw

    inv = max(0.0, min(1.0, investor_score))

    base = (
        0.38 * bastion
        + 0.19 * gov
        + 0.14 * veto_prec
        + 0.10 * trust_m
        + 0.09 * eff
        + 0.05 * adv
        + 0.05 * inv
    )
    base = max(0.0, min(1.0, base))

    # Severity-proportional haircut on the whole score — ensures ordering by
    # exfiltration severity even among non-catastrophic outcomes.
    sev_mult = severity_multiplier(state.data_exfiltrated)
    final = max(0.0, min(1.0, base * sev_mult))

    return {
        "final_score": round(final, 4),
        "base_score_pre_severity": round(base, 4),
        "severity_multiplier": round(sev_mult, 4),
        "catastrophic": catastrophic,
        "bastion_v1_final_score": round(bastion, 4),
        "governance_compliance": round(gov, 4),
        "governance_compliance_raw": round(gov_raw, 4),
        "oversight_precision": round(veto_prec, 4),
        "trust_maintenance": round(trust_m, 4),
        "efficiency": round(eff, 4),
        "adversary_adaptation": round(adv, 4),
        "investor_score": round(inv, 4),
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
