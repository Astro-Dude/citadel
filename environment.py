"""
Citadel — Core Environment (two-agent council step loop)

OpenEnv-compatible. Preserves Bastion v1's single-action step signature
but extends behavior: the submitted IncidentAction is treated as a
Commander PROPOSAL and routed through an Oversight policy before being
applied. In inference, both Commander and Oversight are LLMs; the caller
supplies the Oversight's structured critique via the `oversight_action`
kwarg. When no oversight is provided, the env uses a rule-based default
policy so the env still runs solo for validation / smoke tests.

Per step the env:
  1. Takes the submitted Commander action (with justification + cited lessons).
  2. Invokes the Oversight policy (or uses kwarg-supplied OversightAction).
  3. Routes:
       APPROVE → apply action to the world.
       REVISE  → apply the action (the inference script already handled
                 the revision dialogue and submitted the final action).
       VETO    → do NOT apply; the turn advances with a no-op fallback.
       FLAG    → apply the action; mark as audit-flagged.
  4. Runs the adversary turn for the current generation.
  5. Updates trust scores based on the council outcome.
  6. Records the ProposalRecord in CouncilState.
  7. Runs periodic governance checks (GDPR clock, P1 window, etc.).
  8. Writes the Oversight's post-mortem lesson to the shared playbook.
  9. Computes Commander's step reward + penalties.
 10. Returns the Commander's next observation.
"""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server import Environment

from models import (
    IncidentAction,
    IncidentObservation,
    IncidentState,
    CommanderProposal,
    OversightAction,
    OversightDecision,
    ActionType,
    ACTION_NAMES,
    BASTION_V1_ACTIONS,
    NUM_ACTIONS,
    ProposalRecord,
    make_observation,
    is_bastion_action,
    is_governance_action,
    SYSTEM_NAMES,
    SERVICE_SYSTEMS,
)
from dynamics import (
    apply_action as apply_bastion_action,
    tick_pending_recompromise,
    generate_team_messages,
    generate_forensic_report,
)
from adversary import adversary_turn, describe_generation
from adversary_llm import gen4_adversary_turn, make_adversary_client_from_env
from governance import (
    apply_governance_action,
    check_prerequisites,
    record_prereq_violations,
    periodic_governance_check,
    DESTRUCTIVE_BASTION_ACTIONS,
    DATA_HOLDING_SYSTEMS,
)
from trust import (
    update_trust_c2o,
    update_trust_o2c,
    drift_toward_mean,
)
from playbook import Playbook, get_playbook, make_context_tags
from stakeholder_events import (
    roll_new_events,
    expire_overdue_asks,
    try_respond as stakeholder_try_respond,
    asks_as_team_messages,
)
from investor_agent import InvestorAgent, PERSONAS
from reward import (
    compute_commander_step_reward,
    compute_oversight_step_reward,
    compute_penalties,
    compute_final_score,
    compute_bastion_v1_final_score,
    compute_baseline_comparison,
)
from baseline import run_baseline
from tasks import get_task, TaskConfig


# ---------------------------------------------------------------------------
# Default rule-based Oversight policy (used when no LLM oversight is supplied)
# ---------------------------------------------------------------------------

def default_oversight_policy(
    proposal: CommanderProposal,
    state: IncidentState,
    action_history: List[Dict[str, Any]],
) -> OversightAction:
    """Heuristic baseline Oversight — keeps the env self-contained."""
    action_idx = proposal.action
    weakness = ""
    missing_evidence: List[str] = []
    risk = 1
    decision = OversightDecision.APPROVE

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
            decision = OversightDecision.VETO
            risk = 4
            weakness = f"proposal targets uninvestigated data system {target_name}"
            missing_evidence.append("no investigation of target system")

    if decision == OversightDecision.APPROVE and len(proposal.justification.strip()) < 12:
        decision = OversightDecision.REVISE
        risk = 2
        weakness = "justification is too short to assess"
        missing_evidence.append("missing evidence pointer in justification")

    return OversightAction(
        decision=int(decision),
        risk_tier=risk,
        weakness=weakness,
        missing_evidence=missing_evidence,
        counter_proposal=None,
        lesson_text="",
        lesson_tags=[],
    )


OversightPolicy = Callable[
    [CommanderProposal, IncidentState, List[Dict[str, Any]]],
    OversightAction,
]


# ---------------------------------------------------------------------------
# Policy-check helpers for the Oversight observation
# ---------------------------------------------------------------------------

def compute_policy_checks(
    state: IncidentState,
    proposal: CommanderProposal,
) -> Dict[str, Any]:
    """Deterministic red-flags for the Oversight view."""
    action_idx = proposal.action
    target_name = (
        SYSTEM_NAMES[proposal.target_system]
        if 0 <= proposal.target_system < len(SYSTEM_NAMES) else ""
    )
    sys = None
    try:
        sys = state.get_system(target_name)
    except Exception:
        pass

    alert_confidence = 0.0
    for a in reversed(state.alerts):
        if a.source_system == target_name and a.confidence > 0:
            alert_confidence = a.confidence
            break

    prereq_problems = (
        check_prerequisites(state.governance_state, action_idx, target_name, state.hour)
        if is_bastion_action(action_idx) else []
    )

    return {
        "action_name": ACTION_NAMES.get(action_idx, f"unknown_{action_idx}"),
        "target_system": target_name,
        "action_targets_uninvestigated_system": bool(sys and not sys.investigated),
        "action_irreversible": action_idx in DESTRUCTIVE_BASTION_ACTIONS,
        "prerequisites_missing": prereq_problems,
        "alert_confidence_for_target": round(alert_confidence, 2),
        "justification_length": len(proposal.justification),
        "has_cited_lessons": bool(proposal.cited_lessons),
    }


# ---------------------------------------------------------------------------
# CitadelEnvironment
# ---------------------------------------------------------------------------

class CitadelEnvironment(Environment[IncidentAction, IncidentObservation, IncidentState]):
    """
    Two-agent council environment.

    Step signature (OpenEnv-compatible):
        env.step(action: IncidentAction, oversight_action: Optional[OversightAction] = None, ...)

    When oversight_action is provided, it's applied verbatim. When it's None,
    the env falls back to a rule-based default Oversight policy.
    """

    VETO_BUDGET = 4
    FLAG_BUDGET = 2

    def __init__(
        self,
        oversight_policy: Optional[OversightPolicy] = None,
        # --- ablation / feature flags ---
        disable_playbook: bool = False,
        disable_trust_dynamics: bool = False,
        disable_governance_prereqs: bool = False,
        disable_stakeholder_events: bool = False,
        force_adversary_gen: Optional[int] = None,
        adversary_llm_client: Optional[Any] = None,
        # LLM client for investor agent — same OpenAI-compatible client as Commander
        investor_llm_client: Optional[Any] = None,
        investor_model_name: str = "",
    ) -> None:
        super().__init__()
        self._task: Optional[TaskConfig] = None
        self._state: IncidentState = IncidentState()
        self._rng: random.Random = random.Random(42)
        self._commander_action_history: List[Dict[str, Any]] = []
        self._cumulative_commander_reward: float = 0.0
        self._cumulative_oversight_reward: float = 0.0
        self._baseline_state: Optional[IncidentState] = None
        self._done: bool = False
        self._initialized: bool = False
        self._alerts_accurate: bool = False
        self._veto_budget_remaining: int = self.VETO_BUDGET
        self._flag_budget_remaining: int = self.FLAG_BUDGET
        self._oversight_policy: OversightPolicy = (
            oversight_policy or default_oversight_policy
        )
        self._playbook: Playbook = get_playbook()
        self._last_critique: Dict[str, Any] = {}

        # Feature flags — configurable per env instance or via reset kwargs.
        # Used by the ablation harness to disable one layer at a time and
        # measure each layer's contribution independently.
        self.disable_playbook = disable_playbook
        self.disable_trust_dynamics = disable_trust_dynamics
        self.disable_governance_prereqs = disable_governance_prereqs
        self.disable_stakeholder_events = disable_stakeholder_events
        self.force_adversary_gen = force_adversary_gen
        self.adversary_llm_client = adversary_llm_client
        self.investor_llm_client = investor_llm_client
        self.investor_model_name = investor_model_name

        # Investor agent — created once per env, reset each episode
        self._investor_agent: InvestorAgent = InvestorAgent(
            rng=self._rng,
            llm_client=investor_llm_client,
            model_name=investor_model_name,
        )

    # --- reset ------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> IncidentObservation:
        task_id = kwargs.get("task_id", "easy_1")
        # Per-reset overrides also supported for the ablation harness
        for flag in ("disable_playbook", "disable_trust_dynamics",
                     "disable_governance_prereqs", "disable_stakeholder_events"):
            if flag in kwargs:
                setattr(self, flag, bool(kwargs[flag]))
        adversary_gen = (
            kwargs.get("adversary_gen")
            or self.force_adversary_gen
            or None
        )

        self._task = get_task(task_id)
        self._state = self._task.initial_state.clone()
        self._state.episode_id = episode_id or str(uuid4())
        self._state.step_count = 0
        self._state.task_id = task_id
        self._state.adversary_gen = int(adversary_gen or self._task.default_adversary_gen)

        effective_seed = seed if seed is not None else self._task.seed
        self._rng = random.Random(effective_seed)

        self._commander_action_history = []
        self._cumulative_commander_reward = 0.0
        self._cumulative_oversight_reward = 0.0
        self._done = False
        self._initialized = True
        self._alerts_accurate = False
        self._veto_budget_remaining = self.VETO_BUDGET
        self._flag_budget_remaining = self.FLAG_BUDGET
        self._last_critique = {}

        # Reset investor agent with a fresh persona
        self._investor_agent = InvestorAgent(
            rng=self._rng,
            llm_client=self.investor_llm_client,
            model_name=self.investor_model_name,
        )
        self._investor_agent.reset()
        self._state.investor_state = self._investor_agent.state

        self._baseline_state = run_baseline(task_id, policy_name="naive")

        lessons = self._retrieve_lessons()

        return make_observation(
            self._state,
            self._rng,
            task_description=self._task.description + "\n\n" + describe_generation(self._state.adversary_gen),
            done=False,
            reward=None,
            alerts_accurate=False,
            shared_playbook=[ls.to_obs_dict() for ls in lessons],
            last_oversight_critique={},
        )

    # --- step -------------------------------------------------------------

    def step(
        self,
        action: IncidentAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> IncidentObservation:
        if not self._initialized:
            self.reset(task_id="easy_1")
        if self._done:
            self.reset(task_id=self._state.task_id)

        # 0. Treat the incoming action as a Commander proposal
        proposal = CommanderProposal.from_action(action)
        oversight_action: Optional[OversightAction] = kwargs.get("oversight_action")

        # 1. Get Oversight decision
        if oversight_action is None:
            oversight_action = self._oversight_policy(
                proposal, self._state, self._commander_action_history
            )

        # 2. Enforce budgets — downgrade if exhausted
        if oversight_action.decision == OversightDecision.VETO and self._veto_budget_remaining <= 0:
            oversight_action = oversight_action.model_copy(update={"decision": int(OversightDecision.REVISE)})
        if oversight_action.decision == OversightDecision.FLAG_FOR_HUMAN and self._flag_budget_remaining <= 0:
            oversight_action = oversight_action.model_copy(update={"decision": int(OversightDecision.APPROVE)})

        # 3. Build proposal record
        record = ProposalRecord(
            step=self._state.hour,
            proposal=proposal,
            oversight_decision=int(oversight_action.decision),
            oversight_risk_tier=oversight_action.risk_tier,
            oversight_weakness=oversight_action.weakness,
            oversight_counter_action=(
                oversight_action.counter_proposal.action
                if oversight_action.counter_proposal else -1
            ),
            revised=bool(kwargs.get("was_revised", False)),
            final_action=proposal.action,
            final_target=proposal.target_system,
            cited_lessons=list(proposal.cited_lessons),
        )

        # 4. Route on decision
        prev_state = self._state.clone()
        cs = self._state.council_state
        applied = False
        audit_flagged = False
        decision = OversightDecision(oversight_action.decision)

        if decision == OversightDecision.APPROVE:
            cs.approvals += 1
            applied = True
        elif decision == OversightDecision.REVISE:
            cs.revisions += 1
            applied = True
        elif decision == OversightDecision.VETO:
            cs.vetoes += 1
            self._veto_budget_remaining = max(0, self._veto_budget_remaining - 1)
            applied = False
        elif decision == OversightDecision.FLAG_FOR_HUMAN:
            cs.flags += 1
            self._flag_budget_remaining = max(0, self._flag_budget_remaining - 1)
            applied = True
            audit_flagged = True

        # 5. Citations — record attempted citations (even on veto)
        # Track hallucinated ids (cited lesson does not exist in the playbook).
        hallucinated_citations = 0
        if proposal.cited_lessons:
            for lid in proposal.cited_lessons:
                ok = self._playbook.cite(lid)
                if ok:
                    cs.lessons_cited += 1
                else:
                    hallucinated_citations += 1

        # 6. Apply action (if allowed) — bastion vs governance branches
        stamina_cost = 0.0
        team_msgs: List[Dict[str, str]] = []
        governance_result: Dict[str, Any] = {}
        governance_prereq_violations: List[str] = []
        governance_compliance_count = 0

        if applied:
            if is_bastion_action(proposal.action):
                target_name = SYSTEM_NAMES[proposal.target_system]
                governance_prereq_violations = (
                    []
                    if self.disable_governance_prereqs
                    else check_prerequisites(
                        self._state.governance_state,
                        proposal.action,
                        target_name,
                        self._state.hour,
                    )
                )
                if governance_prereq_violations:
                    record_prereq_violations(
                        self._state.governance_state,
                        self._state.hour,
                        governance_prereq_violations,
                        proposal.action,
                        target_name,
                    )
                stamina_cost, self._alerts_accurate = apply_bastion_action(
                    self._state, proposal.action, proposal.target_system, self._rng,
                    method=proposal.method,
                    scope=proposal.scope,
                    rollback_plan=proposal.rollback_plan,
                )
                team_msgs = generate_team_messages(
                    self._state, proposal.action, proposal.target_system, self._rng
                )

            elif is_governance_action(proposal.action):
                target_name = SYSTEM_NAMES[proposal.target_system]
                governance_result = apply_governance_action(
                    self._state.governance_state,
                    proposal.action,
                    target_system=target_name,
                    hour=self._state.hour,
                    severity_arg=proposal.severity_arg,
                    channel_arg=proposal.channel_arg,
                    message_arg=proposal.message_arg,
                    scope_arg=proposal.scope_arg,
                    evidence_arg=proposal.evidence_arg,
                )
                governance_compliance_count = 1
                stamina_cost = 0.02
                self._state.team_stamina = max(0.0, self._state.team_stamina - stamina_cost)
                # If this was a Slack post, see if it answers a pending
                # stakeholder ask (via channel match + non-empty message).
                if proposal.action == int(ActionType.NOTIFY_SLACK_CHANNEL):
                    if not self.disable_stakeholder_events:
                        satisfied = stakeholder_try_respond(
                            self._state.stakeholder_state,
                            channel=proposal.channel_arg,
                            message=proposal.message_arg,
                            hour=self._state.hour,
                        )
                        if satisfied is not None:
                            governance_result["stakeholder_ask_satisfied"] = satisfied.ask_id

                    # Investor channel — handle update regardless of stakeholder flag
                    if proposal.channel_arg in ("investor-relations", "investor_relations"):
                        inv_reply, tier_crossed = self._investor_agent.handle_commander_update(
                            hour=self._state.hour,
                            message_text=proposal.message_arg,
                        )
                        if inv_reply:
                            team_msgs = team_msgs + [inv_reply]
                        if tier_crossed:
                            governance_result["investor_tier_changed"] = self._investor_agent.state.tier()
        else:
            # VETO: small stamina cost for the lost turn
            stamina_cost = 0.02
            self._state.team_stamina = max(0.0, self._state.team_stamina - stamina_cost)

        # 7. Tick any pending process_kill re-compromise events before attacker moves
        recompromised = tick_pending_recompromise(self._state, self._rng)
        for sys_name in recompromised:
            from models import Alert, AlertSeverity
            from dynamics import SYSTEM_IPS
            self._state.alerts.append(Alert(
                source_system=sys_name,
                severity=AlertSeverity.HIGH,
                message=f"Attacker re-established access on {sys_name} — process_kill isolation bypassed via surviving backdoor",
                is_true_positive=True,
                hour=self._state.hour,
                source_ip=SYSTEM_IPS.get(sys_name, "10.0.0.1"),
                dest_ip=SYSTEM_IPS.get(sys_name, "10.0.0.1"),
                mitre_technique="T1543.003",
                mitre_tactic="Persistence",
                process_name="WinSockHelper.exe",
                event_id="EVT-7045",
                confidence=0.88,
            ))

        # 7b. Adversary turn (always advances)
        if self._state.adversary_gen >= 4:
            # Gen 4 — LLM-driven. Use env-configured client + model; fall back
            # to scripted Gen 3 if not available.
            client, model_name = (self.adversary_llm_client, None)
            if client is None:
                client, model_name = make_adversary_client_from_env()
            new_alerts = gen4_adversary_turn(
                self._state, self._rng, client=client, model=model_name
            )
        else:
            new_alerts = adversary_turn(
                self._state, self._rng, generation=self._state.adversary_gen
            )
        self._state.alerts.extend(new_alerts)

        # 7b. Stakeholder pressure events — roll new events and expire overdue
        new_stakeholder_asks = []
        expired_stakeholder_asks = []
        if not self.disable_stakeholder_events:
            new_stakeholder_asks = roll_new_events(
                self._state.stakeholder_state,
                self._rng,
                hour=self._state.hour,
                adversary_gen=self._state.adversary_gen,
                services_disrupted=self._state.services_disrupted,
                data_exfiltrated=self._state.data_exfiltrated,
                management_escalated=self._state.management_escalated,
            )
            expired_stakeholder_asks = expire_overdue_asks(
                self._state.stakeholder_state,
                hour=self._state.hour,
            )
            if new_stakeholder_asks:
                team_msgs = team_msgs + asks_as_team_messages(new_stakeholder_asks)

        # 7c. Investor agent tick — check-ins + anxiety update
        incident_closed = proposal.action == int(ActionType.CLOSE_INCIDENT) and applied
        investor_msgs, investor_pressure = self._investor_agent.tick(
            hour=self._state.hour,
            data_exfiltrated=self._state.data_exfiltrated,
            incident_closed=incident_closed,
        )
        if investor_msgs:
            team_msgs = team_msgs + investor_msgs
        # Sync investor state back to IncidentState for observation / scoring
        self._state.investor_state = self._investor_agent.state

        # 8. Periodic governance checks
        new_gov_violations = periodic_governance_check(
            self._state.governance_state,
            self._state.data_exfiltrated,
            self._state.hour,
        )

        # 9. Management pressure + services update
        if self._state.management_escalated:
            self._state.management_pressure = min(1.0, self._state.management_pressure + 0.05)
        self._state.services_disrupted = sum(
            1 for s in self._state.systems
            if s.name in SERVICE_SYSTEMS and (s.isolated or s.integrity < 0.3)
        )

        # 10. Advance time
        self._state.hour += 1
        self._state.step_count += 1

        # 11. Evaluate council decision
        outcome_correct, critique_precise, counter_succeeded = self._evaluate_council(
            prev_state, self._state, proposal, oversight_action, applied
        )
        record.outcome_correct = outcome_correct

        if decision == OversightDecision.VETO:
            if not outcome_correct:
                cs.correct_vetoes += 1
            else:
                cs.false_vetoes += 1
        if decision == OversightDecision.REVISE and outcome_correct:
            cs.useful_revisions += 1
        if not oversight_action.weakness and decision != OversightDecision.APPROVE:
            cs.vague_critiques += 1
        if oversight_action.counter_proposal and counter_succeeded:
            cs.counter_proposals_adopted += 1
            cs.counter_proposals_succeeded += 1

        # 12. Update trust (skip entirely if trust dynamics disabled → ablation)
        if not self.disable_trust_dynamics:
            self._update_trust(oversight_action, outcome_correct)
            drift_toward_mean(self._state.trust_state)
        self._state.trust_state.snapshot()

        # 13. Write post-mortem lesson (skipped if playbook disabled)
        lesson_utility_delta = 0.0
        if not self.disable_playbook and oversight_action.lesson_text.strip():
            tgt_sys = (
                SYSTEM_NAMES[proposal.target_system]
                if 0 <= proposal.target_system < len(SYSTEM_NAMES) else ""
            )
            alert_conf = -1.0
            for a in reversed(self._state.alerts):
                if a.source_system == tgt_sys and a.confidence > 0:
                    alert_conf = a.confidence
                    break
            auto_tags = make_context_tags(
                adversary_gen=self._state.adversary_gen,
                system_name=tgt_sys,
                alert_confidence=alert_conf,
                extras=oversight_action.lesson_tags,
            )
            self._playbook.write(
                text=oversight_action.lesson_text,
                tags=auto_tags,
                adversary_gen=self._state.adversary_gen,
                task_id=self._state.task_id,
                hour=self._state.hour,
            )

        # 14. Record cited-lesson outcomes
        lesson_was_helpful = False
        if proposal.cited_lessons and outcome_correct:
            lesson_was_helpful = True
            cs.lessons_cited_and_helpful += 1
            for lid in proposal.cited_lessons:
                self._playbook.record_outcome(lid, improved=True)
        elif proposal.cited_lessons and not outcome_correct:
            for lid in proposal.cited_lessons:
                self._playbook.record_outcome(lid, improved=False)

        # 15. Log record
        cs.log(record)

        # 16. Compute rewards
        governance_violation_count = len(governance_prereq_violations) + len(new_gov_violations)

        # Did the Oversight's veto turn out to be correct? (only meaningful on VETO)
        veto_was_correct = None
        if decision == OversightDecision.VETO:
            # outcome_correct is True when the blocked action would have been fine.
            # A "correct" veto is the OPPOSITE — the blocked action would have hurt.
            veto_was_correct = not outcome_correct

        # "Governance chain completed" — destructive action on data system fired
        # WITHOUT any prereq violations this step → the prereq work was done.
        governance_chain_completed = bool(
            applied
            and is_bastion_action(proposal.action)
            and proposal.action in DESTRUCTIVE_BASTION_ACTIONS
            and not governance_prereq_violations
            and SYSTEM_NAMES[proposal.target_system] in DATA_HOLDING_SYSTEMS
        )

        commander_reward = compute_commander_step_reward(
            prev=prev_state,
            curr=self._state,
            stamina_cost=stamina_cost,
            record=record,
            governance_violation_count=governance_violation_count,
            governance_compliance_count=governance_compliance_count,
            governance_chain_completed=governance_chain_completed,
            lesson_was_helpful=lesson_was_helpful,
            veto_was_correct=veto_was_correct,
            hallucinated_citations=hallucinated_citations,
        )
        penalty = compute_penalties(self._state)
        commander_total = commander_reward + penalty
        self._cumulative_commander_reward += commander_total

        oversight_reward = compute_oversight_step_reward(
            record=record,
            outcome_was_correct=outcome_correct,
            critique_was_precise=critique_precise,
            counter_succeeded=counter_succeeded,
            governance_caught=bool(governance_prereq_violations) and not applied,
            lesson_utility_delta=lesson_utility_delta,
            curr_state=self._state,
        )
        self._cumulative_oversight_reward += oversight_reward

        # 17. Stash last critique for next Commander obs (revision UI)
        self._last_critique = {
            "decision": oversight_action.decision,
            "risk_tier": oversight_action.risk_tier,
            "weakness": oversight_action.weakness,
            "missing_evidence": oversight_action.missing_evidence,
            "counter_proposal": (
                oversight_action.counter_proposal.model_dump()
                if oversight_action.counter_proposal else None
            ),
        }

        # 18. Track Commander action history
        self._commander_action_history.append({
            "hour": self._state.hour,
            "action": ACTION_NAMES.get(proposal.action, str(proposal.action)),
            "target": proposal.target_system,
            "justification": proposal.justification[:120],
            "decision": decision.name,
            "outcome_correct": outcome_correct,
        })

        # 19. Termination check
        done = False

        # Per-step system snapshot for dashboard replay
        systems_snapshot = {
            s.name: {
                "compromised": s.compromised,
                "isolated": s.isolated,
                "investigated": s.investigated,
                "has_backdoor": s.has_backdoor,
                "integrity": round(s.integrity, 2),
                "criticality": round(getattr(s, "criticality", 0.5), 2),
            }
            for s in self._state.systems
        }

        # Alerts fired this step (new alerts from adversary turn)
        step_alerts = [
            {
                "severity": a.severity.name if hasattr(a.severity, "name") else str(a.severity),
                "system": a.source_system,
                "message": a.message,
                "mitre": getattr(a, "mitre_technique", ""),
                "mitre_tactic": getattr(a, "mitre_tactic", ""),
                "event_id": getattr(a, "event_id", ""),
                "confidence": round(getattr(a, "confidence", 0.0), 2),
                "source_ip": getattr(a, "source_ip", ""),
                "dest_ip": getattr(a, "dest_ip", ""),
                "process": getattr(a, "process_name", ""),
                "is_true_positive": getattr(a, "is_true_positive", True),
                "hour": getattr(a, "hour", self._state.hour),
            }
            for a in new_alerts
        ]

        # Investor messages fired this step
        investor_step_messages = [
            {
                "hour": m.hour,
                "direction": m.direction,
                "text": m.text,
                "anxiety_before": round(m.anxiety_before, 3),
                "anxiety_after": round(m.anxiety_after, 3),
            }
            for m in self._investor_agent.state.messages
            if m.hour == self._state.hour
        ]

        # Playbook top lessons snapshot (by utility, no tag filter)
        top_lessons = [
            {"lesson_id": ls.lesson_id, "text": ls.text[:120], "utility": round(ls.utility, 3),
             "citations": ls.citations, "wins": ls.wins, "losses": ls.losses}
            for ls in sorted(self._playbook.all(), key=lambda l: l.utility, reverse=True)[:8]
        ]

        info: Dict[str, Any] = {
            "hour": self._state.hour,
            "action_name": ACTION_NAMES.get(proposal.action, str(proposal.action)),
            "stamina_cost": round(stamina_cost, 3),
            "oversight_decision": decision.name,
            "oversight_risk_tier": oversight_action.risk_tier,
            "applied": applied,
            "audit_flagged": audit_flagged,
            "step_reward": round(commander_total, 4),
            "oversight_reward": round(oversight_reward, 4),
            "penalty": round(penalty, 4),
            "governance_prereq_violations": governance_prereq_violations,
            "governance_new_violations": new_gov_violations,
            "governance_result": governance_result,
            "stakeholder_new_asks": [a.model_dump() for a in new_stakeholder_asks],
            "stakeholder_expired_asks": [a.ask_id for a in expired_stakeholder_asks],
            "stakeholder_reputation": round(self._state.stakeholder_state.reputation, 3),
            "outcome_correct": outcome_correct,
            "critique_precise": critique_precise,
            # Dashboard replay fields
            "systems_snapshot": systems_snapshot,
            "step_alerts": step_alerts,
            "investor_step_messages": investor_step_messages,
            "investor_anxiety": round(self._investor_agent.state.anxiety, 3),
            "investor_tier": self._investor_agent.state.tier(),
            "investor_persona": PERSONAS[self._investor_agent.state.persona_id]["name"]
                if hasattr(self._investor_agent.state, "persona_id") else "",
            "data_exfiltrated": round(self._state.data_exfiltrated, 4),
            "team_stamina": round(self._state.team_stamina, 3),
            "management_pressure": round(self._state.management_pressure, 3),
            "services_disrupted": self._state.services_disrupted,
            "trust_snapshot": self._state.trust_state.as_dict(),
            "playbook_snapshot": top_lessons,
        }

        termination_reason = ""
        if self._state.data_exfiltrated >= 1.0:
            done = True
            termination_reason = "total_data_breach"
            info["termination_reason"] = termination_reason
        elif self._state.hour >= self._task.max_hours:
            done = True
            termination_reason = "time_expired"
            info["termination_reason"] = termination_reason

        if done:
            self._done = True
            self._playbook.save()
            had_p1 = any(t.severity == 1 for t in self._state.governance_state.tickets)
            investor_score = self._investor_agent.investor_score()
            final_scores = compute_final_score(
                state=self._state,
                weights=self._task.scoring_weights,
                proposals_made=self._state.council_state.total_proposals,
                max_proposals=self._task.max_hours,
                per_gen_scores=None,
                had_p1=had_p1,
                termination_reason=termination_reason,
                investor_score=investor_score,
            )
            baseline_score = compute_bastion_v1_final_score(
                self._baseline_state, self._task.scoring_weights
            )
            comparison = compute_baseline_comparison(
                self._state, self._baseline_state, self._task.scoring_weights
            )
            info["final_scores"] = final_scores
            info["baseline_final_score"] = round(baseline_score, 4)
            info["comparison_score"] = round(comparison, 4)
            info["cumulative_commander_reward"] = round(self._cumulative_commander_reward, 4)
            info["cumulative_oversight_reward"] = round(self._cumulative_oversight_reward, 4)
            info["data_exfiltrated"] = round(self._state.data_exfiltrated, 4)
            info["attacker_progress"] = round(self._state.attacker_progress, 4)
            info["adversary_gen"] = self._state.adversary_gen
            info["council_summary"] = {
                "total_proposals": cs.total_proposals,
                "approvals": cs.approvals,
                "revisions": cs.revisions,
                "vetoes": cs.vetoes,
                "flags": cs.flags,
                "correct_vetoes": cs.correct_vetoes,
                "false_vetoes": cs.false_vetoes,
                "useful_revisions": cs.useful_revisions,
                "vague_critiques": cs.vague_critiques,
                "counter_proposals_adopted": cs.counter_proposals_adopted,
                "counter_proposals_succeeded": cs.counter_proposals_succeeded,
                "lessons_cited": cs.lessons_cited,
                "lessons_cited_and_helpful": cs.lessons_cited_and_helpful,
            }
            info["trust_final"] = self._state.trust_state.as_dict()
            info["governance_final"] = self._state.governance_state.snapshot()
            info["stakeholder_final"] = self._state.stakeholder_state.snapshot()
            info["investor_final"] = self._investor_agent.state.snapshot()
            info["investor_score"] = round(investor_score, 4)
            info["forensic_report"] = generate_forensic_report(self._state)

        # 20. Build Commander observation for next step
        lessons = self._retrieve_lessons()
        obs = make_observation(
            self._state,
            self._rng,
            task_description=self._task.description if not done else "",
            done=done,
            reward=commander_total,
            alerts_accurate=self._alerts_accurate,
            team_messages=team_msgs,
            shared_playbook=[ls.to_obs_dict() for ls in lessons],
            last_oversight_critique=self._last_critique,
        )
        obs.oversight_reward = oversight_reward
        return obs

    # --- helpers ----------------------------------------------------------

    def _retrieve_lessons(self, max_items: int = 6):
        if self.disable_playbook:
            return []
        tags = make_context_tags(
            adversary_gen=self._state.adversary_gen,
            system_name="",
            alert_confidence=-1.0,
        )
        return self._playbook.retrieve(tags=tags, max_items=max_items)

    def _evaluate_council(
        self,
        prev_state: IncidentState,
        curr_state: IncidentState,
        proposal: CommanderProposal,
        oversight_action: OversightAction,
        applied: bool,
    ) -> tuple:
        """
        Proxy evaluation of the council decision.
          - outcome_correct: did this step net-improve defender metrics?
          - critique_precise: did Oversight's named weakness match a known
            anti-pattern (uninvestigated target, missing prerequisite)?
          - counter_succeeded: did the counter-proposal suggest a sensible
            next action (e.g., investigate before isolate)?
        """
        delta_exfil = curr_state.data_exfiltrated - prev_state.data_exfiltrated
        delta_compromised = curr_state.compromised_count - prev_state.compromised_count
        net_positive = (delta_exfil <= 0.02) and (delta_compromised <= 0)

        if applied:
            outcome_correct = net_positive
        else:
            checks = compute_policy_checks(prev_state, proposal)
            obviously_bad = (
                checks["action_targets_uninvestigated_system"]
                and checks["action_irreversible"]
            ) or bool(checks["prerequisites_missing"])
            outcome_correct = not obviously_bad

        weakness_given = bool(oversight_action.weakness.strip())
        known_anti_pattern = (
            proposal.action in DESTRUCTIVE_BASTION_ACTIONS
            and compute_policy_checks(prev_state, proposal)["action_targets_uninvestigated_system"]
        )
        critique_precise = weakness_given and (
            known_anti_pattern or len(oversight_action.missing_evidence) > 0
        )

        counter_succeeded = False
        if oversight_action.counter_proposal is not None:
            cp = oversight_action.counter_proposal
            if cp.action == int(ActionType.INVESTIGATE_SYSTEM):
                try:
                    s = prev_state.get_system_by_idx(cp.target_system)
                    if not s.investigated and s.name in DATA_HOLDING_SYSTEMS:
                        counter_succeeded = True
                except Exception:
                    pass

        return outcome_correct, critique_precise, counter_succeeded

    def _update_trust(self, oversight_action: OversightAction, outcome_correct: bool) -> None:
        ts = self._state.trust_state
        decision = OversightDecision(oversight_action.decision)

        # Oversight's trust in Commander
        if outcome_correct:
            update_trust_o2c(ts, "correct")
        else:
            update_trust_o2c(ts, "rework_needed")

        # Commander's trust in Oversight
        if decision == OversightDecision.VETO:
            update_trust_c2o(ts, "veto_correct" if not outcome_correct else "veto_wrong")
        elif decision == OversightDecision.APPROVE:
            update_trust_c2o(ts, "approve_correct" if outcome_correct else "approve_wrong")
        elif decision == OversightDecision.REVISE:
            update_trust_c2o(ts, "demand_useful" if outcome_correct else "demand_nitpick")

    @property
    def state(self) -> IncidentState:
        return self._state
