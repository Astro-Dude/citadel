"""
Citadel — Data Models (extends Bastion v1 with council + governance + trust + playbook)

All models extend OpenEnv base types (Action, Observation, State).

New vs Bastion v1:
    * Action space grows 10 -> 18 (adds 8 governance actions)
    * IncidentAction carries an optional justification + cited_lessons + gov args
    * CommanderProposal = justified IncidentAction (what Commander sends to Oversight)
    * OversightAction: structured critique (decision + risk_tier + weakness + missing_evidence + counter_proposal)
    * OversightObservation: the meta-view (proposal + justification + policy checks + history)
    * CouncilState: per-episode bookkeeping of proposals/critiques/revisions
    * IncidentState gains governance_state, trust_state, council_state, adversary_gen
    * IncidentObservation gains shared_playbook, trust_state, governance_summary, adversary_gen
"""

from __future__ import annotations

import random
from enum import IntEnum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from openenv.core.env_server import Action, Observation, State

from governance import (
    GovernanceState,
    NUM_GOVERNANCE_ACTIONS,
    FIRST_GOVERNANCE_ACTION,
    TOTAL_ACTIONS,
)
from trust import TrustState
from stakeholder_events import StakeholderState
from investor_agent import InvestorState


# ---------------------------------------------------------------------------
# Systems in the network (unchanged from Bastion v1)
# ---------------------------------------------------------------------------

SYSTEM_NAMES = [
    "web_server",
    "app_server",
    "database",
    "file_server",
    "email_server",
    "workstations",
    "backup_server",
    "firewall",
]
NUM_SYSTEMS = len(SYSTEM_NAMES)

NETWORK_ADJACENCY: Dict[str, List[str]] = {
    "web_server": ["app_server", "firewall"],
    "app_server": ["web_server", "database", "file_server"],
    "database": ["app_server", "backup_server"],
    "file_server": ["app_server", "email_server", "workstations"],
    "email_server": ["file_server", "workstations"],
    "workstations": ["file_server", "email_server", "app_server"],
    "backup_server": ["database"],
    "firewall": ["web_server"],
}

SYSTEM_CRITICALITY: Dict[str, float] = {
    "web_server": 0.6,
    "app_server": 0.8,
    "database": 1.0,
    "file_server": 0.7,
    "email_server": 0.4,
    "workstations": 0.3,
    "backup_server": 0.9,
    "firewall": 0.5,
}

DATA_SYSTEMS = {"database", "file_server", "email_server", "backup_server"}
SERVICE_SYSTEMS = {"web_server", "app_server", "database", "email_server"}


# ---------------------------------------------------------------------------
# Alert severity
# ---------------------------------------------------------------------------

class AlertSeverity(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


# ---------------------------------------------------------------------------
# Action Space — 18 actions (10 Bastion + 8 governance)
# ---------------------------------------------------------------------------

class ActionType(IntEnum):
    # Bastion v1 actions (0-9)
    INVESTIGATE_SYSTEM = 0
    ISOLATE_SYSTEM = 1
    PATCH_VULNERABILITY = 2
    RESTORE_FROM_BACKUP = 3
    ANALYZE_ALERTS = 4
    DEPLOY_MONITORING = 5
    ESCALATE_TO_MANAGEMENT = 6
    BLOCK_EXTERNAL_TRAFFIC = 7
    HUNT_THREAT = 8
    COORDINATE_TEAM = 9
    # Governance (10-17) — mirror GovernanceAction enum values
    OPEN_SERVICENOW_INCIDENT = 10
    REQUEST_CAB_APPROVAL = 11
    NOTIFY_SLACK_CHANNEL = 12
    LOG_TO_SOX_AUDIT = 13
    PAGE_ONCALL = 14
    NOTIFY_DATA_OWNER = 15
    START_LEGAL_HOLD = 16
    CLOSE_INCIDENT = 17


ACTION_NAMES: Dict[int, str] = {int(a): a.name.lower() for a in ActionType}
NUM_ACTIONS = len(ActionType)           # 18
BASTION_V1_ACTIONS = 10                 # 0-9 inclusive

assert NUM_ACTIONS == TOTAL_ACTIONS, (
    f"ActionType count {NUM_ACTIONS} must match governance TOTAL_ACTIONS {TOTAL_ACTIONS}"
)

TARGETED_ACTIONS = {
    ActionType.INVESTIGATE_SYSTEM,
    ActionType.ISOLATE_SYSTEM,
    ActionType.PATCH_VULNERABILITY,
    ActionType.RESTORE_FROM_BACKUP,
    ActionType.HUNT_THREAT,
    ActionType.NOTIFY_DATA_OWNER,
    ActionType.REQUEST_CAB_APPROVAL,
}


def is_bastion_action(action_idx: int) -> bool:
    return 0 <= action_idx < BASTION_V1_ACTIONS


def is_governance_action(action_idx: int) -> bool:
    return BASTION_V1_ACTIONS <= action_idx < NUM_ACTIONS


# ---------------------------------------------------------------------------
# IncidentAction (extended) — what Commander submits to the env
# ---------------------------------------------------------------------------

class IsolationMethod(str):
    """
    How to isolate a system. Each method has different speed/reversibility tradeoffs.

      firewall_acl  — block at network layer; reversible, minimal service disruption,
                      but attacker retains process-level access until cleaned
      network_unplug — physical/vlan removal; fastest containment, full service loss,
                       requires on-site to restore (costs an extra step)
      process_kill  — terminate attacker processes only; keeps service up, but risks
                      missing persistence (backdoors survive); 40% chance attacker
                      re-establishes within 2 hours
    """
    FIREWALL_ACL   = "firewall_acl"
    NETWORK_UNPLUG = "network_unplug"
    PROCESS_KILL   = "process_kill"

ISOLATION_METHODS = {IsolationMethod.FIREWALL_ACL, IsolationMethod.NETWORK_UNPLUG, IsolationMethod.PROCESS_KILL}


class PatchStrategy(str):
    """
    How to patch. Affects success probability and service impact.

      hotpatch   — live patching without restart; low disruption, 60% effectiveness
      cold_patch — full restart with patch; high disruption, 90% effectiveness
      virtual_patch — WAF/IDS rule blocks exploit path without touching binary;
                      no disruption, 75% effectiveness, only works for network-facing systems
    """
    HOTPATCH       = "hotpatch"
    COLD_PATCH     = "cold_patch"
    VIRTUAL_PATCH  = "virtual_patch"

PATCH_STRATEGIES = {PatchStrategy.HOTPATCH, PatchStrategy.COLD_PATCH, PatchStrategy.VIRTUAL_PATCH}


class MonitoringScope(str):
    """
    What to monitor. Affects detection quality vs noise ratio.

      process_events   — log all process spawns; catches lateral movement, high volume
      network_traffic  — capture all flows; catches exfil, very high volume
      auth_events      — log all auth attempts; targeted, low volume, misses post-auth
      full_endpoint    — all of the above; maximum detection, significant performance hit
                         (integrity -5% per hour on target while active)
    """
    PROCESS_EVENTS  = "process_events"
    NETWORK_TRAFFIC = "network_traffic"
    AUTH_EVENTS     = "auth_events"
    FULL_ENDPOINT   = "full_endpoint"

MONITORING_SCOPES = {MonitoringScope.PROCESS_EVENTS, MonitoringScope.NETWORK_TRAFFIC, MonitoringScope.AUTH_EVENTS, MonitoringScope.FULL_ENDPOINT}


class IncidentAction(Action):
    """
    Commander's action. Carries optional justification + cited lessons
    + governance-specific arguments + rich method/scope/rollback fields
    that give the action real technical depth.

    New fields (Option A — richer payloads):
      method       — HOW to execute the action (isolation method, patch strategy, monitoring scope)
      scope        — WHAT scope: IP ranges, process names, port filters
      rollback_plan — HOW to undo if it goes wrong
    """
    action: int = Field(
        ..., ge=0, lt=NUM_ACTIONS,
        description=f"Action index (0-{NUM_ACTIONS - 1}). 0-9 = incident response, 10-17 = governance.",
    )
    target_system: int = Field(
        default=0, ge=0, lt=NUM_SYSTEMS,
        description="Target system index (0-7) for targeted actions.",
    )
    # Council fields — optional, ignored by the Bastion-style server flow
    justification: str = Field(
        default="",
        description="Free-text reasoning for this action (required when council is active).",
        max_length=1000,
    )
    cited_lessons: List[str] = Field(
        default_factory=list,
        description="Lesson IDs from the shared playbook that informed this proposal.",
    )
    # Richer action payload — Option A
    method: str = Field(
        default="",
        max_length=32,
        description=(
            "How to execute the action. "
            "isolate: firewall_acl | network_unplug | process_kill. "
            "patch: hotpatch | cold_patch | virtual_patch. "
            "deploy_monitoring: process_events | network_traffic | auth_events | full_endpoint."
        ),
    )
    scope: str = Field(
        default="",
        max_length=200,
        description="Optional scope constraint: IP range, process name filter, port. E.g. '10.1.3.30/32' or 'lsass.exe'.",
    )
    rollback_plan: str = Field(
        default="",
        max_length=300,
        description="How to reverse this action if it causes collateral damage or was wrong.",
    )
    # Governance-specific args — used by actions 10-17, ignored otherwise
    severity_arg: int = Field(default=2, ge=1, le=4, description="For governance actions (P1-P4).")
    channel_arg: str = Field(default="sec-ops", max_length=64)
    message_arg: str = Field(default="", max_length=400)
    scope_arg: str = Field(default="", max_length=200)
    evidence_arg: str = Field(default="", max_length=400)


# ---------------------------------------------------------------------------
# CommanderProposal — the wrapper the env uses to talk to Oversight
# ---------------------------------------------------------------------------

class CommanderProposal(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    """What the Commander proposes in step t. Oversight sees one of these."""
    action: int
    target_system: int
    justification: str = ""
    cited_lessons: List[str] = Field(default_factory=list)
    method: str = ""
    scope: str = ""
    rollback_plan: str = ""
    severity_arg: int = 2
    channel_arg: str = "sec-ops"
    message_arg: str = ""
    scope_arg: str = ""
    evidence_arg: str = ""

    @classmethod
    def from_action(cls, a: IncidentAction) -> "CommanderProposal":
        return cls(
            action=a.action,
            target_system=a.target_system,
            justification=a.justification,
            cited_lessons=list(a.cited_lessons),
            method=a.method,
            scope=a.scope,
            rollback_plan=a.rollback_plan,
            severity_arg=a.severity_arg,
            channel_arg=a.channel_arg,
            message_arg=a.message_arg,
            scope_arg=a.scope_arg,
            evidence_arg=a.evidence_arg,
        )

    def to_action(self) -> IncidentAction:
        return IncidentAction(
            action=self.action,
            target_system=self.target_system,
            justification=self.justification,
            cited_lessons=list(self.cited_lessons),
            method=self.method,
            scope=self.scope,
            rollback_plan=self.rollback_plan,
            severity_arg=self.severity_arg,
            channel_arg=self.channel_arg,
            message_arg=self.message_arg,
            scope_arg=self.scope_arg,
            evidence_arg=self.evidence_arg,
        )


# ---------------------------------------------------------------------------
# Oversight action space — council decision + structured critique
# ---------------------------------------------------------------------------

class OversightDecision(IntEnum):
    APPROVE = 0
    REVISE = 1               # ask Commander to revise once before deciding
    VETO = 2                 # hard reject — Commander must pick a DIFFERENT action
    FLAG_FOR_HUMAN = 3       # execute but audit-flag

NUM_OVERSIGHT_DECISIONS = len(OversightDecision)


class CounterProposal(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    """Oversight's suggested alternative (used with REVISE or VETO)."""
    action: int = Field(..., ge=0, lt=NUM_ACTIONS)
    target_system: int = Field(default=0, ge=0, lt=NUM_SYSTEMS)
    rationale: str = Field(default="", max_length=400)


class OversightAction(Action):
    """
    The structured critique Oversight emits every step. This is the upgrade
    over a classifier-style gate: Oversight must *reason about the domain*.
    """
    decision: int = Field(..., ge=0, lt=NUM_OVERSIGHT_DECISIONS,
                          description="0=approve, 1=revise, 2=veto, 3=flag_for_human")
    risk_tier: int = Field(default=2, ge=1, le=5,
                           description="Commander's proposal risk (1=safe, 5=catastrophic)")
    weakness: str = Field(default="", max_length=400,
                          description="Specific weakness in the proposal — be concrete.")
    missing_evidence: List[str] = Field(
        default_factory=list,
        description="Evidence (alerts/logs/investigations) missing to support this action.",
    )
    counter_proposal: Optional[CounterProposal] = Field(
        default=None,
        description="Optional alternative action (required on VETO, optional on REVISE).",
    )
    # Post-mortem lesson — filled in after action resolves (next step).
    lesson_text: str = Field(default="", max_length=240)
    lesson_tags: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Per-system state
# ---------------------------------------------------------------------------

class SystemState(State):
    name: str = ""
    compromised: bool = False
    isolated: bool = False
    investigated: bool = False
    has_backdoor: bool = False
    integrity: float = Field(default=1.0, ge=0.0, le=1.0)
    criticality: float = Field(default=0.5, ge=0.0, le=1.0)
    monitoring_level: int = Field(default=0, ge=0, le=3)
    patched: bool = False


# ---------------------------------------------------------------------------
# Alert
# ---------------------------------------------------------------------------

class Alert(State):
    source_system: str = ""
    severity: int = 0
    message: str = ""
    is_true_positive: bool = True
    hour: int = 0
    source_ip: str = ""
    dest_ip: str = ""
    mitre_technique: str = ""
    mitre_tactic: str = ""
    process_name: str = ""
    event_id: str = ""
    file_hash: str = ""
    confidence: float = 0.0
    raw_log: str = ""


# ---------------------------------------------------------------------------
# Council state — per-episode bookkeeping of the Commander↔Oversight loop
# ---------------------------------------------------------------------------

class ProposalRecord(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    """One round of the council loop."""
    step: int
    proposal: CommanderProposal
    oversight_decision: int = -1
    oversight_risk_tier: int = 0
    oversight_weakness: str = ""
    oversight_counter_action: int = -1
    revised: bool = False
    final_action: int = -1
    final_target: int = -1
    outcome_correct: Optional[bool] = None
    cited_lessons: List[str] = Field(default_factory=list)


class CouncilState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    """Running record of the council's decisions this episode."""
    history: List[ProposalRecord] = Field(default_factory=list)
    total_proposals: int = 0
    approvals: int = 0
    revisions: int = 0
    vetoes: int = 0
    flags: int = 0
    correct_vetoes: int = 0
    false_vetoes: int = 0
    useful_revisions: int = 0
    vague_critiques: int = 0
    counter_proposals_adopted: int = 0
    counter_proposals_succeeded: int = 0
    lessons_cited: int = 0
    lessons_cited_and_helpful: int = 0

    def log(self, record: ProposalRecord) -> None:
        self.history.append(record)
        self.total_proposals += 1


# ---------------------------------------------------------------------------
# Full incident state (ground truth — extended)
# ---------------------------------------------------------------------------

class IncidentState(State):
    """Full ground-truth state of the incident. Extended from Bastion v1."""
    systems: List[SystemState] = Field(default_factory=list)
    alerts: List[Alert] = Field(default_factory=list)

    attacker_progress: float = Field(default=0.0, ge=0.0, le=1.0)
    attacker_stealth: float = Field(default=0.8, ge=0.0, le=1.0)
    data_exfiltrated: float = Field(default=0.0, ge=0.0, le=1.0)
    services_disrupted: int = Field(default=0, ge=0)
    team_stamina: float = Field(default=1.0, ge=0.0, le=1.0)
    hour: int = Field(default=0, ge=0)

    external_blocked: bool = False
    management_escalated: bool = False
    management_pressure: float = Field(default=0.0, ge=0.0, le=1.0)
    task_id: str = ""

    # --- Citadel additions -----------------------------------------------
    governance_state: GovernanceState = Field(default_factory=GovernanceState)
    trust_state: TrustState = Field(default_factory=TrustState)
    council_state: CouncilState = Field(default_factory=CouncilState)
    stakeholder_state: StakeholderState = Field(default_factory=StakeholderState)
    investor_state: InvestorState = Field(default_factory=InvestorState)
    adversary_gen: int = Field(default=1, ge=1, le=4,
        description="Adversary generation this episode (1/2/3 scripted, 4 LLM).")
    episode_id: str = ""
    step_count: int = 0

    def get_system(self, name: str) -> SystemState:
        for s in self.systems:
            if s.name == name:
                return s
        raise ValueError(f"Unknown system: {name}")

    def get_system_by_idx(self, idx: int) -> SystemState:
        return self.systems[idx]

    @property
    def compromised_count(self) -> int:
        return sum(1 for s in self.systems if s.compromised and not s.isolated)

    @property
    def isolated_count(self) -> int:
        return sum(1 for s in self.systems if s.isolated)

    @property
    def investigated_count(self) -> int:
        return sum(1 for s in self.systems if s.investigated)

    @property
    def services_intact(self) -> int:
        return sum(
            1 for s in self.systems
            if s.name in SERVICE_SYSTEMS and not s.isolated and s.integrity > 0.3
        )

    def snapshot(self) -> Dict[str, Any]:
        d = self.model_dump()
        d["compromised_count"] = self.compromised_count
        d["isolated_count"] = self.isolated_count
        d["investigated_count"] = self.investigated_count
        d["services_intact"] = self.services_intact
        return d

    def clone(self) -> "IncidentState":
        return IncidentState(**self.model_dump())


# ---------------------------------------------------------------------------
# Observations — Commander + Oversight
# ---------------------------------------------------------------------------

class IncidentObservation(Observation):
    """What the Commander sees at step t. Extended with council surfaces."""
    # Bastion v1 surfaces
    systems_visible: List[Dict[str, Any]] = Field(default_factory=list)
    alert_queue: List[Dict[str, Any]] = Field(default_factory=list)
    estimated_breach_severity: str = "unknown"
    estimated_data_at_risk: float = 0.0
    services_disrupted: int = 0
    services_total: int = Field(default=len(SERVICE_SYSTEMS))
    team_stamina: float = 1.0
    hour: int = 0
    hours_remaining: int = 12
    external_blocked: bool = False
    management_escalated: bool = False
    task_description: str = ""
    team_messages: List[Dict[str, str]] = Field(default_factory=list)

    # --- Citadel additions -----------------------------------------------
    governance_summary: Dict[str, Any] = Field(default_factory=dict)
    trust_summary: Dict[str, Any] = Field(default_factory=dict)
    stakeholder_summary: Dict[str, Any] = Field(default_factory=dict)
    investor_summary: Dict[str, Any] = Field(default_factory=dict)
    shared_playbook: List[Dict[str, Any]] = Field(default_factory=list)
    adversary_gen: int = 1
    last_oversight_critique: Dict[str, Any] = Field(default_factory=dict)
    done: bool = False
    reward: Optional[float] = None


class OversightObservation(Observation):
    """The meta-view — what Oversight sees when asked to critique a proposal."""
    proposed_action: Dict[str, Any] = Field(default_factory=dict)
    justification: str = ""
    cited_lessons: List[str] = Field(default_factory=list)
    commander_observation: Dict[str, Any] = Field(default_factory=dict)
    commander_action_history: List[Dict[str, Any]] = Field(default_factory=list)
    policy_checks: Dict[str, Any] = Field(default_factory=dict)
    veto_budget_remaining: int = 0
    flag_budget_remaining: int = 0
    shared_playbook: List[Dict[str, Any]] = Field(default_factory=list)
    trust_summary: Dict[str, Any] = Field(default_factory=dict)
    governance_summary: Dict[str, Any] = Field(default_factory=dict)
    adversary_gen: int = 1
    hour: int = 0
    task_description: str = ""


# ---------------------------------------------------------------------------
# make_observation — Commander's partially-observable view
# ---------------------------------------------------------------------------

def make_observation(
    state: IncidentState,
    rng: random.Random,
    task_description: str = "",
    done: bool = False,
    reward: float | None = None,
    team_messages: list[dict[str, str]] | None = None,
    alerts_accurate: bool = False,
    shared_playbook: list[dict[str, Any]] | None = None,
    last_oversight_critique: dict[str, Any] | None = None,
    hide_adversary_gen: bool = False,
) -> IncidentObservation:
    """Build the Commander's partially-observable view of the true state."""

    systems_vis: List[Dict[str, Any]] = []
    for s in state.systems:
        vis: Dict[str, Any] = {
            "name": s.name,
            "isolated": s.isolated,
            "investigated": s.investigated,
            "integrity": round(s.integrity, 2),
            "criticality": s.criticality,
            "monitoring_level": s.monitoring_level,
            "patched": s.patched,
        }
        if s.investigated:
            vis["compromised"] = s.compromised
            vis["has_backdoor"] = s.has_backdoor
        else:
            vis["compromised"] = "unknown"
            vis["has_backdoor"] = "unknown"
        systems_vis.append(vis)

    alert_vis: List[Dict[str, Any]] = []
    for a in state.alerts[-6:]:
        av: Dict[str, Any] = {
            "source_system": a.source_system,
            "severity": ["low", "medium", "high", "critical"][a.severity],
            "message": a.message,
            "hour": a.hour,
        }
        if a.source_ip:
            av["source_ip"] = a.source_ip
        if a.dest_ip:
            av["dest_ip"] = a.dest_ip
        if a.mitre_technique:
            av["mitre_technique"] = a.mitre_technique
            av["mitre_tactic"] = a.mitre_tactic
        if a.process_name:
            av["process_name"] = a.process_name
        if a.event_id:
            av["event_id"] = a.event_id
        if a.confidence > 0:
            av["confidence"] = round(a.confidence, 2)
        if a.file_hash:
            av["file_hash"] = a.file_hash
        if a.raw_log:
            av["raw_log"] = a.raw_log
        if alerts_accurate:
            av["confirmed"] = a.is_true_positive
        alert_vis.append(av)

    investigated = [s for s in state.systems if s.investigated]
    known_compromised = sum(1 for s in investigated if s.compromised)
    if not investigated:
        severity_est = "unknown"
    elif known_compromised == 0:
        severity_est = "low"
    elif known_compromised <= 2:
        severity_est = "medium"
    elif known_compromised <= 4:
        severity_est = "high"
    else:
        severity_est = "critical"

    noise = rng.gauss(0, 0.1)
    data_est = max(0.0, min(1.0, state.data_exfiltrated + noise))

    disrupted = sum(
        1 for s in state.systems
        if s.name in SERVICE_SYSTEMS and (s.isolated or s.integrity < 0.3)
    )

    state.governance_state.set_current_hour(state.hour)
    governance_summary = state.governance_state.snapshot()
    trust_summary = state.trust_state.as_dict()
    stakeholder_summary = state.stakeholder_state.snapshot()
    investor_summary = state.investor_state.snapshot()

    return IncidentObservation(
        systems_visible=systems_vis,
        alert_queue=alert_vis,
        estimated_breach_severity=severity_est,
        estimated_data_at_risk=round(data_est, 3),
        services_disrupted=disrupted,
        team_stamina=round(state.team_stamina, 2),
        hour=state.hour,
        hours_remaining=max(0, 12 - state.hour),
        external_blocked=state.external_blocked,
        management_escalated=state.management_escalated,
        task_description=task_description,
        team_messages=team_messages or [],
        governance_summary=governance_summary,
        trust_summary=trust_summary,
        stakeholder_summary=stakeholder_summary,
        investor_summary=investor_summary,
        shared_playbook=shared_playbook or [],
        adversary_gen=(0 if hide_adversary_gen else state.adversary_gen),
        last_oversight_critique=last_oversight_critique or {},
        done=done,
        reward=reward,
    )


# ---------------------------------------------------------------------------
# Re-export Lesson from playbook for external imports
# ---------------------------------------------------------------------------

from playbook import Lesson  # noqa: E402,F401  (intentional re-export)
