"""
Citadel — Enterprise Governance Layer (Theme 3.1 + Scaler AI Labs sub-theme)

Six simulated enterprise systems that Commander must interact with during
an incident. Judges will recognize these from real SOC work:

    ServiceNow       — incident tracking + change management
    SOX Audit Log    — immutable compliance trail
    Slack            — stakeholder notifications
    PagerDuty        — on-call escalation
    Data Owner Reg.  — who owns each data system
    Legal / GDPR     — 72-hour breach notification timer

Governance rules the environment enforces (a.k.a. "the business logic"):

  * Pre-requisite chains — you can't isolate a database without first
    (a) getting CAB approval and (b) notifying the data owner. Skipping
    these is a compliance violation with a stiff reward penalty.

  * Severity mapping — a P1 incident requires an INC ticket + PagerDuty
    page + Slack post within a bounded step window. Miss the window and
    you earn a compliance violation.

  * SOX irreversibility — once an action is logged to SOX, it can't be
    rolled back without regulator notification. This matters because
    destructive actions (isolate, block_external_traffic) should be
    logged only when you're confident.

  * RACI enforcement — only the registered data owner can approve changes
    to their data system outside emergency protocols.

  * GDPR timer — if data_exfiltrated > 0.20, a 72-simulated-hour breach
    notification clock starts. Missing the deadline is catastrophic.

Every governance event emits a structured record in GovernanceState.events,
which feeds both the reward function and the observation given to agents.
"""

from __future__ import annotations

import time
from enum import IntEnum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, PrivateAttr


# ---------------------------------------------------------------------------
# Incident severity + governance action enums
# ---------------------------------------------------------------------------

class IncidentSeverity(IntEnum):
    P4 = 4   # low
    P3 = 3   # moderate
    P2 = 2   # high
    P1 = 1   # critical


# Governance actions — these are new Commander actions 10..17
# (Bastion's 0..9 remain unchanged)
class GovernanceAction(IntEnum):
    OPEN_SERVICENOW_INCIDENT = 10   # arg: severity
    REQUEST_CAB_APPROVAL     = 11   # arg: action_to_approve, target
    NOTIFY_SLACK_CHANNEL     = 12   # arg: channel, short message
    LOG_TO_SOX_AUDIT         = 13   # arg: action, evidence
    PAGE_ONCALL              = 14   # arg: team, severity
    NOTIFY_DATA_OWNER        = 15   # arg: system
    START_LEGAL_HOLD         = 16   # arg: scope
    CLOSE_INCIDENT           = 17   # arg: resolution_summary


GOVERNANCE_ACTION_NAMES: Dict[int, str] = {
    int(a): a.name.lower() for a in GovernanceAction
}

NUM_GOVERNANCE_ACTIONS = len(GovernanceAction)

# Bastion had 10 actions; governance adds 8 => 18 total
FIRST_GOVERNANCE_ACTION = int(GovernanceAction.OPEN_SERVICENOW_INCIDENT)   # 10
TOTAL_ACTIONS = FIRST_GOVERNANCE_ACTION + NUM_GOVERNANCE_ACTIONS           # 18


# ---------------------------------------------------------------------------
# Data owner registry — which team owns which data-holding system
# ---------------------------------------------------------------------------

DATA_OWNERS: Dict[str, str] = {
    "database":       "data-platform",
    "file_server":    "it-ops",
    "email_server":   "corporate-it",
    "backup_server":  "dr-team",
    "web_server":     "product-eng",
    "app_server":     "product-eng",
    "workstations":   "it-ops",
    "firewall":       "network-ops",
}


SLACK_CHANNELS = {
    "data-governance":   "notify before touching any data-holding system",
    "sec-leadership":    "P1 incidents and legal escalations",
    "sec-ops":           "general SOC coordination",
    "incident-war-room": "live incident stream",
}


# ---------------------------------------------------------------------------
# Governance state (lives inside IncidentState as a sub-object)
# ---------------------------------------------------------------------------

class ServiceNowTicket(BaseModel):
    ticket_id: str
    severity: int                   # 1-4 (P1 is 1)
    opened_hour: int
    status: str = "open"            # open | in_progress | closed
    closed_hour: Optional[int] = None
    cab_approved: bool = False


class GovernanceEvent(BaseModel):
    """One structured record — every governance action creates one."""
    kind: str                       # e.g. "sox_log", "slack_post"
    hour: int
    detail: Dict[str, Any] = Field(default_factory=dict)


class GovernanceState(BaseModel):
    """All governance-layer bookkeeping for one episode."""
    # ServiceNow
    tickets: List[ServiceNowTicket] = Field(default_factory=list)
    # Change Advisory Board — approvals granted, keyed by (action, target_system)
    cab_approvals: Dict[str, int] = Field(default_factory=dict)   # key -> hour granted
    # SOX audit trail — immutable list of logged actions
    sox_log: List[Dict[str, Any]] = Field(default_factory=list)
    # Slack posts made
    slack_posts: List[Dict[str, Any]] = Field(default_factory=list)
    # PagerDuty pages
    pages: List[Dict[str, Any]] = Field(default_factory=list)
    # Data owners notified per system
    data_owners_notified: Dict[str, int] = Field(default_factory=dict)  # system -> hour
    # Legal / GDPR
    legal_hold_active: bool = False
    gdpr_clock_started_at: Optional[int] = None     # hour the clock started
    gdpr_notified: bool = False
    # Violations tallied this episode (for reward + scoring)
    violations: List[GovernanceEvent] = Field(default_factory=list)
    # Compliance hits (for reward)
    compliance_hits: List[GovernanceEvent] = Field(default_factory=list)
    # Non-serialized — only used for the periodic snapshot
    _hour_for_snapshot: int = PrivateAttr(default=0)

    # --- convenience ---

    def cab_key(self, action_idx: int, target_system: str) -> str:
        return f"{action_idx}:{target_system}"

    def has_cab_approval(self, action_idx: int, target_system: str) -> bool:
        return self.cab_key(action_idx, target_system) in self.cab_approvals

    def has_open_p1(self) -> bool:
        return any(t.severity == 1 and t.status != "closed" for t in self.tickets)

    def has_open_ticket(self) -> bool:
        return any(t.status != "closed" for t in self.tickets)

    def data_owner_notified(self, system: str) -> bool:
        return system in self.data_owners_notified

    # --- observation summary ---

    def snapshot(self) -> Dict[str, Any]:
        return {
            "open_tickets": [
                {"id": t.ticket_id, "severity": f"P{t.severity}", "status": t.status, "cab_approved": t.cab_approved}
                for t in self.tickets if t.status != "closed"
            ],
            "cab_approvals_count": len(self.cab_approvals),
            "sox_log_count": len(self.sox_log),
            "slack_posts_count": len(self.slack_posts),
            "pages_count": len(self.pages),
            "data_owners_notified": list(self.data_owners_notified.keys()),
            "legal_hold_active": self.legal_hold_active,
            "gdpr_clock_hours_remaining": (
                (72 - (self._current_hour() - self.gdpr_clock_started_at))
                if self.gdpr_clock_started_at is not None else None
            ),
            "gdpr_notified": self.gdpr_notified,
            "violations_count": len(self.violations),
            "compliance_hits_count": len(self.compliance_hits),
        }

    def _current_hour(self) -> int:
        """Best-effort — filled in by environment when it calls snapshot()."""
        return self._hour_for_snapshot

    def set_current_hour(self, hour: int) -> None:
        self._hour_for_snapshot = hour


# ---------------------------------------------------------------------------
# Governance action executor
# ---------------------------------------------------------------------------

def _record_violation(gstate: GovernanceState, hour: int, reason: str, **detail: Any) -> None:
    gstate.violations.append(GovernanceEvent(
        kind="violation",
        hour=hour,
        detail={"reason": reason, **detail},
    ))


def _record_compliance(gstate: GovernanceState, hour: int, kind: str, **detail: Any) -> None:
    gstate.compliance_hits.append(GovernanceEvent(
        kind=kind,
        hour=hour,
        detail=detail,
    ))


def apply_governance_action(
    gstate: GovernanceState,
    action_idx: int,
    target_system: str,
    hour: int,
    severity_arg: int = 2,
    channel_arg: str = "sec-ops",
    message_arg: str = "",
    scope_arg: str = "",
    evidence_arg: str = "",
) -> Dict[str, Any]:
    """
    Apply one governance action. Returns a small result dict summarizing
    what happened (used for the step-info block shown to the agent).
    """
    a = GovernanceAction(action_idx)
    result: Dict[str, Any] = {"governance_action": a.name.lower(), "hour": hour, "success": True}

    if a == GovernanceAction.OPEN_SERVICENOW_INCIDENT:
        tid = f"INC{10000 + len(gstate.tickets) + 1}"
        severity = max(1, min(4, severity_arg))
        gstate.tickets.append(ServiceNowTicket(
            ticket_id=tid,
            severity=severity,
            opened_hour=hour,
        ))
        _record_compliance(gstate, hour, "servicenow_opened", ticket=tid, severity=severity)
        result["ticket_id"] = tid
        result["severity"] = f"P{severity}"

    elif a == GovernanceAction.REQUEST_CAB_APPROVAL:
        key = gstate.cab_key(severity_arg, target_system)  # severity_arg used as action_to_approve
        if key not in gstate.cab_approvals:
            gstate.cab_approvals[key] = hour
            # Mark related tickets as approved
            for t in gstate.tickets:
                if t.status != "closed":
                    t.cab_approved = True
            _record_compliance(gstate, hour, "cab_approval", action=severity_arg, target=target_system)
            result["approved"] = True
        else:
            result["approved_already"] = True

    elif a == GovernanceAction.NOTIFY_SLACK_CHANNEL:
        ch = channel_arg if channel_arg in SLACK_CHANNELS else "sec-ops"
        gstate.slack_posts.append({
            "channel": ch,
            "hour": hour,
            "message": (message_arg or "(no message)")[:200],
        })
        _record_compliance(gstate, hour, "slack_posted", channel=ch)
        result["channel"] = ch

    elif a == GovernanceAction.LOG_TO_SOX_AUDIT:
        entry = {
            "hour": hour,
            "action": severity_arg,        # action being logged
            "target": target_system,
            "evidence": (evidence_arg or "(none)")[:200],
            "ts": time.time(),
        }
        gstate.sox_log.append(entry)
        _record_compliance(gstate, hour, "sox_logged", target=target_system)
        result["sox_entry_count"] = len(gstate.sox_log)

    elif a == GovernanceAction.PAGE_ONCALL:
        sev = max(1, min(4, severity_arg))
        gstate.pages.append({
            "hour": hour,
            "team": channel_arg or "security",
            "severity": sev,
        })
        _record_compliance(gstate, hour, "paged_oncall", team=channel_arg, severity=sev)
        result["severity"] = f"P{sev}"

    elif a == GovernanceAction.NOTIFY_DATA_OWNER:
        owner = DATA_OWNERS.get(target_system, "unknown")
        gstate.data_owners_notified[target_system] = hour
        _record_compliance(gstate, hour, "data_owner_notified", system=target_system, owner=owner)
        result["owner"] = owner

    elif a == GovernanceAction.START_LEGAL_HOLD:
        gstate.legal_hold_active = True
        _record_compliance(gstate, hour, "legal_hold_started", scope=scope_arg or "incident")
        result["scope"] = scope_arg or "incident"

    elif a == GovernanceAction.CLOSE_INCIDENT:
        closed = 0
        for t in gstate.tickets:
            if t.status != "closed":
                t.status = "closed"
                t.closed_hour = hour
                closed += 1
        result["closed_count"] = closed
        if closed:
            _record_compliance(gstate, hour, "incident_closed", summary=(evidence_arg or "resolved")[:160])

    else:
        result["success"] = False
        result["reason"] = f"unknown governance action: {action_idx}"

    return result


# ---------------------------------------------------------------------------
# Pre-requisite checks (called BEFORE a destructive Bastion action executes)
# ---------------------------------------------------------------------------

DESTRUCTIVE_BASTION_ACTIONS = {1, 7}  # ISOLATE_SYSTEM, BLOCK_EXTERNAL_TRAFFIC
DATA_HOLDING_SYSTEMS = {"database", "file_server", "email_server", "backup_server"}


def check_prerequisites(
    gstate: GovernanceState,
    action_idx: int,
    target_system: str,
    hour: int,
) -> List[str]:
    """
    Returns a list of violation reasons (empty list == all prerequisites met).
    Only applies to destructive actions on data-holding systems.
    """
    problems: List[str] = []

    if action_idx not in DESTRUCTIVE_BASTION_ACTIONS:
        return problems

    # For destructive actions on data systems, we need CAB approval + owner notify
    if target_system in DATA_HOLDING_SYSTEMS:
        if not gstate.has_cab_approval(action_idx, target_system):
            problems.append(f"no_cab_approval_for_{target_system}")
        if not gstate.data_owner_notified(target_system):
            problems.append(f"data_owner_{DATA_OWNERS.get(target_system, 'unknown')}_not_notified")

    # Block external traffic is global — needs a Slack heads-up to sec-leadership
    if action_idx == 7:  # BLOCK_EXTERNAL_TRAFFIC
        posted_to_leadership = any(
            p["channel"] == "sec-leadership" for p in gstate.slack_posts
        )
        if not posted_to_leadership:
            problems.append("block_external_without_sec_leadership_notification")

    return problems


def record_prereq_violations(
    gstate: GovernanceState,
    hour: int,
    violations: List[str],
    action_idx: int,
    target_system: str,
) -> None:
    for v in violations:
        _record_violation(
            gstate, hour, v,
            action=action_idx,
            target=target_system,
        )


# ---------------------------------------------------------------------------
# Periodic governance checks (called each step from the environment)
# ---------------------------------------------------------------------------

P1_RESPONSE_WINDOW_HOURS = 2     # P1 requires INC+PD+Slack within 2 hours of opening
GDPR_WINDOW_HOURS = 72


def periodic_governance_check(
    gstate: GovernanceState,
    data_exfiltrated: float,
    hour: int,
) -> List[str]:
    """
    Run every step AFTER the action is applied. Returns any new violations
    that occurred this step. Also starts/advances the GDPR clock as needed.
    """
    new_violations: List[str] = []

    # --- GDPR clock ---
    if data_exfiltrated > 0.20 and gstate.gdpr_clock_started_at is None:
        gstate.gdpr_clock_started_at = hour
        _record_compliance(gstate, hour, "gdpr_clock_started")

    if (
        gstate.gdpr_clock_started_at is not None
        and not gstate.gdpr_notified
        and hour - gstate.gdpr_clock_started_at >= GDPR_WINDOW_HOURS
    ):
        # Missed GDPR deadline — catastrophic
        v = "gdpr_notification_missed"
        _record_violation(gstate, hour, v)
        new_violations.append(v)

    # --- P1 response window ---
    for t in gstate.tickets:
        if t.severity == 1 and t.status != "closed":
            age = hour - t.opened_hour
            # Within window, check that PagerDuty + Slack leadership were posted
            if age == P1_RESPONSE_WINDOW_HOURS:
                has_page = any(
                    p["hour"] >= t.opened_hour for p in gstate.pages
                )
                has_sl_post = any(
                    p["channel"] == "sec-leadership" and p["hour"] >= t.opened_hour
                    for p in gstate.slack_posts
                )
                if not has_page:
                    v = f"p1_no_page_{t.ticket_id}"
                    _record_violation(gstate, hour, v)
                    new_violations.append(v)
                if not has_sl_post:
                    v = f"p1_no_sec_leadership_post_{t.ticket_id}"
                    _record_violation(gstate, hour, v)
                    new_violations.append(v)

    return new_violations


# ---------------------------------------------------------------------------
# Governance compliance score (used in final_score)
# ---------------------------------------------------------------------------

def compliance_score(gstate: GovernanceState, had_p1: bool = False) -> float:
    """
    Returns a 0..1 score reflecting overall governance hygiene:
      - violations drag the score down fast
      - positive compliance actions (SOX log, owner notify, CAB approval) lift it
      - having no tickets open at all is not credit — you need to have DONE the work
    """
    hits = len(gstate.compliance_hits)
    viols = len(gstate.violations)

    # Base: compliance actions minus violations (each violation 2x weight)
    base = hits - 2 * viols

    # Without any activity at all, score is low (agent must engage the layer)
    if hits == 0 and viols == 0 and not had_p1:
        return 0.3

    # Map base (~-6..+15) to [0,1] via sigmoidish function
    import math
    return 1.0 / (1.0 + math.exp(-base / 4.0))
