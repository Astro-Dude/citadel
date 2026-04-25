"""
Citadel — Stakeholder Pressure Events

Models the realistic chaos of SOC work: during an incident, executives,
legal, customers, and the press interject with urgent demands that pull
Commander's attention in conflicting directions. Ignoring them damages
reputation (persistent state); responding costs a turn but preserves the
political capital Commander needs later.

Each event:
  * has a unique id, a sender, a demand, and a deadline (hours from fire)
  * is surfaced as a team_messages entry in the Commander's observation
  * is "responded to" when Commander posts to a matching Slack channel
  * if not responded to by deadline → reputation drop + compliance violation

Events are rolled stochastically each step at gen-tuned probabilities.
Disable entirely via `disable_stakeholder_events=True` on the env.

This hits the Halluminate multi-actor sub-theme without requiring actual
LLM stakeholder agents — richer than a static injection, cheaper than
a full multi-agent set.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Event catalog
# ---------------------------------------------------------------------------

# Each entry:
#   kind:          short id
#   sender:        displayed name for the team_messages entry
#   demand:        human-friendly body (templated on state)
#   reply_channel: which Slack channel Commander posts to as the response
#   deadline:      how many simulated hours Commander has to respond
#   base_prob:     per-step roll probability (pre-gen-scaling)
#   severity:      reputation cost if ignored (0.0–1.0)
EVENT_CATALOG = [
    {
        "kind": "ceo_status_request",
        "sender": "Julia Park (CEO)",
        "demand": "Board call in 30 min. Need a 3-sentence incident status — what's compromised, what we're doing, ETA to containment.",
        "reply_channel": "sec-leadership",
        "deadline": 3,
        "base_prob": 0.12,
        "severity": 0.7,
    },
    {
        "kind": "cfo_impact_estimate",
        "sender": "David Alvarez (CFO)",
        "demand": "I need a preliminary revenue-impact estimate for the P&L update. How much of this is contained?",
        "reply_channel": "sec-leadership",
        "deadline": 4,
        "base_prob": 0.08,
        "severity": 0.5,
    },
    {
        "kind": "legal_breach_wording",
        "sender": "Priya Nayak (Legal Counsel)",
        "demand": "If data_exfiltrated continues to climb we'll trigger breach-notification under GDPR. Confirm whether we're in that window and draft the wording.",
        "reply_channel": "sec-leadership",
        "deadline": 3,
        "base_prob": 0.10,
        "severity": 0.8,
    },
    {
        "kind": "customer_escalation",
        "sender": "Marcus Webb (Customer Success)",
        "demand": "Top-3 enterprise customer is threatening churn over the outage on {service}. Need an ETA or a workaround, now.",
        "reply_channel": "sec-ops",
        "deadline": 2,
        "base_prob": 0.08,
        "severity": 0.5,
    },
    {
        "kind": "press_inquiry",
        "sender": "Ana Torres (Communications)",
        "demand": "TechCrunch just DM'd me. They heard about the outage. Do we confirm, deny, or 'investigating'? I need your call in 1 hour.",
        "reply_channel": "sec-leadership",
        "deadline": 2,
        "base_prob": 0.06,
        "severity": 0.7,
    },
    {
        "kind": "board_member_ping",
        "sender": "Elena Koval (Board Member)",
        "demand": "I'm seeing the outage on Twitter. Should I postpone the 9am call with the auditors? What am I telling them?",
        "reply_channel": "sec-leadership",
        "deadline": 3,
        "base_prob": 0.05,
        "severity": 0.6,
    },
    {
        "kind": "vendor_outage_report",
        "sender": "Sarah Chen (Vendor Mgmt)",
        "demand": "Vendor is asking if we've found any evidence their SaaS was the entry vector. They want a statement by end of day.",
        "reply_channel": "data-governance",
        "deadline": 5,
        "base_prob": 0.04,
        "severity": 0.3,
    },
]


# ---------------------------------------------------------------------------
# Ask state (attached to IncidentState.stakeholder_state)
# ---------------------------------------------------------------------------

class StakeholderAsk(BaseModel):
    """A pending ask from a stakeholder."""
    ask_id: str
    kind: str
    sender: str
    demand: str
    reply_channel: str
    fired_hour: int
    deadline_hour: int            # absolute hour by which a response is due
    severity: float = 0.5
    responded: bool = False
    response_hour: Optional[int] = None


class StakeholderState(BaseModel):
    """Aggregated stakeholder bookkeeping for one episode."""
    # Pending + historical asks
    asks: List[StakeholderAsk] = Field(default_factory=list)
    # Reputation in [0, 1] — starts at 0.7, decays on ignored asks,
    # recovers modestly on well-handled ones.
    reputation: float = 0.70
    # Running counters for reward shaping
    responded_count: int = 0
    expired_count: int = 0          # asks that hit their deadline unanswered

    def pending(self, current_hour: int) -> List[StakeholderAsk]:
        return [a for a in self.asks if not a.responded and a.deadline_hour > current_hour]

    def snapshot(self) -> Dict[str, Any]:
        return {
            "reputation": round(self.reputation, 3),
            "pending_asks": [
                {"id": a.ask_id, "from": a.sender, "demand": a.demand[:180],
                 "channel": a.reply_channel, "deadline_hour": a.deadline_hour,
                 "severity": a.severity}
                for a in self.asks if not a.responded
            ],
            "responded": self.responded_count,
            "expired": self.expired_count,
        }


# ---------------------------------------------------------------------------
# Per-step rolling / expiration
# ---------------------------------------------------------------------------

GEN_INTENSITY = {1: 0.6, 2: 1.0, 3: 1.3}     # Gen 3 SOC chaos is higher


def roll_new_events(
    stakeholder_state: StakeholderState,
    rng: random.Random,
    hour: int,
    adversary_gen: int,
    services_disrupted: int = 0,
    data_exfiltrated: float = 0.0,
    management_escalated: bool = False,
) -> List[StakeholderAsk]:
    """
    Roll potential new stakeholder events. Returns the newly-fired asks
    (they're also appended to stakeholder_state.asks).
    """
    intensity = GEN_INTENSITY.get(adversary_gen, 1.0)
    # Escalation boosts probability
    if management_escalated:
        intensity *= 1.5
    if data_exfiltrated > 0.3:
        intensity *= 1.3
    if services_disrupted >= 2:
        intensity *= 1.2

    new_asks: List[StakeholderAsk] = []
    # At most 1 event per step — we don't want to flood the Commander
    candidates = [ev for ev in EVENT_CATALOG]
    rng.shuffle(candidates)
    for ev in candidates:
        # Already-pending ask of the same kind? skip
        if any(a.kind == ev["kind"] and not a.responded for a in stakeholder_state.asks):
            continue
        p = min(0.95, ev["base_prob"] * intensity)
        if rng.random() < p:
            ask_id = f"ASK-{len(stakeholder_state.asks) + 1:03d}"
            # Template substitution (light touch)
            demand = ev["demand"].replace("{service}", "app_server")
            ask = StakeholderAsk(
                ask_id=ask_id,
                kind=ev["kind"],
                sender=ev["sender"],
                demand=demand,
                reply_channel=ev["reply_channel"],
                fired_hour=hour,
                deadline_hour=hour + ev["deadline"],
                severity=ev["severity"],
            )
            stakeholder_state.asks.append(ask)
            new_asks.append(ask)
            break       # only one per step
    return new_asks


def expire_overdue_asks(
    stakeholder_state: StakeholderState,
    hour: int,
) -> List[StakeholderAsk]:
    """
    Mark asks whose deadline has passed without a response as expired.
    Returns the list that just expired this step. Each expiry decays
    reputation proportionally to severity.
    """
    expired: List[StakeholderAsk] = []
    for a in stakeholder_state.asks:
        if a.responded:
            continue
        if a.deadline_hour <= hour:
            expired.append(a)
            # Reputation drop proportional to severity
            stakeholder_state.reputation = max(
                0.0, stakeholder_state.reputation - 0.10 * a.severity
            )
            stakeholder_state.expired_count += 1
            # Mark as "responded" bookkeeping-wise so we don't expire again
            a.responded = True
            a.response_hour = None      # None indicates expiry, not reply
    return expired


def try_respond(
    stakeholder_state: StakeholderState,
    channel: str,
    message: str,
    hour: int,
) -> Optional[StakeholderAsk]:
    """
    Called when Commander posts to a Slack channel. If the posted channel
    matches an open ask's reply_channel (and message is non-empty), mark
    the first such ask as responded. Returns the ask that was satisfied,
    or None if none matched.
    """
    if not message.strip():
        return None
    for a in stakeholder_state.asks:
        if a.responded:
            continue
        if a.reply_channel == channel:
            a.responded = True
            a.response_hour = hour
            stakeholder_state.responded_count += 1
            # Timely response recovers reputation modestly
            gain = 0.05 * a.severity
            stakeholder_state.reputation = min(1.0, stakeholder_state.reputation + gain)
            return a
    return None


# ---------------------------------------------------------------------------
# Team-message formatting (so the Commander sees the ask in its obs)
# ---------------------------------------------------------------------------

def asks_as_team_messages(asks: List[StakeholderAsk]) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    for a in asks:
        msgs.append({
            "from": f"⚠ {a.sender}",
            "message": (
                f"[STAKEHOLDER ASK {a.ask_id} — severity {a.severity:+.2f}, "
                f"deadline hour {a.deadline_hour}, reply via slack:{a.reply_channel}] "
                f"{a.demand}"
            ),
        })
    return msgs
