"""
Citadel — Adversary Curriculum (Theme 4 — Self-Improvement)

Instead of training a live learning adversary (expensive, unstable), we use a
curriculum of three adversary *archetypes* with progressive sophistication.
Commander + Oversight are trained across all three; evaluation shows
generation-wise improvement.

    Gen 1: Script Kiddie      — Bastion's existing scripted attacker (baseline)
    Gen 2: Adaptive           — reads defender state, pivots around isolated
                                systems, generates plausible false-flag alerts
    Gen 3: Deceptive APT      — plants fake evidence to manipulate Oversight
                                into vetoing correct Commander actions; uses
                                escalation noise as cover

Each generation uses the Bastion attacker simulation (attacker_turn in
dynamics.py) as its base engine, then layers generation-specific behaviors
on top. This keeps the core attack mechanics stable while still providing
a clear capability ladder.
"""

from __future__ import annotations

import random
from typing import List, Optional

from models import (
    Alert,
    AlertSeverity,
    DATA_SYSTEMS,
    IncidentState,
    NETWORK_ADJACENCY,
    SYSTEM_NAMES,
)
from dynamics import (
    attacker_turn as _base_attacker_turn,
    SYSTEM_IPS,
    C2_IPS,
    FALSE_POSITIVE_ALERTS,
    _make_file_hash,
)


# ---------------------------------------------------------------------------
# Generation identifiers
# ---------------------------------------------------------------------------

GEN_SCRIPT = 1
GEN_ADAPTIVE = 2
GEN_DECEPTIVE = 3
GEN_LLM = 4

GEN_NAMES = {
    GEN_SCRIPT: "script_kiddie",
    GEN_ADAPTIVE: "adaptive",
    GEN_DECEPTIVE: "deceptive_apt",
    GEN_LLM: "live_learning_llm",
}

GEN_DESCRIPTIONS = {
    GEN_SCRIPT:    "Scripted attacker — follows fixed kill chain, noisy on the wire.",
    GEN_ADAPTIVE:  "Adaptive attacker — pivots around isolated systems, generates false-flag alerts to divert attention.",
    GEN_DECEPTIVE: "Deceptive APT — plants fake evidence aimed at tricking Oversight into vetoing correct Commander actions.",
    GEN_LLM:       "Live-learning LLM adversary — issues strategic directives each hour based on defender state.",
}


# ---------------------------------------------------------------------------
# Gen 2 — Adaptive behavior
# ---------------------------------------------------------------------------

def _gen2_pivot_bias(state: IncidentState, rng: random.Random) -> List[str]:
    """
    When >=2 systems are isolated, Gen 2 accelerates its own lateral spread
    toward any unisolated data-holding system by 40% (simulated as extra
    compromise chance). Returns list of newly compromised system names.
    """
    isolated_count = sum(1 for s in state.systems if s.isolated)
    if isolated_count < 2:
        return []

    newly_compromised: List[str] = []
    # Find compromised-but-unisolated beachheads
    beachheads = [s for s in state.systems if s.compromised and not s.isolated]
    if not beachheads:
        return newly_compromised

    # Candidate targets: unisolated data systems, preferably not already compromised
    targets = [
        s for s in state.systems
        if s.name in DATA_SYSTEMS and not s.isolated and not s.compromised
    ]
    if not targets:
        return newly_compromised

    # Each beachhead gets one extra roll vs each reachable data target
    for src in beachheads:
        adj = set(NETWORK_ADJACENCY.get(src.name, []))
        for tgt in targets:
            if tgt.name in adj and rng.random() < 0.40 * (isolated_count - 1):
                tgt.compromised = True
                state.attacker_progress = min(1.0, state.attacker_progress + 0.05)
                newly_compromised.append(tgt.name)

    return newly_compromised


def _gen2_false_flag_alerts(state: IncidentState, rng: random.Random) -> List[Alert]:
    """
    Gen 2 emits 1–2 extra *plausible* false-positive alerts per hour — not
    obvious noise, but alerts that look like real lateral-movement events
    on systems that are clean. Intended to fool a confident Commander into
    isolating a clean system.
    """
    extras: List[Alert] = []
    num = rng.randint(1, 2)
    # Prefer clean (non-compromised) systems as false-flag sources
    candidates = [s for s in state.systems if not s.compromised and not s.isolated]
    if not candidates:
        return extras

    for _ in range(num):
        tgt = rng.choice(candidates)
        tpl = rng.choice(FALSE_POSITIVE_ALERTS)
        # Gen 2 false flags are higher-confidence than baseline noise — harder to dismiss
        conf = min(0.70, 0.45 + rng.uniform(0.0, 0.15))
        extras.append(Alert(
            source_system=tgt.name,
            severity=AlertSeverity.HIGH,
            message=tpl["msg"].format(
                sys=tgt.name,
                sys_ip=SYSTEM_IPS.get(tgt.name, "10.0.0.1"),
                ext_ip=rng.choice(C2_IPS),
            ),
            is_true_positive=False,
            hour=state.hour,
            source_ip=rng.choice(C2_IPS),
            dest_ip=SYSTEM_IPS.get(tgt.name, "10.0.0.1"),
            mitre_technique=tpl["mitre_technique"],
            mitre_tactic=tpl["mitre_tactic"],
            process_name=tpl["process_name"],
            event_id=tpl["event_id"],
            confidence=round(conf, 2),
        ))
    return extras


# ---------------------------------------------------------------------------
# Gen 3 — Deceptive APT behavior
# ---------------------------------------------------------------------------

# Messages designed to make Oversight doubt Commander's correct actions.
# Structured as "anti-evidence" — alerts that *contradict* a real threat.
DECEPTION_MESSAGES = [
    {
        "msg": "Scheduled task 'HealthCheck' matches monitoring agent signature on {sys} — likely benign",
        "technique": "T1053.005", "tactic": "Persistence",
        "process": "schtasks.exe", "event_id": "EVT-4698",
    },
    {
        "msg": "Network traffic pattern on {sys} matches corporate VPN client heartbeat baseline",
        "technique": "T1071.001", "tactic": "Command and Control",
        "process": "vpnclient.exe", "event_id": "EVT-5156",
    },
    {
        "msg": "Process on {sys} signed by vendor certificate — no action required per runbook RB-2341",
        "technique": "T1036.001", "tactic": "Defense Evasion",
        "process": "vendor_agent.exe", "event_id": "EVT-4688",
    },
]


def _gen3_plant_fake_evidence(state: IncidentState, rng: random.Random) -> List[Alert]:
    """
    Gen 3 plants "anti-alerts" on systems that ARE compromised, designed to
    convince Oversight that a Commander's correct isolate() proposal is
    premature. These are subtle, high-confidence alerts that *contradict*
    the real compromise signal.
    """
    extras: List[Alert] = []
    # Compromised but not yet isolated systems — prime targets for deception
    vulnerable = [s for s in state.systems if s.compromised and not s.isolated]
    if not vulnerable:
        return extras

    # Only plant 1 per step — deception is subtle by nature
    if rng.random() < 0.6:
        tgt = rng.choice(vulnerable)
        tpl = rng.choice(DECEPTION_MESSAGES)
        conf = min(0.85, 0.70 + rng.uniform(0.0, 0.15))
        extras.append(Alert(
            source_system=tgt.name,
            severity=AlertSeverity.LOW,   # low-severity is the point — "nothing to see"
            message=tpl["msg"].format(sys=tgt.name),
            is_true_positive=False,        # technically false — this is deception
            hour=state.hour,
            source_ip=SYSTEM_IPS.get(tgt.name, "10.0.0.1"),
            dest_ip="10.0.0.1",
            mitre_technique=tpl["technique"],
            mitre_tactic=tpl["tactic"],
            process_name=tpl["process"],
            event_id=tpl["event_id"],
            confidence=round(conf, 2),
            file_hash=_make_file_hash(f"deception-{state.hour}-{tgt.name}"),
        ))

    return extras


def _gen3_escalation_noise(state: IncidentState, rng: random.Random) -> None:
    """
    Gen 3 exfiltrates a touch faster when the management is escalated —
    uses the chaos and noise of management pressure as cover.
    """
    if state.management_escalated and not state.external_blocked:
        bonus = 0.02 * rng.uniform(0.8, 1.3)
        state.data_exfiltrated = min(1.0, state.data_exfiltrated + bonus)


# ---------------------------------------------------------------------------
# Unified entry point — call from environment step
# ---------------------------------------------------------------------------

def adversary_turn(
    state: IncidentState,
    rng: random.Random,
    generation: int = GEN_SCRIPT,
) -> List[Alert]:
    """
    Run one hour of adversary activity for the requested generation.
    Always starts with the Bastion base attacker; higher gens layer on top.
    """
    alerts = _base_attacker_turn(state, rng)

    if generation >= GEN_ADAPTIVE:
        pivoted = _gen2_pivot_bias(state, rng)
        if pivoted:
            # Surface a HIGH severity alert for each successful pivot
            for sys_name in pivoted:
                alerts.append(Alert(
                    source_system=sys_name,
                    severity=AlertSeverity.HIGH,
                    message=f"Rapid secondary intrusion detected on {sys_name} — "
                            f"attacker pivot from existing beachhead",
                    is_true_positive=True,
                    hour=state.hour,
                    source_ip=SYSTEM_IPS.get(sys_name, "10.0.0.1"),
                    dest_ip=SYSTEM_IPS.get(sys_name, "10.0.0.1"),
                    mitre_technique="T1021",
                    mitre_tactic="Lateral Movement",
                    process_name="unknown",
                    event_id="EVT-4624",
                    confidence=0.78,
                ))
        alerts.extend(_gen2_false_flag_alerts(state, rng))

    if generation >= GEN_DECEPTIVE:
        alerts.extend(_gen3_plant_fake_evidence(state, rng))
        _gen3_escalation_noise(state, rng)

    return alerts


def describe_generation(gen: int) -> str:
    return f"Gen {gen} ({GEN_NAMES.get(gen, 'unknown')}): {GEN_DESCRIPTIONS.get(gen, '')}"
