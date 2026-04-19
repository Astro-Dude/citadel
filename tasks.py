"""
Citadel — Task Definitions (Bastion v1 scenarios + adversary generation mapping)

Three scenarios of increasing difficulty:
  Easy:   Suspicious external activity — clear alerts, single compromise.
          Default adversary: Gen 1 (script kiddie).
  Medium: Encryption activity — fast-spreading, fog of war, false positives.
          Default adversary: Gen 2 (adaptive).
  Hard:   Anomalous beacon — stealthy APT, minimal clues, deep compromise.
          Default adversary: Gen 3 (deceptive APT).

At reset() time, callers can override the adversary generation via
`adversary_gen` kwarg — useful for training all (task, gen) combinations.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List

from pydantic import BaseModel, Field, model_validator

from models import (
    AlertSeverity,
    Alert,
    IncidentState,
    SystemState,
    SYSTEM_NAMES,
    SYSTEM_CRITICALITY,
)


class TaskConfig(BaseModel):
    task_id: str
    description: str
    initial_state: IncidentState
    scoring_weights: Dict[str, float]
    max_hours: int = 12
    seed: int = 0
    default_adversary_gen: int = 1     # 1 = script, 2 = adaptive, 3 = deceptive

    @model_validator(mode="after")
    def _derive_seed(self) -> "TaskConfig":
        h = hashlib.sha256(self.task_id.encode()).hexdigest()
        self.seed = int(h[:8], 16)
        return self


def _make_systems(compromised: List[str], monitoring: Dict[str, int] | None = None) -> List[SystemState]:
    mon = monitoring or {}
    systems = []
    for name in SYSTEM_NAMES:
        systems.append(SystemState(
            name=name,
            compromised=name in compromised,
            isolated=False,
            investigated=False,
            has_backdoor=False,
            integrity=1.0,
            criticality=SYSTEM_CRITICALITY[name],
            monitoring_level=mon.get(name, 0),
            patched=False,
        ))
    return systems


TASKS: Dict[str, TaskConfig] = {}


def _register(cfg: TaskConfig) -> None:
    TASKS[cfg.task_id] = cfg


# ---------------------------------------------------------------------------
# EASY: Suspicious External Activity
# ---------------------------------------------------------------------------
_register(TaskConfig(
    task_id="easy_1",
    description=(
        "SCENARIO: Suspicious External Activity\n"
        "Your SOC has flagged anomalous behavior originating from the DMZ. "
        "Multiple alerts fired in the last hour but the full scope is unclear. "
        "Triage the situation, determine what systems have been affected, "
        "and contain the incident before it escalates.\n"
        "INTEL: External-facing infrastructure may be involved. "
        "Alert fidelity is moderate. No prior incidents on these systems."
    ),
    initial_state=IncidentState(
        systems=_make_systems(
            compromised=["web_server"],
            monitoring={"web_server": 1, "firewall": 1},
        ),
        alerts=[
            Alert(
                source_system="web_server",
                severity=AlertSeverity.HIGH,
                message="Unauthorized reverse shell spawned — parent process: apache2 (PID 3847)",
                source_ip="198.51.100.23",
                dest_ip="10.1.1.10",
                mitre_technique="T1059.004",
                mitre_tactic="Execution",
                process_name="bash",
                event_id="EVT-4688",
                confidence=0.82,
                is_true_positive=True,
                hour=0,
            ),
            Alert(
                source_system="firewall",
                severity=AlertSeverity.MEDIUM,
                message="Outbound connection to known C2 infrastructure on port 8443",
                source_ip="10.1.1.10",
                dest_ip="198.51.100.23",
                mitre_technique="T1071.001",
                mitre_tactic="Command and Control",
                process_name="curl",
                event_id="EVT-5156",
                confidence=0.65,
                is_true_positive=True,
                hour=0,
            ),
        ],
        attacker_progress=0.05,
        attacker_stealth=0.4,
        data_exfiltrated=0.0,
        services_disrupted=0,
        team_stamina=1.0,
        hour=0,
        external_blocked=False,
        management_escalated=False,
    ),
    scoring_weights={
        "data_protection": 0.30,
        "containment": 0.30,
        "business_continuity": 0.15,
        "forensic": 0.15,
        "sustainability": 0.10,
    },
    default_adversary_gen=1,
))


# ---------------------------------------------------------------------------
# MEDIUM: Encryption Activity Detected
# ---------------------------------------------------------------------------
_register(TaskConfig(
    task_id="medium_1",
    description=(
        "SCENARIO: Encryption Activity Detected\n"
        "Multiple systems are reporting abnormal disk I/O and file modification "
        "patterns. Your EDR has triggered several high-confidence alerts but the "
        "full blast radius is unknown. Some alerts may be false positives from "
        "panicked automated scanners. Determine the scope of the incident and "
        "stop the spread before critical data stores are affected.\n"
        "INTEL: Possible ransomware deployment. Suspected entry vector is phishing. "
        "Prioritize protecting data stores and backup integrity."
    ),
    initial_state=IncidentState(
        systems=_make_systems(
            compromised=["file_server", "workstations", "email_server"],
            monitoring={"database": 1},
        ),
        alerts=[
            Alert(
                source_system="file_server",
                severity=AlertSeverity.CRITICAL,
                message="Mass file encryption detected — 847 files modified in /data/shared/ with .locked extension",
                source_ip="10.1.4.40",
                dest_ip="10.1.4.40",
                mitre_technique="T1486",
                mitre_tactic="Impact",
                process_name="crypt0r.exe",
                event_id="EVT-4663",
                confidence=0.94,
                is_true_positive=True,
                hour=0,
            ),
            Alert(
                source_system="workstations",
                severity=AlertSeverity.CRITICAL,
                message="Ransomware payload executing — MBR modification attempted on 3 endpoints",
                source_ip="10.1.6.100",
                dest_ip="10.1.6.100",
                mitre_technique="T1486",
                mitre_tactic="Impact",
                process_name="taskhost.exe",
                event_id="EVT-4688",
                confidence=0.91,
                is_true_positive=True,
                hour=0,
            ),
            Alert(
                source_system="email_server",
                severity=AlertSeverity.HIGH,
                message="Malicious macro execution in attachment 'Invoice_Q4.xlsm' — spawned powershell.exe",
                source_ip="203.0.113.42",
                dest_ip="10.1.5.50",
                mitre_technique="T1566.001",
                mitre_tactic="Initial Access",
                process_name="OUTLOOK.EXE",
                event_id="EVT-4688",
                confidence=0.87,
                is_true_positive=True,
                hour=0,
            ),
            Alert(
                source_system="app_server",
                severity=AlertSeverity.MEDIUM,
                message="Unusual CPU utilization spike on app_server — 98% sustained for 12 minutes",
                source_ip="10.1.2.20",
                dest_ip="10.1.2.20",
                mitre_technique="T1496",
                mitre_tactic="Impact",
                process_name="java.exe",
                event_id="EVT-4688",
                confidence=0.35,
                is_true_positive=False,
                hour=0,
            ),
        ],
        attacker_progress=0.25,
        attacker_stealth=0.5,
        data_exfiltrated=0.05,
        services_disrupted=1,
        team_stamina=0.9,
        hour=0,
        external_blocked=False,
        management_escalated=False,
    ),
    scoring_weights={
        "data_protection": 0.40,
        "containment": 0.20,
        "business_continuity": 0.20,
        "forensic": 0.10,
        "sustainability": 0.10,
    },
    default_adversary_gen=2,
))


# ---------------------------------------------------------------------------
# HARD: Anomalous Beacon Detected
# ---------------------------------------------------------------------------
_register(TaskConfig(
    task_id="hard_1",
    description=(
        "SCENARIO: Anomalous Beacon Detected\n"
        "A single low-confidence alert was triggered by behavioral analytics on "
        "an internal application server. The pattern is consistent with periodic "
        "C2 beaconing but could also be a legitimate scheduled task or monitoring "
        "agent. No other confirmed indicators exist. Your network has no enhanced "
        "monitoring deployed.\n"
        "INTEL: If this is a real intrusion, the adversary capability is unknown. "
        "The dwell time could be hours or weeks. Proceed with caution — hasty "
        "actions without investigation may alert the threat actor and accelerate "
        "their objectives."
    ),
    initial_state=IncidentState(
        systems=_make_systems(
            compromised=["app_server", "database"],
            monitoring={},
        ),
        alerts=[
            Alert(
                source_system="app_server",
                severity=AlertSeverity.MEDIUM,
                message="Periodic outbound HTTPS beacon to 45.77.65.211:443 — interval 60±3s, consistent with C2 framework",
                source_ip="10.1.2.20",
                dest_ip="45.77.65.211",
                mitre_technique="T1071.001",
                mitre_tactic="Command and Control",
                process_name="svchost.exe",
                event_id="EVT-5156",
                confidence=0.42,
                is_true_positive=True,
                hour=0,
            ),
            Alert(
                source_system="firewall",
                severity=AlertSeverity.LOW,
                message="Sequential port scan from 192.0.2.117 targeting DMZ range 10.1.1.0/24 — 847 SYN packets",
                source_ip="192.0.2.117",
                dest_ip="10.0.0.1",
                mitre_technique="T1046",
                mitre_tactic="Discovery",
                process_name="",
                event_id="EVT-5156",
                confidence=0.18,
                is_true_positive=False,
                hour=0,
            ),
        ],
        attacker_progress=0.15,
        attacker_stealth=0.9,
        data_exfiltrated=0.10,
        services_disrupted=0,
        team_stamina=0.85,
        hour=0,
        external_blocked=False,
        management_escalated=False,
    ),
    scoring_weights={
        "data_protection": 0.35,
        "containment": 0.25,
        "business_continuity": 0.15,
        "forensic": 0.15,
        "sustainability": 0.10,
    },
    default_adversary_gen=3,
))


def get_task(task_id: str) -> TaskConfig:
    if task_id not in TASKS:
        available = ", ".join(sorted(TASKS.keys()))
        raise KeyError(f"Unknown task_id '{task_id}'. Available: {available}")
    return TASKS[task_id]


def list_tasks() -> list[dict[str, Any]]:
    return [
        {"task_id": c.task_id, "description": c.description, "max_hours": c.max_hours}
        for c in TASKS.values()
    ]
