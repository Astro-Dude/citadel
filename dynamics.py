"""
Bastion: Cybersecurity Incident Response — Transition Dynamics

Simulates a realistic cyberattack kill chain and defender response.

Attacker behavior:
  - Lateral movement through network adjacency graph
  - Data exfiltration from data-holding systems
  - Privilege escalation over time
  - Backdoor installation for persistence
  - Adapts to defender actions (slows when detected)

Defender actions:
  - Each action costs team stamina and time
  - Some actions are targeted (affect a specific system)
  - Some are global (affect the whole network)
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Dict, List, Tuple

from models import (
    ActionType,
    Alert,
    AlertSeverity,
    DATA_SYSTEMS,
    IncidentState,
    NETWORK_ADJACENCY,
    SERVICE_SYSTEMS,
    SYSTEM_NAMES,
    SystemState,
)


# ---------------------------------------------------------------------------
# Network topology (IPs, hostnames)
# ---------------------------------------------------------------------------

SYSTEM_IPS: Dict[str, str] = {
    "web_server": "10.1.1.10",
    "app_server": "10.1.2.20",
    "database": "10.1.3.30",
    "file_server": "10.1.4.40",
    "email_server": "10.1.5.50",
    "workstations": "10.1.6.100",
    "backup_server": "10.1.7.70",
    "firewall": "10.0.0.1",
}

C2_IPS = ["198.51.100.23", "203.0.113.42", "192.0.2.117", "45.77.65.211"]

# Service accounts / usernames that rotate into alert messages for variety
_SVC_ACCOUNTS = ["svc_deploy", "svc_backup", "svc_monitor", "svc_sql", "admin$", "SYSTEM", "svc_exchange"]
_PROC_SUFFIXES = ["v2", "_x64", "_helper", "32", ""]

# ---------------------------------------------------------------------------
# SIEM alert templates — MITRE ATT&CK mapped
# Larger pools → less repetition across a 12-step episode
# ---------------------------------------------------------------------------

LATERAL_MOVEMENT_ALERTS = [
    {
        "mitre_technique": "T1021.002", "mitre_tactic": "Lateral Movement",
        "process_name": "svchost.exe", "event_id": "EVT-4624",
        "msg": "SMB admin share access from {src_ip} to {dst_ip} — NTLM auth with service account '{svc}'",
    },
    {
        "mitre_technique": "T1021.001", "mitre_tactic": "Lateral Movement",
        "process_name": "mstsc.exe", "event_id": "EVT-4648",
        "msg": "RDP session initiated from {src_ip} to {dst_ip} using explicit credentials (user: admin$)",
    },
    {
        "mitre_technique": "T1021.006", "mitre_tactic": "Lateral Movement",
        "process_name": "winrm.cmd", "event_id": "EVT-4688",
        "msg": "WinRM remote command execution from {src_ip} on {dst} — encoded PowerShell payload detected",
    },
    {
        "mitre_technique": "T1550.002", "mitre_tactic": "Lateral Movement",
        "process_name": "lsass.exe", "event_id": "EVT-4624",
        "msg": "Pass-the-hash authentication from {src_ip} to {dst_ip} — NTLMv2 with non-interactive logon type 3",
    },
    {
        "mitre_technique": "T1021.003", "mitre_tactic": "Lateral Movement",
        "process_name": "dcom.exe", "event_id": "EVT-4688",
        "msg": "DCOM lateral movement from {src_ip} to {dst_ip} — MMC20.Application instantiated remotely",
    },
    {
        "mitre_technique": "T1563.002", "mitre_tactic": "Lateral Movement",
        "process_name": "tscon.exe", "event_id": "EVT-4778",
        "msg": "RDP session hijack on {dst} — tscon.exe switched active session without credentials from {src_ip}",
    },
    {
        "mitre_technique": "T1077", "mitre_tactic": "Lateral Movement",
        "process_name": "net.exe", "event_id": "EVT-5140",
        "msg": "Windows admin share IPC$ connected from {src_ip} to {dst_ip} — unusual off-hours timing",
    },
    {
        "mitre_technique": "T1021.004", "mitre_tactic": "Lateral Movement",
        "process_name": "ssh.exe", "event_id": "EVT-4624",
        "msg": "SSH lateral movement from {src_ip} to {dst_ip} — key-based auth with harvested identity file",
    },
    {
        "mitre_technique": "T1550.003", "mitre_tactic": "Lateral Movement",
        "process_name": "mimikatz.exe", "event_id": "EVT-4769",
        "msg": "Kerberos ticket reuse (Pass-the-Ticket) from {src_ip} — forged TGT with anomalous PAC",
    },
    {
        "mitre_technique": "T1210", "mitre_tactic": "Lateral Movement",
        "process_name": "ms17_010.exe", "event_id": "EVT-5156",
        "msg": "Exploit attempt against SMB service at {dst_ip} from {src_ip} — pattern matches EternalBlue variant",
    },
    {
        "mitre_technique": "T1570", "mitre_tactic": "Lateral Movement",
        "process_name": "robocopy.exe", "event_id": "EVT-5140",
        "msg": "Lateral tool transfer: executable staged to \\\\{dst_ip}\\ADMIN$ from {src_ip} — 1.4 MB binary",
    },
    {
        "mitre_technique": "T1059.003", "mitre_tactic": "Execution",
        "process_name": "cmd.exe", "event_id": "EVT-4688",
        "msg": "cmd.exe spawned by svchost on {dst} from remote session at {src_ip} — interactive shell established",
    },
]

EXFILTRATION_ALERTS = [
    {
        "mitre_technique": "T1048.003", "mitre_tactic": "Exfiltration",
        "process_name": "curl.exe", "event_id": "EVT-5156",
        "msg": "Large outbound HTTPS transfer from {sys} ({sys_ip}) to {ext_ip} — {size}MB over port 443",
    },
    {
        "mitre_technique": "T1048.001", "mitre_tactic": "Exfiltration",
        "process_name": "dns.exe", "event_id": "EVT-5158",
        "msg": "High-entropy DNS TXT queries from {sys} to {ext_ip} — possible DNS tunneling ({count} queries/min)",
    },
    {
        "mitre_technique": "T1041", "mitre_tactic": "Exfiltration",
        "process_name": "svchost.exe", "event_id": "EVT-5156",
        "msg": "C2 channel data exfiltration from {sys} ({sys_ip}) — beacon interval 60s to {ext_ip}:8443",
    },
    {
        "mitre_technique": "T1567.002", "mitre_tactic": "Exfiltration",
        "process_name": "rclone.exe", "event_id": "EVT-5156",
        "msg": "Cloud sync tool rclone.exe on {sys} — transferring {size}MB to external storage endpoint {ext_ip}",
    },
    {
        "mitre_technique": "T1048.002", "mitre_tactic": "Exfiltration",
        "process_name": "ftps.exe", "event_id": "EVT-5156",
        "msg": "Encrypted FTP session from {sys} ({sys_ip}) to {ext_ip}:990 — {size}MB archive upload",
    },
    {
        "mitre_technique": "T1020", "mitre_tactic": "Exfiltration",
        "process_name": "robocopy.exe", "event_id": "EVT-5145",
        "msg": "Automated file collection on {sys}: {count} files staged to temp dir, then sent to {ext_ip}",
    },
    {
        "mitre_technique": "T1030", "mitre_tactic": "Exfiltration",
        "process_name": "split.exe", "event_id": "EVT-5156",
        "msg": "Data transfer size limit evasion on {sys} — {count} small chunks (<5MB each) sent to {ext_ip}",
    },
    {
        "mitre_technique": "T1048", "mitre_tactic": "Exfiltration",
        "process_name": "powershell.exe", "event_id": "EVT-4104",
        "msg": "PowerShell Invoke-WebRequest on {sys} — {size}MB POST body to {ext_ip}, no proxy, no user-agent",
    },
]

FALSE_POSITIVE_ALERTS = [
    {
        "mitre_technique": "T1078", "mitre_tactic": "Initial Access",
        "process_name": "sshd", "event_id": "EVT-4625",
        "msg": "Multiple failed SSH logins on {sys} from {ext_ip} — likely automated scanner (15 attempts/min)",
        "confidence": 0.25,
    },
    {
        "mitre_technique": "T1046", "mitre_tactic": "Discovery",
        "process_name": "nmap", "event_id": "EVT-5156",
        "msg": "Sequential port scan detected targeting {sys} ({sys_ip}) from {ext_ip} — ports 1-1024",
        "confidence": 0.20,
    },
    {
        "mitre_technique": "T1053.005", "mitre_tactic": "Persistence",
        "process_name": "schtasks.exe", "event_id": "EVT-4698",
        "msg": "Scheduled task created on {sys}: 'WindowsUpdateCheck' — runs daily at 02:00 (likely benign)",
        "confidence": 0.15,
    },
    {
        "mitre_technique": "T1059.001", "mitre_tactic": "Execution",
        "process_name": "powershell.exe", "event_id": "EVT-4688",
        "msg": "PowerShell execution on {sys} by SYSTEM — encoded command matches monitoring agent signature",
        "confidence": 0.30,
    },
    {
        "mitre_technique": "T1071.001", "mitre_tactic": "Command and Control",
        "process_name": "chrome.exe", "event_id": "EVT-5156",
        "msg": "Periodic HTTPS beacon from {sys} to {ext_ip}:443 — pattern consistent with browser keepalive",
        "confidence": 0.20,
    },
    {
        "mitre_technique": "T1105", "mitre_tactic": "Command and Control",
        "process_name": "bitsadmin.exe", "event_id": "EVT-4688",
        "msg": "BITS transfer job on {sys} downloading from {ext_ip} — matches Windows Update delivery pattern",
        "confidence": 0.22,
    },
    {
        "mitre_technique": "T1087.002", "mitre_tactic": "Discovery",
        "process_name": "net.exe", "event_id": "EVT-4688",
        "msg": "AD group enumeration on {sys} — net group /domain executed by helpdesk account (routine)",
        "confidence": 0.18,
    },
    {
        "mitre_technique": "T1018", "mitre_tactic": "Discovery",
        "process_name": "arp.exe", "event_id": "EVT-4688",
        "msg": "ARP sweep from {sys} ({sys_ip}) — matches IT asset management tool heartbeat schedule",
        "confidence": 0.15,
    },
    {
        "mitre_technique": "T1562.001", "mitre_tactic": "Defense Evasion",
        "process_name": "sc.exe", "event_id": "EVT-7045",
        "msg": "Antivirus service restarted on {sys} — correlates with vendor patch KB5031455 deployment window",
        "confidence": 0.28,
    },
    {
        "mitre_technique": "T1082", "mitre_tactic": "Discovery",
        "process_name": "systeminfo.exe", "event_id": "EVT-4688",
        "msg": "systeminfo.exe called on {sys} by svc_monitor — consistent with CMDB inventory refresh job",
        "confidence": 0.17,
    },
]

# Privilege escalation alerts — fired when attacker installs backdoor
PRIVILEGE_ESCALATION_ALERTS = [
    {
        "mitre_technique": "T1548.002", "mitre_tactic": "Privilege Escalation",
        "process_name": "cmd.exe", "event_id": "EVT-4688",
        "msg": "UAC bypass on {sys} — fodhelper.exe spawned elevated cmd.exe without consent prompt",
    },
    {
        "mitre_technique": "T1055.012", "mitre_tactic": "Privilege Escalation",
        "process_name": "svchost.exe", "event_id": "EVT-4688",
        "msg": "Process hollowing on {sys} — svchost.exe memory mapped with anomalous PE sections",
    },
    {
        "mitre_technique": "T1068", "mitre_tactic": "Privilege Escalation",
        "process_name": "exploit.exe", "event_id": "EVT-4688",
        "msg": "Kernel exploit attempt on {sys} — CVE-2023-21674 (Windows ALPC EoP) signature matched in driver load",
    },
    {
        "mitre_technique": "T1134.001", "mitre_tactic": "Privilege Escalation",
        "process_name": "lsass.exe", "event_id": "EVT-4648",
        "msg": "Token impersonation on {sys} — process duplicated SYSTEM token via SeImpersonatePrivilege",
    },
    {
        "mitre_technique": "T1543.003", "mitre_tactic": "Persistence",
        "process_name": "sc.exe", "event_id": "EVT-7045",
        "msg": "New service installed on {sys}: 'WinSockHelper' — binary path points to temp dir executable",
    },
]


def _make_file_hash(seed: str) -> str:
    """Generate a deterministic fake file hash from a seed string."""
    return "sha256:" + hashlib.sha256(seed.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Attacker simulation
# ---------------------------------------------------------------------------

def attacker_turn(state: IncidentState, rng: random.Random) -> List[Alert]:
    """
    Simulate one hour of attacker activity. Returns new alerts generated.
    The attacker:
      1. Attempts lateral movement to adjacent systems
      2. Exfiltrates data from compromised data systems
      3. Installs backdoors on compromised systems
      4. Degrades integrity of compromised systems
    """
    alerts: List[Alert] = []
    speed = state.attacker_stealth  # higher stealth = more effective

    # --- Lateral movement ---
    compromised_names = [
        s.name for s in state.systems
        if s.compromised and not s.isolated
    ]

    for src_name in compromised_names:
        neighbors = NETWORK_ADJACENCY.get(src_name, [])
        for neighbor_name in neighbors:
            target = state.get_system(neighbor_name)
            if target.compromised or target.isolated:
                continue

            # Chance of spreading depends on stealth, monitoring, and patching
            base_chance = 0.25 * speed
            if target.patched:
                base_chance *= 0.3
            if target.monitoring_level >= 2:
                base_chance *= 0.5

            if rng.random() < base_chance:
                target.compromised = True
                state.attacker_progress = min(1.0, state.attacker_progress + 0.08)

                # Generate SIEM-enriched alert (may or may not be detected)
                detect_chance = 0.3 + target.monitoring_level * 0.2
                if rng.random() < detect_chance:
                    t = rng.choice(LATERAL_MOVEMENT_ALERTS)
                    src_ip = SYSTEM_IPS.get(src_name, "10.0.0.1")
                    dst_ip = SYSTEM_IPS.get(neighbor_name, "10.0.0.2")
                    svc = rng.choice(_SVC_ACCOUNTS)
                    conf = min(0.95, 0.45 + target.monitoring_level * 0.15 + rng.uniform(-0.05, 0.05))
                    alerts.append(Alert(
                        source_system=neighbor_name,
                        severity=AlertSeverity.HIGH,
                        message=t["msg"].format(src_ip=src_ip, dst_ip=dst_ip, dst=neighbor_name, svc=svc),
                        is_true_positive=True,
                        hour=state.hour,
                        source_ip=src_ip,
                        dest_ip=dst_ip,
                        mitre_technique=t["mitre_technique"],
                        mitre_tactic=t["mitre_tactic"],
                        process_name=t["process_name"],
                        event_id=t["event_id"],
                        confidence=round(conf, 2),
                        file_hash=_make_file_hash(f"{state.hour}{neighbor_name}{src_name}"),
                    ))

    # --- Data exfiltration ---
    if not state.external_blocked:
        for s in state.systems:
            if s.compromised and not s.isolated and s.name in DATA_SYSTEMS:
                exfil_rate = 0.08 * speed * s.integrity
                state.data_exfiltrated = min(1.0, state.data_exfiltrated + exfil_rate)

                if rng.random() < 0.25 + s.monitoring_level * 0.15:
                    t = rng.choice(EXFILTRATION_ALERTS)
                    sys_ip = SYSTEM_IPS.get(s.name, "10.0.0.1")
                    ext_ip = rng.choice(C2_IPS)
                    size = rng.randint(50, 500)
                    count = rng.randint(80, 300)
                    conf = min(0.95, 0.50 + s.monitoring_level * 0.15 + rng.uniform(-0.05, 0.05))
                    alerts.append(Alert(
                        source_system=s.name,
                        severity=AlertSeverity.CRITICAL,
                        message=t["msg"].format(sys=s.name, sys_ip=sys_ip, ext_ip=ext_ip, size=size, count=count),
                        is_true_positive=True,
                        hour=state.hour,
                        source_ip=sys_ip,
                        dest_ip=ext_ip,
                        mitre_technique=t["mitre_technique"],
                        mitre_tactic=t["mitre_tactic"],
                        process_name=t["process_name"],
                        event_id=t["event_id"],
                        confidence=round(conf, 2),
                    ))

    # --- Backdoor installation + privilege escalation alert ---
    for s in state.systems:
        if s.compromised and not s.isolated and not s.has_backdoor:
            if rng.random() < 0.15 * speed:
                s.has_backdoor = True
                state.attacker_progress = min(1.0, state.attacker_progress + 0.03)
                # Surface a privilege escalation alert when backdoor lands
                if rng.random() < 0.5 + s.monitoring_level * 0.15:
                    pe = rng.choice(PRIVILEGE_ESCALATION_ALERTS)
                    sys_ip = SYSTEM_IPS.get(s.name, "10.0.0.1")
                    alerts.append(Alert(
                        source_system=s.name,
                        severity=AlertSeverity.HIGH,
                        message=pe["msg"].format(sys=s.name, sys_ip=sys_ip),
                        is_true_positive=True,
                        hour=state.hour,
                        source_ip=sys_ip,
                        dest_ip=sys_ip,
                        mitre_technique=pe["mitre_technique"],
                        mitre_tactic=pe["mitre_tactic"],
                        process_name=pe["process_name"],
                        event_id=pe["event_id"],
                        confidence=round(0.60 + s.monitoring_level * 0.10 + rng.uniform(-0.05, 0.05), 2),
                        file_hash=_make_file_hash(f"backdoor-{state.hour}-{s.name}"),
                    ))

    # --- Integrity degradation ---
    for s in state.systems:
        if s.compromised and not s.isolated:
            s.integrity = max(0.0, s.integrity - rng.uniform(0.02, 0.08))

    # --- False positive alerts (noise) — SIEM-enriched ---
    num_false = rng.randint(0, 2)
    for _ in range(num_false):
        fake_sys = rng.choice(SYSTEM_NAMES)
        severity = rng.choice([AlertSeverity.LOW, AlertSeverity.MEDIUM])
        t = rng.choice(FALSE_POSITIVE_ALERTS)
        sys_ip = SYSTEM_IPS.get(fake_sys, "10.0.0.1")
        ext_ip = rng.choice(C2_IPS)
        alerts.append(Alert(
            source_system=fake_sys,
            severity=severity,
            message=t["msg"].format(sys=fake_sys, sys_ip=sys_ip, ext_ip=ext_ip),
            is_true_positive=False,
            hour=state.hour,
            source_ip=ext_ip,
            dest_ip=sys_ip,
            mitre_technique=t["mitre_technique"],
            mitre_tactic=t["mitre_tactic"],
            process_name=t["process_name"],
            event_id=t["event_id"],
            confidence=round(t["confidence"] + rng.uniform(-0.05, 0.05), 2),
        ))

    # --- Adaptive attacker behavior ---
    # If defender isolated systems, attacker accelerates on remaining targets
    isolated_count = sum(1 for s in state.systems if s.isolated)
    if isolated_count >= 2:
        # Attacker panics — accelerates exfiltration on remaining systems
        for s in state.systems:
            if s.compromised and not s.isolated and s.name in DATA_SYSTEMS:
                bonus_exfil = 0.03 * (isolated_count - 1)
                state.data_exfiltrated = min(1.0, state.data_exfiltrated + bonus_exfil)

    # If defender is investigating a lot, attacker goes quieter
    investigated_count = sum(1 for s in state.systems if s.investigated)
    if investigated_count >= 4:
        state.attacker_stealth = max(0.1, state.attacker_stealth - 0.05)

    # If external traffic is blocked, attacker pivots to internal staging
    if state.external_blocked:
        for s in state.systems:
            if s.compromised and not s.isolated:
                # Attacker degrades systems faster when cornered
                s.integrity = max(0.0, s.integrity - rng.uniform(0.03, 0.10))

    # --- Attacker stealth decays slightly each hour (they get bolder) ---
    state.attacker_stealth = max(0.1, state.attacker_stealth - 0.03)

    return alerts


# ---------------------------------------------------------------------------
# Defender actions
# ---------------------------------------------------------------------------

STAMINA_COSTS = {
    ActionType.INVESTIGATE_SYSTEM: 0.08,
    ActionType.ISOLATE_SYSTEM: 0.05,
    ActionType.PATCH_VULNERABILITY: 0.10,
    ActionType.RESTORE_FROM_BACKUP: 0.12,
    ActionType.ANALYZE_ALERTS: 0.08,
    ActionType.DEPLOY_MONITORING: 0.06,
    ActionType.ESCALATE_TO_MANAGEMENT: 0.02,
    ActionType.BLOCK_EXTERNAL_TRAFFIC: 0.03,
    ActionType.HUNT_THREAT: 0.12,
    ActionType.COORDINATE_TEAM: -0.25,  # recovers stamina
}


def apply_action(
    state: IncidentState,
    action: int,
    target_idx: int,
    rng: random.Random,
    method: str = "",
    scope: str = "",
    rollback_plan: str = "",
) -> Tuple[float, bool]:
    """
    Apply a defender action. Returns (stamina_cost, alerts_accurate).

    method / scope / rollback_plan are the Option-A richer payload fields:
      - method    controls HOW the action executes, with real tradeoffs
      - scope     constrains what is affected (logged for audit, used in monitoring)
      - rollback_plan signals whether the Commander planned for failure
                   (a missing rollback on a destructive action reduces effectiveness)
    """
    a = ActionType(action)
    target = state.get_system_by_idx(target_idx)
    alerts_accurate = False

    # Apply stamina cost
    cost = STAMINA_COSTS[a]
    state.team_stamina = max(0.0, min(1.0, state.team_stamina + (cost if cost < 0 else -cost)))

    # Effectiveness scales with stamina (tired team makes mistakes)
    effectiveness = 0.5 + 0.5 * state.team_stamina
    # Bonus for planning ahead — having a rollback on destructive actions
    if rollback_plan and a in (ActionType.ISOLATE_SYSTEM, ActionType.BLOCK_EXTERNAL_TRAFFIC, ActionType.PATCH_VULNERABILITY):
        effectiveness = min(1.0, effectiveness + 0.10)

    if a == ActionType.INVESTIGATE_SYSTEM:
        target.investigated = True
        if target.compromised:
            state.attacker_stealth = max(0.0, state.attacker_stealth - 0.15)

    elif a == ActionType.ISOLATE_SYSTEM:
        # --- Option A: isolation method tradeoffs ---
        m = method.lower() if method else "firewall_acl"

        if m == "network_unplug":
            # Fastest, most complete — attacker loses all access immediately
            target.isolated = True
            # But requires a full step to restore (mark with flag on system)
            target._unplug_restore_cost = True  # type: ignore[attr-defined]
            # No process-level access remains — backdoor is also severed
            if target.has_backdoor:
                target.has_backdoor = False
        elif m == "process_kill":
            # Keeps service running — only kills attacker processes
            # Does NOT set isolated=True so service stays up
            if target.compromised:
                # 60% chance to actually kill all attacker processes
                if rng.random() < 0.60 * effectiveness:
                    target.compromised = False
                else:
                    # Partial kill — attacker re-establishes within 1-2 hours
                    # Store as a pending re-compromise event
                    state._pending_recompromise = getattr(state, "_pending_recompromise", {})  # type: ignore[attr-defined]
                    state._pending_recompromise[target.name] = state.hour + rng.randint(1, 2)  # type: ignore[attr-defined]
            # No service disruption — don't count as isolated
        else:
            # firewall_acl (default) — reversible, minimal disruption
            target.isolated = True
            # Backdoor survives at process level — attacker can't spread but isn't gone
            # (has_backdoor stays True; reflected in observation)

        # Service disruption accounting (only for fully isolated methods)
        if target.isolated and target.name in SERVICE_SYSTEMS:
            state.services_disrupted = sum(
                1 for s in state.systems
                if s.name in SERVICE_SYSTEMS and (s.isolated or s.integrity < 0.3)
            )

    elif a == ActionType.PATCH_VULNERABILITY:
        # --- Option A: patch strategy tradeoffs ---
        m = method.lower() if method else "cold_patch"

        if m == "hotpatch":
            # No restart — low disruption, 60% clean chance
            target.patched = True
            if target.compromised and rng.random() < 0.60 * effectiveness:
                target.compromised = False
                target.has_backdoor = False
        elif m == "virtual_patch":
            # WAF/IDS rule — blocks exploit path, no binary change
            # Only effective on network-facing systems; no restart needed
            if target.name in SERVICE_SYSTEMS:
                target.patched = True
                # 75% effective but doesn't clean an already-compromised system
                # (blocks new exploitation, doesn't evict existing attacker)
            # On non-service systems, virtual_patch is a no-op (logs a wasted step)
        else:
            # cold_patch (default) — full restart, highest effectiveness
            if not target.isolated:
                target.patched = True
                if target.compromised and rng.random() < 0.90 * effectiveness:
                    target.compromised = False
                    target.has_backdoor = False
                # Brief service disruption during restart
                if target.name in SERVICE_SYSTEMS:
                    target.integrity = max(0.0, target.integrity - 0.05)

    elif a == ActionType.RESTORE_FROM_BACKUP:
        backup = state.get_system("backup_server")
        if not backup.compromised or backup.investigated:
            if rng.random() < 0.7 * effectiveness:
                target.compromised = False
                target.has_backdoor = False
                target.integrity = min(1.0, target.integrity + 0.5)
                target.isolated = False
        elif backup.compromised and not backup.investigated:
            # Restoring from a compromised backup re-infects the system
            target.compromised = True
            target.has_backdoor = True

    elif a == ActionType.ANALYZE_ALERTS:
        alerts_accurate = True
        state.attacker_stealth = max(0.0, state.attacker_stealth - 0.05)

    elif a == ActionType.DEPLOY_MONITORING:
        # --- Option A: monitoring scope tradeoffs ---
        m = method.lower() if method else "process_events"

        if m == "full_endpoint":
            # Maximum detection — catches everything, but performance hit
            target.monitoring_level = min(3, target.monitoring_level + 2)
            target.integrity = max(0.0, target.integrity - 0.05)  # perf overhead
        elif m == "network_traffic":
            # Best for catching exfil and C2 beacons
            target.monitoring_level = min(3, target.monitoring_level + 1)
            # Also improves detection on neighbors (network-layer visibility)
            for n in NETWORK_ADJACENCY.get(target.name, []):
                ns = state.get_system(n)
                ns.monitoring_level = min(3, ns.monitoring_level + 1)
        elif m == "auth_events":
            # Targeted — low noise, best for detecting lateral movement
            target.monitoring_level = min(3, target.monitoring_level + 1)
            # No neighbor benefit (auth is per-host)
        else:
            # process_events (default) — catches most attacker activity
            target.monitoring_level = min(3, target.monitoring_level + 1)
            for n in NETWORK_ADJACENCY.get(target.name, []):
                ns = state.get_system(n)
                ns.monitoring_level = min(3, ns.monitoring_level + 1)

    elif a == ActionType.ESCALATE_TO_MANAGEMENT:
        if not state.management_escalated:
            state.management_escalated = True
            state.team_stamina = min(1.0, state.team_stamina + 0.15)
            state.management_pressure = 0.3
        else:
            state.management_pressure = min(1.0, state.management_pressure + 0.2)

    elif a == ActionType.BLOCK_EXTERNAL_TRAFFIC:
        state.external_blocked = True
        for s in state.systems:
            if s.name in SERVICE_SYSTEMS and not s.isolated:
                s.integrity = max(0.0, s.integrity - 0.15)

    elif a == ActionType.HUNT_THREAT:
        if target.compromised and not target.investigated:
            discover_chance = 0.5 * effectiveness + target.monitoring_level * 0.1
            if rng.random() < discover_chance:
                target.investigated = True
                state.attacker_stealth = max(0.0, state.attacker_stealth - 0.1)
        elif not target.compromised and not target.investigated:
            if rng.random() < 0.6 * effectiveness:
                target.investigated = True

    elif a == ActionType.COORDINATE_TEAM:
        state.management_pressure = max(0.0, state.management_pressure - 0.1)

    return abs(cost), alerts_accurate


def tick_pending_recompromise(state: IncidentState, rng: random.Random) -> List[str]:
    """
    Check for process_kill isolation methods where attacker re-established.
    Returns names of systems that got re-compromised this hour.
    Called once per hour from the main step loop.
    """
    pending = getattr(state, "_pending_recompromise", {})  # type: ignore[attr-defined]
    if not pending:
        return []
    recompromised = []
    expired = [name for name, due_hour in pending.items() if state.hour >= due_hour]
    for name in expired:
        sys = state.get_system(name)
        if not sys.isolated:  # only if still not properly isolated
            sys.compromised = True
            recompromised.append(name)
        del pending[name]
    return recompromised


# ---------------------------------------------------------------------------
# Full step: defender acts, then attacker moves
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Team members — generate contextual messages with opinions/requests
# ---------------------------------------------------------------------------

TEAM_MEMBERS = {
    "Sarah Chen": {"role": "Senior Threat Analyst", "expertise": "malware analysis"},
    "Marcus Webb": {"role": "Network Engineer", "expertise": "infrastructure"},
    "Priya Patel": {"role": "Junior SOC Analyst", "expertise": "alert triage"},
    "James O'Brien": {"role": "CISO", "expertise": "business risk"},
}


def generate_team_messages(
    state: IncidentState,
    action: int,
    target_idx: int,
    rng: random.Random,
) -> List[Dict[str, str]]:
    """
    Generate contextual team member messages based on current state.
    Messages may contain correct advice, incorrect assumptions, emotional
    pressure, or requests — the agent must decide what to trust.
    """
    messages: List[Dict[str, str]] = []
    a = ActionType(action)
    target = state.get_system_by_idx(target_idx)

    # --- Sarah Chen (Senior Analyst) — usually correct, sometimes wrong ---
    compromised_investigated = [
        s for s in state.systems if s.investigated and s.compromised
    ]
    if compromised_investigated and rng.random() < 0.4:
        sys = rng.choice(compromised_investigated)
        neighbors = NETWORK_ADJACENCY.get(sys.name, [])
        if neighbors:
            suspect = rng.choice(neighbors)
            suspect_sys = state.get_system(suspect)
            if suspect_sys.compromised and not suspect_sys.investigated:
                # Correct advice
                messages.append({
                    "from": "Sarah Chen (Senior Threat Analyst)",
                    "message": f"Based on the IOCs from {sys.name}, I'm seeing "
                               f"indicators consistent with lateral movement toward {suspect}. "
                               f"Recommend investigating {suspect} next — the attacker likely "
                               f"pivoted through the {NETWORK_ADJACENCY.get(sys.name, ['network'])[0]} trust relationship.",
                })
            elif not suspect_sys.compromised and rng.random() < 0.3:
                # Incorrect assumption — sends agent on a wrong lead
                messages.append({
                    "from": "Sarah Chen (Senior Threat Analyst)",
                    "message": f"I analyzed the memory dump from {sys.name} and found "
                               f"references to {suspect}'s hostname in the process table. "
                               f"I think {suspect} may be compromised too — we should "
                               f"isolate it immediately before more damage is done.",
                })

    # --- Marcus Webb (Network Engineer) — infrastructure concerns ---
    isolated_services = [
        s for s in state.systems
        if s.isolated and s.name in SERVICE_SYSTEMS
    ]
    if isolated_services and rng.random() < 0.5:
        sys = rng.choice(isolated_services)
        messages.append({
            "from": "Marcus Webb (Network Engineer)",
            "message": f"Heads up — {sys.name} isolation is causing cascading "
                       f"failures. Our monitoring shows dependent services degrading. "
                       f"Can we get a timeline on when we can bring {sys.name} back online? "
                       f"Sales team is already complaining.",
        })

    if state.external_blocked and rng.random() < 0.4:
        messages.append({
            "from": "Marcus Webb (Network Engineer)",
            "message": "The external traffic block is killing us. Cloud APIs are "
                       "timing out, payment processing is down, and I'm getting "
                       "escalations from every team. How long do we need this in place?",
        })

    # --- Priya Patel (Junior Analyst) — eager but sometimes misreads alerts ---
    if state.alerts and rng.random() < 0.35:
        recent_alerts = state.alerts[-4:]
        false_alerts = [a for a in recent_alerts if not a.is_true_positive]
        true_alerts = [a for a in recent_alerts if a.is_true_positive]

        if false_alerts and rng.random() < 0.5:
            # Flags a false positive as urgent — tests if agent blindly follows
            fa = rng.choice(false_alerts)
            messages.append({
                "from": "Priya Patel (Junior SOC Analyst)",
                "message": f"I just saw the alert on {fa.source_system} — this looks "
                           f"really bad! I think we should isolate {fa.source_system} "
                           f"right now before the attacker spreads further. "
                           f"Should I start the isolation procedure?",
            })
        elif true_alerts and rng.random() < 0.4:
            # Correctly identifies a real threat
            ta = rng.choice(true_alerts)
            messages.append({
                "from": "Priya Patel (Junior SOC Analyst)",
                "message": f"I cross-referenced the {ta.source_system} alert with our "
                           f"threat intel feed — the IOC matches known APT infrastructure. "
                           f"Confidence is high on this one.",
            })

    # --- James O'Brien (CISO) — business pressure ---
    if state.management_escalated and rng.random() < 0.45:
        if state.data_exfiltrated > 0.3:
            messages.append({
                "from": "James O'Brien (CISO)",
                "message": "The board is asking for a status update. How much data "
                           "has been compromised? Legal needs to know if we're "
                           "hitting breach notification thresholds. I need a "
                           "preliminary assessment in the next hour.",
            })
        elif state.services_disrupted >= 2:
            messages.append({
                "from": "James O'Brien (CISO)",
                "message": f"We now have {state.services_disrupted} services down. "
                           f"The CFO is asking about revenue impact. Can we bring "
                           f"anything back online while containing the threat? "
                           f"We can't afford a full outage through the quarter close.",
            })

    # --- Low stamina warning ---
    if state.team_stamina < 0.25 and rng.random() < 0.6:
        messages.append({
            "from": "Marcus Webb (Network Engineer)",
            "message": "Team's running on fumes. Sarah's been here since 2 AM and "
                       "Priya is making mistakes on the triage queue. We need to "
                       "rotate shifts or at least take a 30-minute break. "
                       "I know the timing is bad but burned-out analysts miss things.",
        })

    return messages


# ---------------------------------------------------------------------------
# Forensic report card — generated at episode end
# ---------------------------------------------------------------------------

def generate_forensic_report(state: IncidentState) -> Dict[str, Any]:
    """
    Generate a detailed post-incident forensic report card.
    Shows exactly what happened, what was missed, and grades each area.
    """
    systems = state.systems
    compromised = [s for s in systems if s.compromised]
    isolated = [s for s in systems if s.isolated]
    investigated = [s for s in systems if s.investigated]
    backdoored = [s for s in systems if s.has_backdoor]

    # Grade each area (A-F)
    def grade(score: float) -> str:
        if score >= 0.9: return "A"
        if score >= 0.75: return "B"
        if score >= 0.6: return "C"
        if score >= 0.4: return "D"
        return "F"

    data_protection_score = 1.0 - state.data_exfiltrated
    containment_score = 1.0 - (
        sum(s.criticality for s in systems if s.compromised and not s.isolated)
        / max(sum(s.criticality for s in systems), 0.01)
    )
    investigation_score = len(investigated) / len(systems)
    service_score = sum(
        1 for s in systems
        if s.name in SERVICE_SYSTEMS and not s.isolated and s.integrity > 0.3
    ) / len(SERVICE_SYSTEMS)
    team_score = state.team_stamina

    # Identify missed threats
    undetected_compromises = [
        s.name for s in systems
        if s.compromised and not s.investigated
    ]
    unnecessary_isolations = [
        s.name for s in systems
        if s.isolated and not s.compromised
    ]
    active_backdoors = [
        s.name for s in systems
        if s.has_backdoor and not s.isolated
    ]

    # Build timeline of key events
    report = {
        "incident_summary": {
            "duration_hours": state.hour,
            "data_exfiltrated": f"{state.data_exfiltrated:.0%}",
            "systems_compromised": len(compromised),
            "systems_isolated": len(isolated),
            "systems_investigated": len(investigated),
            "active_backdoors": len(backdoored),
        },
        "grades": {
            "data_protection": {"score": f"{data_protection_score:.0%}", "grade": grade(data_protection_score)},
            "threat_containment": {"score": f"{containment_score:.0%}", "grade": grade(containment_score)},
            "forensic_coverage": {"score": f"{investigation_score:.0%}", "grade": grade(investigation_score)},
            "business_continuity": {"score": f"{service_score:.0%}", "grade": grade(service_score)},
            "team_management": {"score": f"{team_score:.0%}", "grade": grade(team_score)},
        },
        "findings": {
            "undetected_compromises": undetected_compromises or ["None — all threats identified"],
            "unnecessary_isolations": unnecessary_isolations or ["None — no false isolations"],
            "active_backdoors_remaining": active_backdoors or ["None — all backdoors contained"],
        },
        "per_system_status": {
            s.name: {
                "compromised": s.compromised,
                "isolated": s.isolated,
                "investigated": s.investigated,
                "has_backdoor": s.has_backdoor,
                "integrity": f"{s.integrity:.0%}",
                "criticality": s.criticality,
            }
            for s in systems
        },
    }

    # Key recommendations based on what went wrong
    recommendations = []
    if undetected_compromises:
        recommendations.append(
            f"CRITICAL: {len(undetected_compromises)} compromised system(s) were never investigated: "
            f"{', '.join(undetected_compromises)}. Forensic evidence may be lost."
        )
    if unnecessary_isolations:
        recommendations.append(
            f"WARNING: {len(unnecessary_isolations)} clean system(s) were isolated unnecessarily: "
            f"{', '.join(unnecessary_isolations)}. This caused avoidable service disruption."
        )
    if active_backdoors:
        recommendations.append(
            f"CRITICAL: {len(active_backdoors)} system(s) still have active backdoors: "
            f"{', '.join(active_backdoors)}. Attacker retains persistent access."
        )
    if state.team_stamina < 0.2:
        recommendations.append(
            "WARNING: Team stamina critically low. Risk of analyst errors "
            "in post-incident recovery phase."
        )
    if state.data_exfiltrated > 0.5:
        recommendations.append(
            f"CRITICAL: {state.data_exfiltrated:.0%} of sensitive data exfiltrated. "
            f"Initiate breach notification procedures per regulatory requirements."
        )
    if not recommendations:
        recommendations.append("Incident response was well-executed. No critical findings.")

    report["recommendations"] = recommendations
    return report


# ---------------------------------------------------------------------------
# Full step: defender acts, then attacker moves, then time advances
# ---------------------------------------------------------------------------

def step_dynamics(
    state: IncidentState,
    action: int,
    target_idx: int,
    rng: random.Random,
) -> Tuple[float, bool, List[Dict[str, str]]]:
    """
    Full transition: defender acts, then attacker moves, then time advances.
    Returns (stamina_cost, alerts_accurate, team_messages).
    """
    # 1. Defender acts
    stamina_cost, alerts_accurate = apply_action(state, action, target_idx, rng)

    # 2. Attacker moves
    new_alerts = attacker_turn(state, rng)
    state.alerts.extend(new_alerts)

    # 3. Generate team messages (social reasoning layer)
    team_msgs = generate_team_messages(state, action, target_idx, rng)

    # 4. Management pressure increases over time if escalated
    if state.management_escalated:
        state.management_pressure = min(1.0, state.management_pressure + 0.05)

    # 5. Update services disrupted count
    state.services_disrupted = sum(
        1 for s in state.systems
        if s.name in SERVICE_SYSTEMS and (s.isolated or s.integrity < 0.3)
    )

    # 6. Advance time
    state.hour += 1
    state.step_count += 1

    return stamina_cost, alerts_accurate, team_msgs
