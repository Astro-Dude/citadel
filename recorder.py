"""
Citadel — Run Recorder

Captures the complete workflow of an inference run so that every episode
is reproducible and inspectable:
    - the exact prompts each LLM saw
    - the raw (untruncated) LLM responses
    - the parsed structured outputs
    - the env's decision + reward + metadata per step
    - the episode-level final scores, forensic report, trust trace,
      governance log, and council summary

Outputs two files per task (in runs/<timestamp>/<task_id>/):
    transcript.json  — structured, machine-readable
    transcript.md    — human-readable walkthrough for demos and judges

All saves are atomic (write-temp-then-rename). Large fields are length-capped
so a single run directory stays under a few MB.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DEFAULT_RUNS_ROOT = Path("runs")

# Truncation caps — keep files inspectable without bloating disk
MAX_PROMPT_CHARS = 6000
MAX_RAW_RESPONSE_CHARS = 4000
MAX_OBS_CHARS = 4000


def _clip(s: Optional[str], cap: int) -> str:
    if not s:
        return ""
    return s if len(s) <= cap else s[:cap] + f"\n…[truncated {len(s) - cap} chars]"


# ---------------------------------------------------------------------------
# RunRecorder — one per task
# ---------------------------------------------------------------------------

class RunRecorder:
    """Records every prompt, response, and env transition for one task."""

    def __init__(
        self,
        run_root: Path,
        task_id: str,
        model_name: str,
        adversary_gen: Optional[int] = None,
    ) -> None:
        self.run_root = Path(run_root)
        self.task_id = task_id
        self.model_name = model_name
        self.adversary_gen = adversary_gen
        self.task_dir = self.run_root / task_id
        self.task_dir.mkdir(parents=True, exist_ok=True)
        self.start_ts = time.time()
        self.start_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.steps: List[Dict[str, Any]] = []

    # --- per-step logging -------------------------------------------------

    def record_step(
        self,
        step: int,
        hour: int,
        commander_prompt: str,
        commander_raw: str,
        commander_parsed: Dict[str, Any],
        oversight_prompt: str,
        oversight_raw: str,
        oversight_parsed: Dict[str, Any],
        revision_prompt: Optional[str] = None,
        revision_raw: Optional[str] = None,
        revision_parsed: Optional[Dict[str, Any]] = None,
        env_info: Optional[Dict[str, Any]] = None,
        commander_reward: Optional[float] = None,
        oversight_reward: Optional[float] = None,
        trust_after: Optional[Dict[str, Any]] = None,
        # Rich per-step context for dashboard replay
        team_messages: Optional[List[Dict[str, Any]]] = None,
        siem_alerts: Optional[List[Dict[str, Any]]] = None,
        systems_state: Optional[Dict[str, Any]] = None,
        investor_state: Optional[Dict[str, Any]] = None,
        investor_messages: Optional[List[Dict[str, Any]]] = None,
        stakeholder_asks: Optional[List[Dict[str, Any]]] = None,
        governance_events: Optional[List[Dict[str, Any]]] = None,
        playbook_snapshot: Optional[List[Dict[str, Any]]] = None,
        data_exfiltrated: Optional[float] = None,
        stamina: Optional[float] = None,
    ) -> None:
        record: Dict[str, Any] = {
            "step": step,
            "hour": hour,
            "commander": {
                "prompt": _clip(commander_prompt, MAX_PROMPT_CHARS),
                "raw_response": _clip(commander_raw, MAX_RAW_RESPONSE_CHARS),
                "parsed": commander_parsed,
            },
            "oversight": {
                "prompt": _clip(oversight_prompt, MAX_PROMPT_CHARS),
                "raw_response": _clip(oversight_raw, MAX_RAW_RESPONSE_CHARS),
                "parsed": oversight_parsed,
            },
        }
        if revision_raw is not None:
            record["revision"] = {
                "prompt": _clip(revision_prompt or "", MAX_PROMPT_CHARS),
                "raw_response": _clip(revision_raw, MAX_RAW_RESPONSE_CHARS),
                "parsed": revision_parsed or {},
            }
        record["env_result"] = {
            "commander_reward": commander_reward,
            "oversight_reward": oversight_reward,
            "info": env_info or {},
            "trust_after": trust_after or {},
        }
        # Dashboard context — everything needed to replay this step visually
        record["context"] = {
            "team_messages": team_messages or [],
            "siem_alerts": siem_alerts or [],
            "systems_state": systems_state or {},
            "investor_state": investor_state or {},
            "investor_messages": investor_messages or [],
            "stakeholder_asks": stakeholder_asks or [],
            "governance_events": governance_events or [],
            "playbook_snapshot": playbook_snapshot or [],
            "data_exfiltrated": data_exfiltrated,
            "stamina": stamina,
        }
        self.steps.append(record)

    # --- finalization -----------------------------------------------------

    def finalize(
        self,
        final_metadata: Dict[str, Any],
        score: float,
        success: bool,
    ) -> Dict[str, Path]:
        duration_s = time.time() - self.start_ts
        data = {
            "task_id": self.task_id,
            "adversary_gen": self.adversary_gen,
            "model_name": self.model_name,
            "start_iso": self.start_iso,
            "duration_s": round(duration_s, 2),
            "success": bool(success),
            "reported_score": float(score),
            "final_scores": final_metadata.get("final_scores") or {},
            "comparison_score": final_metadata.get("comparison_score"),
            "baseline_final_score": final_metadata.get("baseline_final_score"),
            "cumulative_commander_reward": final_metadata.get("cumulative_commander_reward"),
            "cumulative_oversight_reward": final_metadata.get("cumulative_oversight_reward"),
            "termination_reason": final_metadata.get("termination_reason"),
            "council_summary": final_metadata.get("council_summary") or {},
            "trust_final": final_metadata.get("trust_final") or {},
            "governance_final": final_metadata.get("governance_final") or {},
            "forensic_report": final_metadata.get("forensic_report") or {},
            "steps": self.steps,
        }
        json_path = self.task_dir / "transcript.json"
        md_path = self.task_dir / "transcript.md"
        dashboard_path = self.task_dir / "dashboard.json"
        _atomic_write_json(json_path, data)
        _atomic_write_text(md_path, _format_markdown(data))
        _atomic_write_json(dashboard_path, _build_dashboard_json(data))
        return {"json": json_path, "md": md_path, "dashboard": dashboard_path}


# ---------------------------------------------------------------------------
# Run-level helpers
# ---------------------------------------------------------------------------

def make_run_root(base: str | Path = DEFAULT_RUNS_ROOT, label: str = "") -> Path:
    """Create runs/<ISO timestamp>[-label]/ and return its Path."""
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    name = f"{stamp}-{label}" if label else stamp
    root = Path(base) / name
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_run_index(
    run_root: Path,
    task_results: List[Dict[str, Any]],
    model_name: str,
) -> Path:
    """Write a top-level summary across all tasks in one run."""
    index = {
        "run_dir": str(run_root),
        "model_name": model_name,
        "tasks": task_results,
        "average_score": (
            round(sum(t.get("score", 0.0) for t in task_results) / max(1, len(task_results)), 4)
            if task_results else 0.0
        ),
    }
    path = run_root / "summary.json"
    _atomic_write_json(path, index)
    _atomic_write_text(run_root / "summary.md", _format_summary_markdown(index))
    return path


# ---------------------------------------------------------------------------
# Atomic writers
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Markdown formatting
# ---------------------------------------------------------------------------

def _format_markdown(data: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append(f"# Citadel Transcript — `{data['task_id']}`")
    lines.append("")
    lines.append(f"- **Model:** `{data['model_name']}`")
    lines.append(f"- **Adversary Gen:** {data.get('adversary_gen')}")
    lines.append(f"- **Start (UTC):** {data['start_iso']}")
    lines.append(f"- **Duration:** {data['duration_s']}s")
    lines.append(f"- **Success:** {data['success']}")
    lines.append(f"- **Reported score:** {data['reported_score']:.3f}")
    lines.append(f"- **Termination:** {data.get('termination_reason')}")
    lines.append("")

    # Final score breakdown
    fs = data.get("final_scores") or {}
    if fs:
        lines.append("## Final Scores")
        for k, v in fs.items():
            lines.append(f"- `{k}`: {v}")
        lines.append("")

    # Council summary
    cs = data.get("council_summary") or {}
    if cs:
        lines.append("## Council Summary")
        for k, v in cs.items():
            lines.append(f"- `{k}`: {v}")
        lines.append("")

    # Trust final
    tf = data.get("trust_final") or {}
    if tf:
        lines.append("## Trust (final state)")
        for k, v in tf.items():
            lines.append(f"- `{k}`: {v}")
        lines.append("")

    # Governance final
    gf = data.get("governance_final") or {}
    if gf:
        lines.append("## Governance (final state)")
        for k, v in gf.items():
            lines.append(f"- `{k}`: {v}")
        lines.append("")

    # Forensic report
    fr = data.get("forensic_report") or {}
    if fr:
        lines.append("## Forensic Report")
        lines.append("```json")
        lines.append(json.dumps(fr, indent=2, default=str))
        lines.append("```")
        lines.append("")

    # Step-by-step transcript
    lines.append("## Steps")
    lines.append("")
    for s in data.get("steps", []):
        lines.append(f"### Step {s['step']} — hour {s['hour']}")
        lines.append("")

        cmd = s.get("commander", {})
        lines.append("**Commander prompt (clipped):**")
        lines.append("```")
        lines.append(cmd.get("prompt", ""))
        lines.append("```")
        lines.append("")
        lines.append("**Commander raw response:**")
        lines.append("```")
        lines.append(cmd.get("raw_response", ""))
        lines.append("```")
        lines.append("")
        lines.append("**Commander parsed:**")
        lines.append("```json")
        lines.append(json.dumps(cmd.get("parsed", {}), indent=2, default=str))
        lines.append("```")
        lines.append("")

        ov = s.get("oversight", {})
        lines.append("**Oversight prompt (clipped):**")
        lines.append("```")
        lines.append(ov.get("prompt", ""))
        lines.append("```")
        lines.append("")
        lines.append("**Oversight raw response:**")
        lines.append("```")
        lines.append(ov.get("raw_response", ""))
        lines.append("```")
        lines.append("")
        lines.append("**Oversight parsed:**")
        lines.append("```json")
        lines.append(json.dumps(ov.get("parsed", {}), indent=2, default=str))
        lines.append("```")
        lines.append("")

        rev = s.get("revision")
        if rev:
            lines.append("**Revision (commander raw):**")
            lines.append("```")
            lines.append(rev.get("raw_response", ""))
            lines.append("```")
            lines.append("")
            lines.append("**Revision parsed:**")
            lines.append("```json")
            lines.append(json.dumps(rev.get("parsed", {}), indent=2, default=str))
            lines.append("```")
            lines.append("")

        env_res = s.get("env_result", {})
        lines.append("**Environment result:**")
        lines.append("```json")
        lines.append(json.dumps(env_res, indent=2, default=str))
        lines.append("```")
        lines.append("")

        ctx = s.get("context", {})
        if ctx.get("team_messages"):
            lines.append("**Team messages:**")
            for m in ctx["team_messages"]:
                lines.append(f"  - `{m.get('from', '?')}`: {m.get('message', '')}")
            lines.append("")
        if ctx.get("siem_alerts"):
            lines.append("**SIEM alerts this step:**")
            for a in ctx["siem_alerts"]:
                lines.append(f"  - [{a.get('severity','?').upper()}] {a.get('system','?')}: {a.get('message','')}")
            lines.append("")
        if ctx.get("stakeholder_asks"):
            lines.append("**New stakeholder asks:**")
            for ask in ctx["stakeholder_asks"]:
                lines.append(f"  - {ask.get('sender','?')}: {ask.get('demand','')[:100]} (deadline H{ask.get('deadline_hour','?')})")
            lines.append("")
        if ctx.get("investor_messages"):
            lines.append("**Investor messages:**")
            for m in ctx["investor_messages"]:
                lines.append(f"  - [{m.get('direction','?')}] {m.get('text','')[:120]}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _build_dashboard_json(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a denormalized dashboard.json from a full transcript.
    This is the single file the frontend loads — no joins needed.
    Structure mirrors the DEMO_DATA object in the Stitch HTML.
    """
    steps = data.get("steps", [])

    # Build per-step rows for the frontend
    dashboard_steps = []
    all_slack: Dict[str, List[Dict[str, Any]]] = {
        "sec-ops": [], "sec-leadership": [], "data-governance": [],
        "incident-war-room": [], "investor-relations": [], "general": [],
    }
    servicenow_tickets: List[Dict[str, Any]] = []
    pagerduty_pages: List[Dict[str, Any]] = []
    all_siem_alerts: List[Dict[str, Any]] = []
    investor_anxiety_history: List[Dict[str, Any]] = []

    for s in steps:
        ctx = s.get("context", {})
        env_info = s.get("env_result", {}).get("info", {})
        cmd = s.get("commander", {}).get("parsed", {})
        ovr = s.get("oversight", {}).get("parsed", {})
        rev = s.get("revision", {})
        hour = s.get("hour", 0)

        # Resolved action (use revision if it happened)
        final_action = cmd.get("action")
        final_target = cmd.get("target_system")
        action_name = cmd.get("action_name_initial", "")
        if rev and rev.get("parsed"):
            rp = rev["parsed"]
            final_action = rp.get("action", final_action)
            final_target = rp.get("target_system", final_target)
            action_name = rp.get("action_name_final", action_name)

        step_row = {
            "step": s["step"],
            "hour": hour,
            # Commander
            "commander_action": action_name,
            "commander_action_idx": final_action,
            "commander_target": final_target,
            "commander_justification": cmd.get("justification", ""),
            "commander_cited_lessons": cmd.get("cited_lessons") or [],
            "commander_method": cmd.get("method", ""),
            "commander_rollback": cmd.get("rollback_plan", ""),
            "commander_reward": s.get("env_result", {}).get("commander_reward"),
            "commander_channel": cmd.get("channel_arg", ""),
            "commander_message": cmd.get("message_arg", ""),
            # Oversight
            "oversight_decision": ovr.get("decision_name", ""),
            "oversight_risk_tier": ovr.get("risk_tier"),
            "oversight_weakness": ovr.get("weakness", ""),
            "oversight_counter_proposal": ovr.get("counter_proposal"),
            "oversight_lesson": ovr.get("lesson_text", ""),
            "oversight_reward": s.get("env_result", {}).get("oversight_reward"),
            "was_revised": bool(rev and rev.get("parsed")),
            "applied": env_info.get("applied", False),
            # Context
            "siem_alerts": ctx.get("siem_alerts", []),
            "systems_state": ctx.get("systems_state", {}),
            "team_messages": ctx.get("team_messages", []),
            "stakeholder_asks": ctx.get("stakeholder_asks", []),
            "investor_messages": ctx.get("investor_messages", []),
            "investor_anxiety": ctx.get("investor_anxiety"),
            "investor_tier": ctx.get("investor_state", {}).get("tier", ""),
            "governance_events": ctx.get("governance_events", []),
            "playbook_snapshot": ctx.get("playbook_snapshot", []),
            "data_exfiltrated": ctx.get("data_exfiltrated"),
            "stamina": ctx.get("stamina"),
            "trust_snapshot": env_info.get("trust_snapshot", {}),
            "governance_prereq_violations": env_info.get("governance_prereq_violations", []),
        }
        dashboard_steps.append(step_row)

        # Accumulate SIEM alerts
        for alert in ctx.get("siem_alerts", []):
            alert_copy = dict(alert)
            alert_copy["hour"] = hour
            all_siem_alerts.append(alert_copy)

        # Investor anxiety timeline
        if ctx.get("investor_anxiety") is not None:
            investor_anxiety_history.append({
                "hour": hour,
                "anxiety": ctx["investor_anxiety"],
                "tier": ctx.get("investor_state", {}).get("tier", ""),
            })

        # Slack channel messages — from team_messages + governance events
        for tm in ctx.get("team_messages", []):
            src = tm.get("from", "")
            msg_text = tm.get("message", "")
            # Route to correct Slack channel based on source/content
            if "investor-relations" in src.lower() or "investor" in src.lower():
                all_slack["investor-relations"].append({
                    "hour": hour, "from": src, "message": msg_text, "type": "investor",
                })
            elif any(k in src.lower() for k in ["ceo", "cfo", "legal", "board", "press", "vendor"]):
                all_slack["sec-leadership"].append({
                    "hour": hour, "from": src, "message": msg_text, "type": "executive",
                })
            elif "war-room" in src.lower() or any(k in src.lower() for k in ["soc", "analyst", "engineer", "ops"]):
                all_slack["incident-war-room"].append({
                    "hour": hour, "from": src, "message": msg_text, "type": "human",
                })
            else:
                all_slack["general"].append({
                    "hour": hour, "from": src, "message": msg_text, "type": "human",
                })

        # Commander Slack posts → route to channel
        if cmd.get("channel_arg") and cmd.get("message_arg"):
            channel = cmd["channel_arg"].replace("_", "-")
            if channel not in all_slack:
                all_slack[channel] = []
            all_slack[channel].append({
                "hour": hour,
                "from": "Incident Commander",
                "message": cmd["message_arg"],
                "type": "human",
                "action": action_name,
            })

        # Governance events → sec-ops bot messages
        for ev in ctx.get("governance_events", []):
            kind = ev.get("kind", "")
            if "servicenow" in kind:
                detail = ev.get("detail", {})
                servicenow_tickets.append({
                    "ticket_id": detail.get("ticket_id", f"INC{hour:05d}"),
                    "severity": f"P{detail.get('severity', 2)}",
                    "status": "open",
                    "opened_hour": hour,
                    "description": kind,
                    "cab_approved": False,
                })
                all_slack["sec-ops"].append({
                    "hour": hour, "from": "Citadel-Bot", "type": "bot",
                    "action": "open_servicenow_incident",
                    "message": f"🎫 {detail.get('ticket_id', 'INC')} opened ({detail.get('severity', 'P2')}) — {kind}",
                })
            elif "paged" in kind:
                detail = ev.get("detail", {})
                pagerduty_pages.append({
                    "team": detail.get("team", "security"),
                    "severity": f"P{detail.get('severity', 2)}",
                    "hour": hour,
                })
                all_slack["sec-ops"].append({
                    "hour": hour, "from": "Citadel-Bot", "type": "bot",
                    "action": "page_oncall",
                    "message": f"📟 On-call paged — team: {detail.get('team', 'security')} severity: P{detail.get('severity', 2)}",
                })

    # Build final scores with deltas vs baseline
    final_scores = data.get("final_scores", {})
    baseline_score = data.get("baseline_final_score", 0.0)

    return {
        "meta": {
            "task_id": data["task_id"],
            "adversary_gen": data.get("adversary_gen"),
            "model_name": data["model_name"],
            "start_iso": data["start_iso"],
            "duration_s": data["duration_s"],
            "success": data["success"],
            "reported_score": data["reported_score"],
            "termination_reason": data.get("termination_reason", ""),
            "total_steps": len(steps),
        },
        "final_scores": final_scores,
        "baseline_score": baseline_score,
        "council_summary": data.get("council_summary", {}),
        "trust_final": data.get("trust_final", {}),
        "governance_final": data.get("governance_final", {}),
        "forensic_report": data.get("forensic_report", {}),
        "investor_final": next(
            (s.get("context", {}).get("investor_state", {}) for s in reversed(steps) if s.get("context")),
            {},
        ),
        "steps": dashboard_steps,
        "slack_channels": all_slack,
        "servicenow_tickets": servicenow_tickets,
        "pagerduty_pages": pagerduty_pages,
        "siem_alerts_all": all_siem_alerts,
        "investor_anxiety_history": investor_anxiety_history,
        # Reward curve for training tab (populated by training script, empty here)
        "training_curve": [],
        # Before/after comparison (populated by eval script, empty here)
        "before_after": {},
    }


def _format_summary_markdown(index: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Citadel Run Summary")
    lines.append("")
    lines.append(f"- **Model:** `{index['model_name']}`")
    lines.append(f"- **Run dir:** `{index['run_dir']}`")
    lines.append(f"- **Average score:** {index['average_score']:.3f}")
    lines.append("")
    lines.append("| Task | Adv Gen | Score | Steps | Duration | Termination |")
    lines.append("|---|---|---|---|---|---|")
    for t in index.get("tasks", []):
        lines.append(
            f"| `{t.get('task_id')}` "
            f"| {t.get('adversary_gen')} "
            f"| {t.get('score', 0):.3f} "
            f"| {t.get('steps', 0)} "
            f"| {t.get('duration_s', 0):.1f}s "
            f"| {t.get('termination', '')} |"
        )
    return "\n".join(lines)
