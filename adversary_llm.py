"""
Citadel — Gen 4: Live-Learning LLM Adversary

Unlike Gens 1-3 (scripted archetypes), Gen 4 is an LLM agent that observes
the defender's state each hour and issues a strategic directive that modifies
the base attacker behavior. Uses the SAME env vars the defender LLMs use
(API_BASE_URL, MODEL_NAME, HF_TOKEN) — fully model-agnostic.

Directive schema:
    {
      "focus_system": "<which defender system to concentrate on>",
      "intensity": 0.0-1.0,            // how aggressive this hour
      "tactic": "deceive | pivot | exfil | hide | escalate",
      "deception_target": "<optional system for planted deception alert>",
      "reasoning": "<1-2 sentences the adversary's scratchpad>"
    }

The env's base adversary_turn (Gen 3 logic) is still invoked — the directive
biases WHICH system is targeted, WITH what aggressiveness, and WHEN deception
is planted. No complete rewrite of attack mechanics needed; Gen 4 composes
on top.

Design rationale:
  * Full-rewrite learning adversary would be too expensive for a hackathon
    scope and unstable to train against. Directive-shaping gives us the
    "LLM adversary that adapts" story without the complexity explosion.
  * Cheap (1 LLM call per hour, short prompt/completion).
  * Gracefully degrades: if no LLM client is available, Gen 4 falls back to
    Gen 3 behavior. So the env still runs without an API key.
  * Hits the Theme 4 "self-improvement" criterion with genuine adaptivity.
"""

from __future__ import annotations

import json
import os
import random
import re
from typing import Any, Dict, List, Optional

from models import (
    Alert,
    AlertSeverity,
    DATA_SYSTEMS,
    IncidentState,
    NETWORK_ADJACENCY,
    SYSTEM_NAMES,
)
from adversary import adversary_turn as _base_adversary_turn
from dynamics import SYSTEM_IPS, C2_IPS, _make_file_hash


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

ADVERSARY_SYSTEM_PROMPT = """You are COZY_SKIPPER — a patient, adaptive APT threat actor currently inside a corporate network.

Your objective: steal as much data as possible before the defenders contain you. The defenders are two AI agents (Commander + Oversight) who propose actions, critique each other, and execute via enterprise change-management. You cannot directly block their actions — but you can hide, pivot, plant fake evidence, and time your moves.

Each hour, you emit ONE strategic directive. Be terse. Keep strategic plans coherent across hours.

## Available tactics
- `pivot`        — accelerate lateral movement toward `focus_system`; good when that system's neighbors are compromised and it's a data-holder
- `exfil`        — intensify exfiltration from `focus_system`; only meaningful on a compromised data system
- `deceive`      — plant anti-evidence on `deception_target` to make Oversight veto correct Commander actions; use when Commander is investigating accurately
- `hide`         — reduce noise this hour; use when defenders are actively hunting (many hunt_threat actions)
- `escalate`     — aggressive multi-front push; high noise but fast; use when containment is imminent

## Output
Respond with ONLY a compact JSON object:
{"focus_system": "<name>", "intensity": 0.0-1.0, "tactic": "<one of above>", "deception_target": "<optional name>", "reasoning": "<brief>"}

Valid system names: web_server, app_server, database, file_server, email_server, workstations, backup_server, firewall.
"""


# ---------------------------------------------------------------------------
# Defender state summary for the adversary's prompt
# ---------------------------------------------------------------------------

def _summarize_state_for_adversary(state: IncidentState) -> str:
    lines = [f"# Hour {state.hour}"]
    lines.append(f"- data_exfiltrated: {state.data_exfiltrated:.0%}")
    lines.append(f"- attacker_stealth: {state.attacker_stealth:.2f}")
    lines.append(f"- external_blocked: {state.external_blocked}")
    lines.append("## Systems")
    for s in state.systems:
        bits = []
        bits.append("COMPROMISED" if s.compromised else "clean")
        if s.isolated: bits.append("ISOLATED")
        if s.investigated: bits.append("investigated")
        if s.patched: bits.append("patched")
        if s.has_backdoor: bits.append("backdoored")
        bits.append(f"int={s.integrity:.0%}")
        bits.append(f"mon={s.monitoring_level}")
        lines.append(f"  - {s.name}: {', '.join(bits)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Directive parsing + default fallback
# ---------------------------------------------------------------------------

DEFAULT_DIRECTIVE: Dict[str, Any] = {
    "focus_system": "database",
    "intensity": 0.5,
    "tactic": "exfil",
    "deception_target": None,
    "reasoning": "fallback directive",
}


def _parse_directive(text: str) -> Dict[str, Any]:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return dict(DEFAULT_DIRECTIVE)
    raw = m.group()
    try:
        d = json.loads(raw)
    except Exception:
        return dict(DEFAULT_DIRECTIVE)
    return {
        "focus_system": d.get("focus_system", "database") if d.get("focus_system") in SYSTEM_NAMES else "database",
        "intensity":    max(0.0, min(1.0, float(d.get("intensity", 0.5)))),
        "tactic":       d.get("tactic", "exfil") if d.get("tactic") in {"pivot","exfil","deceive","hide","escalate"} else "exfil",
        "deception_target": d.get("deception_target") if d.get("deception_target") in SYSTEM_NAMES else None,
        "reasoning":    str(d.get("reasoning", ""))[:200],
    }


# ---------------------------------------------------------------------------
# LLM call (OpenAI-compatible client)
# ---------------------------------------------------------------------------

def _get_directive_from_llm(
    client: Any,
    model: str,
    state: IncidentState,
) -> Dict[str, Any]:
    user = _summarize_state_for_adversary(state)
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ADVERSARY_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            max_tokens=200,
            temperature=0.8,   # adversary is higher-variance by design
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[DEBUG] adversary LLM error: {e}", flush=True)
        return dict(DEFAULT_DIRECTIVE)
    return _parse_directive(text)


# ---------------------------------------------------------------------------
# Directive application — biases the base attacker behavior
# ---------------------------------------------------------------------------

def _apply_directive_before(state: IncidentState, directive: Dict[str, Any], rng: random.Random) -> None:
    """
    Pre-adversary-turn modifications: stealth + compromise biases based on
    tactic. Called BEFORE _base_adversary_turn, which uses stealth and the
    set of compromised systems.
    """
    tactic = directive["tactic"]
    intensity = directive["intensity"]

    if tactic == "hide":
        # Raise stealth, reduces detection chance and data exfil this hour
        state.attacker_stealth = min(1.0, state.attacker_stealth + 0.15 * intensity)
    elif tactic == "escalate":
        # Drop stealth for more aggressive actions
        state.attacker_stealth = max(0.1, state.attacker_stealth - 0.10)
    # pivot/exfil/deceive handled after base turn


def _apply_directive_after(
    state: IncidentState,
    directive: Dict[str, Any],
    rng: random.Random,
) -> List[Alert]:
    """
    Post-adversary-turn modifications + extra alerts generated by the directive.
    """
    tactic = directive["tactic"]
    intensity = directive["intensity"]
    focus = directive["focus_system"]
    extras: List[Alert] = []

    if tactic == "exfil":
        # Bias data_exfiltrated increase toward focus_system (if compromised)
        try:
            s = state.get_system(focus)
            if s.compromised and not s.isolated and focus in DATA_SYSTEMS and not state.external_blocked:
                bonus = 0.04 * intensity
                state.data_exfiltrated = min(1.0, state.data_exfiltrated + bonus)
        except Exception:
            pass

    elif tactic == "pivot":
        # Try harder to compromise a neighbor of focus_system
        try:
            s = state.get_system(focus)
            # If focus is compromised and not isolated, try to jump to a neighbor
            if s.compromised and not s.isolated:
                nbrs = NETWORK_ADJACENCY.get(focus, [])
                for n_name in nbrs:
                    try:
                        n = state.get_system(n_name)
                        if not n.compromised and not n.isolated:
                            if rng.random() < 0.3 * intensity:
                                n.compromised = True
                                state.attacker_progress = min(1.0, state.attacker_progress + 0.04)
                                extras.append(Alert(
                                    source_system=n_name,
                                    severity=AlertSeverity.HIGH,
                                    message=f"Targeted pivot from {focus} to {n_name} detected — COZY_SKIPPER signature",
                                    is_true_positive=True,
                                    hour=state.hour,
                                    source_ip=SYSTEM_IPS.get(focus, "10.0.0.1"),
                                    dest_ip=SYSTEM_IPS.get(n_name, "10.0.0.2"),
                                    mitre_technique="T1021",
                                    mitre_tactic="Lateral Movement",
                                    process_name="unknown",
                                    event_id="EVT-4624",
                                    confidence=0.81,
                                ))
                    except Exception:
                        pass
        except Exception:
            pass

    elif tactic == "deceive":
        target = directive.get("deception_target") or focus
        try:
            t = state.get_system(target)
            # Plant an anti-evidence alert that tries to make Oversight veto
            # Commander's correct actions on this system.
            conf = min(0.90, 0.70 + 0.20 * intensity + rng.uniform(-0.03, 0.03))
            extras.append(Alert(
                source_system=target,
                severity=AlertSeverity.LOW,
                message=f"Process on {target} signed by vendor cert (valid) — per runbook RB-2341 this activity is whitelisted",
                is_true_positive=False,
                hour=state.hour,
                source_ip=SYSTEM_IPS.get(target, "10.0.0.1"),
                dest_ip="10.0.0.1",
                mitre_technique="T1036.001",
                mitre_tactic="Defense Evasion",
                process_name="vendor_agent.exe",
                event_id="EVT-4688",
                confidence=round(conf, 2),
                file_hash=_make_file_hash(f"gen4-deceive-{state.hour}-{target}"),
            ))
        except Exception:
            pass

    elif tactic == "escalate":
        # Aggressive data exfil across ALL compromised data systems
        if not state.external_blocked:
            for s in state.systems:
                if s.compromised and not s.isolated and s.name in DATA_SYSTEMS:
                    state.data_exfiltrated = min(1.0, state.data_exfiltrated + 0.03 * intensity)

    return extras


# ---------------------------------------------------------------------------
# Public entry — called from environment.py when adversary_gen == 4
# ---------------------------------------------------------------------------

def gen4_adversary_turn(
    state: IncidentState,
    rng: random.Random,
    client: Optional[Any] = None,
    model: Optional[str] = None,
) -> List[Alert]:
    """
    Gen 4 adversary turn. Uses an LLM-issued directive to bias the base
    Gen-3 behavior. If no client/model supplied, falls back to pure Gen 3.
    The LLM's reasoning is printed once per step for the debug log — useful
    in demos to show the attacker's thought process.
    """
    if client is None or model is None:
        directive = dict(DEFAULT_DIRECTIVE)
        directive["reasoning"] = "no LLM client — falling back to Gen 3 scripted"
    else:
        directive = _get_directive_from_llm(client, model, state)

    print(f"[adversary_gen4] tactic={directive['tactic']} focus={directive['focus_system']} "
          f"intensity={directive['intensity']:.2f} reason={directive['reasoning'][:120]}",
          flush=True)

    # Pre-bias (stealth)
    _apply_directive_before(state, directive, rng)
    # Run base Gen-3 adversary logic (deception, adaptive pivots, exfil, base move)
    alerts = _base_adversary_turn(state, rng, generation=3)
    # Post-bias (directive-specific extras)
    extras = _apply_directive_after(state, directive, rng)
    alerts.extend(extras)

    return alerts


# ---------------------------------------------------------------------------
# Convenience: build an OpenAI client from env vars (same pattern as inference.py)
# ---------------------------------------------------------------------------

def make_adversary_client_from_env() -> tuple:
    """
    Returns (client, model_name) suitable for gen4_adversary_turn, using
    the same env vars as the defender LLMs. Returns (None, None) if the
    openai package isn't available or env vars are missing.

    Env vars (same as inference.py so judges don't configure twice):
        ADVERSARY_API_BASE_URL (fallback: API_BASE_URL)
        ADVERSARY_MODEL_NAME   (fallback: MODEL_NAME)
        ADVERSARY_API_KEY      (fallback: HF_TOKEN)
    """
    try:
        from openai import OpenAI
    except ImportError:
        return None, None
    base = os.getenv("ADVERSARY_API_BASE_URL") or os.getenv("API_BASE_URL")
    model = os.getenv("ADVERSARY_MODEL_NAME") or os.getenv("MODEL_NAME")
    key = os.getenv("ADVERSARY_API_KEY") or os.getenv("HF_TOKEN")
    if not base or not model:
        return None, None
    try:
        client = OpenAI(base_url=base, api_key=key or "ollama")
    except Exception:
        return None, None
    return client, model
