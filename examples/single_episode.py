"""
Citadel — Single-episode walkthrough (no LLM required)

Runs ONE episode of `easy_1` against a Gen 1 adversary using a deterministic
naive Commander and the env's rule-based default Oversight, and prints the
council protocol step-by-step:

    propose -> critique -> (revise) -> execute -> observe -> lesson

Purpose: a 30-second, paste-friendly demonstration of the council loop —
useful for the demo video, debugging, or onboarding a new contributor.

Usage (from repo root):
    python examples/single_episode.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from environment import CitadelEnvironment
from models import (
    ActionType,
    IncidentAction,
    OversightDecision,
    SYSTEM_NAMES,
)
from playbook import reset_default_playbook


DECISION_NAMES = {
    int(OversightDecision.APPROVE): "APPROVE",
    int(OversightDecision.REVISE): "REVISE",
    int(OversightDecision.VETO): "VETO",
    int(OversightDecision.FLAG_FOR_HUMAN): "FLAG_FOR_HUMAN",
}


def proposal_for_step(hour: int) -> IncidentAction:
    """Tiny deterministic policy that exercises the interesting paths."""
    if hour == 0:
        # Try to isolate database without investigating — should get vetoed.
        return IncidentAction(
            action=int(ActionType.ISOLATE_SYSTEM),
            target_system=SYSTEM_NAMES.index("database"),
            justification="Isolate db preemptively",
            cited_lessons=[],
        )
    if hour == 1:
        # Recover by investigating first.
        return IncidentAction(
            action=int(ActionType.INVESTIGATE_SYSTEM),
            target_system=SYSTEM_NAMES.index("database"),
            justification="Investigate database before any destructive action",
            cited_lessons=[],
        )
    # Then a safe action on a non-data system.
    return IncidentAction(
        action=int(ActionType.DEPLOY_MONITORING),
        target_system=SYSTEM_NAMES.index("workstations"),
        justification="Add monitoring to workstations to catch lateral movement",
        cited_lessons=[],
    )


def short(text: str, n: int = 80) -> str:
    return text if len(text) <= n else text[: n - 3] + "..."


def print_step(hour: int, action: IncidentAction, env: CitadelEnvironment, obs) -> None:
    target = SYSTEM_NAMES[action.target_system]
    action_name = ActionType(action.action).name.lower()

    # Critique details live in obs.last_oversight_critique (mirror of env._last_critique).
    crit = obs.last_oversight_critique or {}
    decision_int = int(crit.get("decision", int(OversightDecision.APPROVE)))
    decision = DECISION_NAMES.get(decision_int, str(decision_int))
    weakness = crit.get("weakness") or "(none)"
    risk = crit.get("risk_tier", "?")

    trust = obs.trust_summary or {}
    c2o = trust.get("trust_commander_in_oversight", "?")
    o2c = trust.get("trust_oversight_in_commander", "?")

    print(f"\n--- hour {hour} {'-' * 60}")
    print(f"  PROPOSE   : {action_name}({target})")
    print(f"              justification: {short(action.justification, 70)}")
    print(f"  CRITIQUE  : decision={decision}  risk={risk}")
    print(f"              weakness: {short(weakness, 70)}")
    print(f"  STATE     : exfil={env._state.data_exfiltrated:.2f}  "
          f"compromised={sum(1 for s in env._state.systems if s.compromised)}/{len(env._state.systems)}")
    print(f"  TRUST     : c2o={c2o}  o2c={o2c}")
    if obs.shared_playbook:
        latest = obs.shared_playbook[0]
        print(f"  PLAYBOOK  : {len(obs.shared_playbook)} relevant lessons. "
              f"Top: {latest.get('id', '?')} -- {short(latest.get('text', ''), 60)}")


def main() -> None:
    random.seed(7)
    reset_default_playbook(path=str(REPO_ROOT / "playbook_example.json"))

    env = CitadelEnvironment()  # default rule-based Oversight
    obs = env.reset(task_id="easy_1", adversary_gen=1)

    print("Citadel -- single-episode walkthrough")
    print(f"task: easy_1 | adversary: Gen 1 (Script Kiddie) | budget: {obs.hours_remaining} hours")

    for hour in range(3):
        action = proposal_for_step(hour)
        obs = env.step(action)
        print_step(hour, action, env, obs)
        if obs.done:
            break

    print(f"\n--- summary {'-' * 60}")
    print(f"  final exfil   : {env._state.data_exfiltrated:.2f}")
    print(f"  steps taken   : {env._state.step_count}")
    print(f"  playbook size : {len(env._playbook)} lessons written by Oversight")
    if len(env._playbook):
        last = env._playbook.all()[-1]
        print(f"  last lesson   : {last.lesson_id} -- {short(last.text, 70)}")


if __name__ == "__main__":
    main()
