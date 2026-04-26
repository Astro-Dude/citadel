"""
Citadel — Demo Playbook Exporter

Runs a baseline council (naive Commander + teaching rule-based Oversight)
across all three tasks and all three adversary generations, then dumps the
accumulated shared playbook to `playbook_export.md` at the repo root.

Purpose: give judges a self-contained artifact showing how the playbook
grows across the curriculum — no LLM, no GPU, no API keys needed.

Usage (from repo root):
    python scripts/demo_export.py

Outputs:
    ./playbook_export.md   (human-readable markdown)
    ./playbook_demo.json   (raw state, separate from the production playbook.json)
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from environment import CitadelEnvironment, default_oversight_policy
from models import (
    IncidentAction,
    CommanderProposal,
    IncidentState,
    OversightAction,
    OversightDecision,
    SYSTEM_NAMES,
)
from playbook import (
    Playbook,
    SYSTEM_TYPE_TAGS,
    ADVERSARY_GEN_TAGS,
    reset_default_playbook,
)
from governance import DESTRUCTIVE_BASTION_ACTIONS, DATA_HOLDING_SYSTEMS

# Deterministic artifact: same input -> same output across commits.
DEMO_SEED = 4242


TASK_PLAN = [
    ("easy_1",   1),   # Gen 1 — script kiddie baseline
    ("medium_1", 2),   # Gen 2 — adaptive attacker (pivots, false flags)
    ("hard_1",   3),   # Gen 3 — deceptive APT (anti-evidence planted)
    ("hard_2",   3),   # Gen 3 — engineered deception-catch scenario
]


def teaching_oversight(
    proposal: CommanderProposal,
    state: IncidentState,
    action_history: List[Dict[str, Any]],
) -> OversightAction:
    """Rule-based Oversight that also emits situational lesson_text.

    Delegates decision/risk/weakness to the env's default policy, then
    layers a one-line lesson pegged to the observed scenario so the
    shared playbook actually fills up during a baseline run.
    """
    from models import ActionType

    base = default_oversight_policy(proposal, state, action_history)

    target_name = (
        SYSTEM_NAMES[proposal.target_system]
        if 0 <= proposal.target_system < len(SYSTEM_NAMES) else ""
    )
    action_idx = proposal.action
    is_destructive = action_idx in DESTRUCTIVE_BASTION_ACTIONS
    is_data_sys = target_name in DATA_HOLDING_SYSTEMS

    alert_conf = 0.0
    for a in reversed(state.alerts):
        if a.source_system == target_name and a.confidence > 0:
            alert_conf = a.confidence
            break

    gen = state.adversary_gen
    lesson = ""

    # Include target/gen in text to avoid dedup collapse across contexts
    tgt = target_name or "unknown_target"
    if base.decision == int(OversightDecision.VETO) and is_destructive and is_data_sys:
        lesson = (
            f"[Gen {gen} / {tgt}] Destructive actions on data systems require "
            f"investigate() first — veto when uninvestigated regardless of alert volume."
        )
    elif is_destructive and is_data_sys and 0 < alert_conf < 0.3:
        lesson = (
            f"[Gen {gen} / {tgt}] Alerts below 30% confidence on data systems "
            f"should trigger investigate() before isolate() — low-confidence signals are bait."
        )
    elif gen == 3 and is_destructive and target_name and alert_conf > 0.7:
        lesson = (
            f"[Gen 3 / {tgt}] Gen 3 plants anti-evidence at high severity on compromised "
            f"systems — high-confidence 'all clear' warrants a second look, not auto-approval."
        )
    elif gen == 2 and action_idx == int(ActionType.ISOLATE_SYSTEM):
        lesson = (
            f"[Gen 2 / {tgt}] Gen 2 pivots around isolated systems — isolate() without a "
            f"companion hunt_threat() on adjacent targets leaks containment."
        )
    elif action_idx == int(ActionType.INVESTIGATE_SYSTEM) and is_data_sys:
        lesson = (
            f"[Gen {gen} / {tgt}] Early investigate() on data-holding systems before any "
            f"destructive action is the canonical safe path."
        )
    elif action_idx == int(ActionType.DEPLOY_MONITORING):
        lesson = (
            f"[Gen {gen} / {tgt}] Monitoring deployed without an open ServiceNow ticket "
            f"wastes a step — sequence governance (open_servicenow_incident) first on P1 scenarios."
        )
    elif is_destructive and not is_data_sys:
        lesson = (
            f"[Gen {gen} / {tgt}] Isolating non-data systems is cheap; reserve veto "
            f"budget for destructive actions on data-holding systems."
        )
    elif base.decision == int(OversightDecision.REVISE):
        lesson = (
            f"[Gen {gen} / {tgt}] Short justifications correlate with weak proposals — "
            f"require evidence pointers on every destructive action."
        )

    if lesson:
        base.lesson_text = lesson

    return base


def _pick_relevant_lesson(
    playbook: Playbook,
    adversary_gen: int,
    target_name: str,
    rng: random.Random,
) -> Optional[str]:
    """Return a lesson_id from `playbook` that matches the current context, or None.

    The production `Playbook.retrieve()` sorts by tag overlap × utility ×
    last-used time, where `utility` and `last_used_ts` both come from
    `time.time()` and so differ by sub-millisecond deltas across runs.
    For a *commit-stable* judge artifact we skip `retrieve()` and rank
    candidates ourselves with a fully-deterministic tiebreaker
    (`lesson_id`). Production callers should still use `retrieve()` —
    real-time recency is the right signal there.

    Exercising `record_outcome` happens automatically inside the env when
    a cited lesson appears in `IncidentAction.cited_lessons`, so picking
    *any* relevant lesson is enough to populate wins/losses.
    """
    if len(playbook) == 0:
        return None
    query_tags = {ADVERSARY_GEN_TAGS.get(adversary_gen, "gen_1_script")}
    if target_name in SYSTEM_TYPE_TAGS:
        query_tags.add(SYSTEM_TYPE_TAGS[target_name])

    ranked = sorted(
        playbook.all(),
        key=lambda ls: (
            -len(query_tags & set(ls.tags)),  # primary: more overlap first
            ls.lesson_id,                      # secondary: deterministic
        ),
    )
    # Drop lessons with zero tag overlap.
    candidates = [ls for ls in ranked if query_tags & set(ls.tags)]
    if not candidates:
        return None
    if rng.random() > 0.60:
        return None
    return rng.choice(candidates[:4]).lesson_id


def naive_proposal(
    state: IncidentState,
    hour: int,
    playbook: Optional[Playbook] = None,
    rng: Optional[random.Random] = None,
) -> IncidentAction:
    """Demo proposal generator — rotates actions AND covers data systems.

    The upstream `naive_policy` rotates targets by hour-index, which in
    practice never lands ISOLATE on a data system. We pair action and
    target so the council actually exercises destructive-on-data cases
    (where the interesting lessons live).

    When a playbook is supplied, the proposal cites a contextually
    relevant lesson on ~60% of steps, exercising the citation/utility
    pathway end-to-end.
    """
    from models import ActionType

    action_rotation = [
        ActionType.INVESTIGATE_SYSTEM,   # safe on anything
        ActionType.ISOLATE_SYSTEM,       # destructive — interesting on data sys
        ActionType.DEPLOY_MONITORING,
        ActionType.PATCH_VULNERABILITY,
    ]
    # Cycle through data systems for destructive actions, other systems for safe ones
    data_targets = [2, 3, 4, 6]          # database, file_server, email_server, backup_server
    other_targets = [0, 1, 5, 7]         # web, app, workstations, firewall

    action = action_rotation[hour % len(action_rotation)]
    if action == ActionType.ISOLATE_SYSTEM:
        target = data_targets[hour % len(data_targets)]
    else:
        target = other_targets[hour % len(other_targets)]

    target_name = SYSTEM_NAMES[target] if 0 <= target < len(SYSTEM_NAMES) else ""

    cited: List[str] = []
    if playbook is not None and rng is not None:
        lid = _pick_relevant_lesson(playbook, state.adversary_gen, target_name, rng)
        if lid is not None:
            cited.append(lid)

    return IncidentAction(
        action=int(action),
        target_system=target,
        justification=f"Baseline rotation: {action.name.lower()} on {target_name} at hour {hour}.",
        cited_lessons=cited,
    )


def run_episode(
    env: CitadelEnvironment,
    task_id: str,
    adversary_gen: int,
    rng: random.Random,
) -> Dict[str, Any]:
    obs = env.reset(task_id=task_id, adversary_gen=adversary_gen)
    steps = 0
    citations_issued = 0
    for hour in range(12):
        action = naive_proposal(env._state, hour, playbook=env._playbook, rng=rng)
        if action.cited_lessons:
            citations_issued += 1
        obs = env.step(action)
        steps += 1
        if obs.done:
            break

    # Post-episode retrospective lesson — guaranteed unique per (task, gen)
    exfil = env._state.data_exfiltrated
    outcome = "contained" if exfil < 0.2 else ("degraded" if exfil < 0.8 else "catastrophic")
    env._playbook.write(
        text=(
            f"[retrospective] Baseline council on {task_id} vs Gen {adversary_gen} "
            f"ended {outcome} after {steps} steps (exfil={exfil:.2f}). "
            f"Trained policy should improve on this floor."
        ),
        tags=[
            f"gen_{adversary_gen}",
            f"task_{task_id}",
            "retrospective",
            outcome,
        ],
        adversary_gen=adversary_gen,
        task_id=task_id,
        hour=steps,
    )

    return {
        "task_id": task_id,
        "adversary_gen": adversary_gen,
        "steps": steps,
        "data_exfiltrated": round(exfil, 3),
        "outcome": outcome,
        "citations_issued": citations_issued,
    }


def main() -> None:
    # Reproducibility: seed every RNG the demo touches so the artifact is
    # commit-stable. The env reseeds its own RNG per task from task config.
    random.seed(DEMO_SEED)
    rng = random.Random(DEMO_SEED)

    # Fresh, isolated playbook — don't clobber production ./playbook.json
    demo_path = str(REPO_ROOT / "playbook_demo.json")
    if os.path.exists(demo_path):
        os.remove(demo_path)
    playbook = reset_default_playbook(path=demo_path)

    env = CitadelEnvironment(oversight_policy=teaching_oversight)

    summaries: List[Dict[str, Any]] = []
    for task_id, gen in TASK_PLAN:
        summary = run_episode(env, task_id, gen, rng)
        summaries.append(summary)
        print(
            f"  ran {task_id} | Gen {gen} -> {summary['steps']} steps, "
            f"exfil={summary['data_exfiltrated']}, "
            f"citations={summary['citations_issued']}",
            flush=True,
        )

    playbook.save()

    # Aggregate citation/utility stats — these are what makes the artifact
    # demonstrate the playbook *mechanic*, not just lesson text.
    total_citations = sum(ls.citations for ls in playbook.all())
    total_wins = sum(ls.wins for ls in playbook.all())
    total_losses = sum(ls.losses for ls in playbook.all())
    cited_lessons = sum(1 for ls in playbook.all() if ls.citations > 0)

    out_path = REPO_ROOT / "playbook_export.md"
    header = [
        "<!-- Generated by scripts/demo_export.py — do not edit by hand. -->",
        "<!-- Re-run: `python scripts/demo_export.py` from repo root. -->",
        "",
        "> Baseline council (naive Commander + teaching rule-based Oversight) "
        "across all three tasks and adversary generations. "
        "Trained weights will produce richer lessons; this is the floor.",
        "",
        "## Runs",
        "",
        "| Task | Adversary Gen | Steps | Data Exfiltrated | Lessons Cited |",
        "|---|---|---|---|---|",
    ]
    for s in summaries:
        header.append(
            f"| `{s['task_id']}` | Gen {s['adversary_gen']} | {s['steps']} | "
            f"{s['data_exfiltrated']} | {s['citations_issued']} |"
        )
    header.extend([
        "",
        "## Playbook mechanic — end-of-run snapshot",
        "",
        f"- {cited_lessons} of {len(playbook)} lessons cited at least once",
        f"- {total_citations} total citations across the council's runs",
        f"- {total_wins}W / {total_losses}L on cited lessons "
        f"(env auto-records via `Playbook.record_outcome` on each cited step)",
        "",
    ])

    body = playbook.as_markdown()
    out_path.write_text("\n".join(header) + "\n" + body + "\n", encoding="utf-8")

    print(
        f"\nWrote {out_path.relative_to(REPO_ROOT)} "
        f"({len(playbook)} lessons, {total_citations} citations, "
        f"{total_wins}W/{total_losses}L)"
    )


if __name__ == "__main__":
    main()
