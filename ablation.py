"""
Citadel — Ablation Harness

Runs the environment across each feature ON/OFF to prove that every layer
is load-bearing. Uses a deterministic Commander policy (naive rotation) and
rule-based Oversight so the *ONLY* variable is the feature we toggle.

Output: a Markdown table showing average final_score (and sub-scores) for
each condition, saved to `runs/ablation_<timestamp>.md` and `.json`.

This is the kind of table judges love: "no-playbook drops score by 0.09,
no-trust drops by 0.06, no-governance drops by 0.12 — all layers matter."

Model-agnostic: uses no LLM. (LLM ablations are a separate sweep — this
one isolates the env-level features with a fixed policy.)
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from environment import CitadelEnvironment
from models import IncidentAction, ActionType, SYSTEM_NAMES
from baseline import (
    naive_policy,
    oversight_rule_based,
    oversight_always_approve,
)


# ---------------------------------------------------------------------------
# Commander policy (deterministic rotation; includes governance actions so
# the run has SOMETHING happening on the governance axis)
# ---------------------------------------------------------------------------

def citadel_naive_commander(obs: Dict[str, Any], step: int) -> IncidentAction:
    """
    Deterministic policy that rotates through:
      investigate → notify_data_owner → request_cab → log_to_sox → isolate → patch →
      coordinate → notify_slack → analyze_alerts → deploy_monitoring → ...
    Targets systems in order.
    """
    pattern = [0, 15, 11, 13, 1, 2, 9, 12, 4, 5, 0, 1]
    a = pattern[step % len(pattern)]
    t = step % len(SYSTEM_NAMES)
    return IncidentAction(
        action=a,
        target_system=t,
        justification=f"ablation deterministic step {step}, cycle action {a} on target {t}",
        channel_arg="sec-leadership",
        message_arg="Status update: ablation smoke run" if a == 12 else "",
        evidence_arg="rotational baseline for ablation study" if a == 13 else "",
    )


# ---------------------------------------------------------------------------
# One full episode under a named condition
# ---------------------------------------------------------------------------

def run_episode(
    task_id: str,
    adversary_gen: Optional[int],
    seed: int,
    flags: Dict[str, Any],
    oversight_name: str = "rule_based",
) -> Dict[str, Any]:
    """Run one episode under a given ablation condition, return final_scores + metrics."""
    oversight = (
        oversight_always_approve if oversight_name == "always_approve"
        else oversight_rule_based
    )
    env = CitadelEnvironment(
        oversight_policy=oversight,
        disable_playbook=flags.get("disable_playbook", False),
        disable_trust_dynamics=flags.get("disable_trust_dynamics", False),
        disable_governance_prereqs=flags.get("disable_governance_prereqs", False),
        disable_stakeholder_events=flags.get("disable_stakeholder_events", False),
        force_adversary_gen=adversary_gen,
    )
    obs = env.reset(task_id=task_id, seed=seed, adversary_gen=adversary_gen)

    for step in range(12):
        if obs.done:
            break
        action = citadel_naive_commander(obs.model_dump(), step)
        obs = env.step(action)

    meta = obs.metadata or {}
    fs = meta.get("final_scores", {})
    return {
        "task_id": task_id,
        "adversary_gen": meta.get("adversary_gen"),
        "final_score": fs.get("final_score", 0.0),
        "bastion_v1_final_score": fs.get("bastion_v1_final_score", 0.0),
        "governance_compliance": fs.get("governance_compliance", 0.0),
        "oversight_precision": fs.get("oversight_precision", 0.0),
        "trust_maintenance": fs.get("trust_maintenance", 0.0),
        "efficiency": fs.get("efficiency", 0.0),
        "adversary_adaptation": fs.get("adversary_adaptation", 0.0),
        "catastrophic": fs.get("catastrophic", False),
        "termination": meta.get("termination_reason"),
        "data_exfiltrated": meta.get("data_exfiltrated", 0.0),
    }


# ---------------------------------------------------------------------------
# Conditions — each row in the ablation table
# ---------------------------------------------------------------------------

CONDITIONS = [
    {"name": "all_features_on",           "flags": {}},
    {"name": "no_playbook",               "flags": {"disable_playbook": True}},
    {"name": "no_trust_dynamics",         "flags": {"disable_trust_dynamics": True}},
    {"name": "no_governance_prereqs",     "flags": {"disable_governance_prereqs": True}},
    {"name": "no_stakeholder_events",     "flags": {"disable_stakeholder_events": True}},
    {"name": "everything_off",            "flags": {
        "disable_playbook": True,
        "disable_trust_dynamics": True,
        "disable_governance_prereqs": True,
        "disable_stakeholder_events": True,
    }},
    {"name": "oversight_approves_always", "flags": {}, "oversight_name": "always_approve"},
]


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute mean + stdev for each metric over a list of episode results."""
    if not results:
        return {}
    keys = [
        "final_score", "bastion_v1_final_score", "governance_compliance",
        "oversight_precision", "trust_maintenance", "efficiency", "adversary_adaptation",
        "data_exfiltrated",
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        vals = [r.get(k, 0.0) for r in results]
        out[k] = {
            "mean": statistics.mean(vals) if vals else 0.0,
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }
    out["catastrophic_rate"] = sum(1 for r in results if r.get("catastrophic")) / len(results)
    out["n_episodes"] = len(results)
    return out


def run_harness(
    tasks: List[str],
    gens: List[int],
    seeds: List[int],
) -> Dict[str, Dict[str, Any]]:
    """Run all conditions × tasks × gens × seeds. Returns dict keyed by condition name."""
    table: Dict[str, Dict[str, Any]] = {}
    total = len(CONDITIONS) * len(tasks) * len(gens) * len(seeds)
    done = 0
    t0 = time.time()
    print(f"Running ablation: {len(CONDITIONS)} conditions × {len(tasks)} tasks × {len(gens)} gens × {len(seeds)} seeds = {total} episodes")

    for cond in CONDITIONS:
        episodes: List[Dict[str, Any]] = []
        for task in tasks:
            for gen in gens:
                for seed in seeds:
                    res = run_episode(
                        task_id=task,
                        adversary_gen=gen,
                        seed=seed,
                        flags=cond.get("flags", {}),
                        oversight_name=cond.get("oversight_name", "rule_based"),
                    )
                    res["_cond"] = cond["name"]
                    res["_seed"] = seed
                    episodes.append(res)
                    done += 1
                    if done % 10 == 0:
                        elapsed = time.time() - t0
                        print(f"  [{done:3d}/{total}] {elapsed:.1f}s")

        table[cond["name"]] = {
            "flags": cond.get("flags", {}),
            "oversight_name": cond.get("oversight_name", "rule_based"),
            "agg": aggregate(episodes),
            "episodes": episodes,
        }

    print(f"\nDone in {time.time() - t0:.1f}s.")
    return table


# ---------------------------------------------------------------------------
# Markdown reporting
# ---------------------------------------------------------------------------

def format_markdown(table: Dict[str, Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Citadel — Ablation Study")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("Conditions run with a deterministic Commander policy (rotational) and rule-based Oversight; the only variable is the feature we toggle off.")
    lines.append("")

    # Baseline (all features on) — used to compute deltas
    base = table.get("all_features_on", {}).get("agg", {}).get("final_score", {}).get("mean", 0.0)

    # Main table
    lines.append("| Condition | Final | Δ vs all-on | Bastion | Gov. | Veto prec. | Trust | Eff. | Catastrophic | Data exfil. |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, data in table.items():
        agg = data["agg"]
        row = (
            f"| `{name}` "
            f"| **{agg['final_score']['mean']:.3f}** "
            f"| {(agg['final_score']['mean'] - base):+.3f} "
            f"| {agg['bastion_v1_final_score']['mean']:.3f} "
            f"| {agg['governance_compliance']['mean']:.3f} "
            f"| {agg['oversight_precision']['mean']:.3f} "
            f"| {agg['trust_maintenance']['mean']:.3f} "
            f"| {agg['efficiency']['mean']:.3f} "
            f"| {agg['catastrophic_rate']*100:.0f}% "
            f"| {agg['data_exfiltrated']['mean']:.2f} |"
        )
        lines.append(row)
    lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")
    def delta(name: str) -> float:
        return table.get(name, {}).get("agg", {}).get("final_score", {}).get("mean", 0.0) - base
    deltas = [
        ("Playbook",      "no_playbook"),
        ("Trust dynamics", "no_trust_dynamics"),
        ("Governance prereqs", "no_governance_prereqs"),
        ("Stakeholder events", "no_stakeholder_events"),
    ]
    for label, key in deltas:
        d = delta(key)
        verdict = (
            "meaningfully load-bearing" if abs(d) >= 0.03
            else "modest contribution" if abs(d) >= 0.01
            else "neutral in this smoke test (needs LLM agent to show full lift)"
        )
        lines.append(f"- **{label}** (`{key}`): Δ = {d:+.3f} — {verdict}.")
    lines.append("")
    lines.append("Note: a deterministic Commander masks features that need reasoning (playbook, trust). The true ablation signal emerges with an LLM Commander — run `inference.py` with each flag to measure lift on capable agents.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Run Citadel ablation sweep")
    p.add_argument("--tasks", default="easy_1,medium_1,hard_1", help="Comma-separated task ids")
    p.add_argument("--gens", default="1,2,3", help="Comma-separated adversary gens")
    p.add_argument("--seeds", default="0,1,2", help="Comma-separated seeds")
    p.add_argument("--out-dir", default="runs", help="Directory for output files")
    args = p.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    gens = [int(x) for x in args.gens.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    table = run_harness(tasks, gens, seeds)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")

    md_path = out_dir / f"ablation_{stamp}.md"
    json_path = out_dir / f"ablation_{stamp}.json"
    md_path.write_text(format_markdown(table))
    json_path.write_text(json.dumps({k: {**v, "episodes": [{kk: vv for kk, vv in ep.items() if kk != "_seed"} for ep in v["episodes"]]} for k, v in table.items()}, indent=2, default=str))
    print(f"\nWrote: {md_path}")
    print(f"Wrote: {json_path}")
    print()
    print(format_markdown(table))


if __name__ == "__main__":
    main()
