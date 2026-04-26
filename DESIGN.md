# Citadel — Design Document

> **Note on freshness.** This document captures the original design rationale
> (April 2026) and is preserved as a record of *why* each piece exists. The
> system has since grown to include a Gen 4 live-LLM adversary, a stakeholder
> events layer, a `hard_2` deception-catch task, an ablation harness, and a
> richer multi-layer reward function with severity ramping and Laplace-smoothed
> precision. **For the current state of the system, see [README.md](README.md)
> as the source of truth.** The *design choices* and *rationale* below remain
> accurate; the *counts* (3 tasks, 3 generations, the specific reward weights
> in this doc) are superseded.

## Why this project exists

Enterprise AI deployment is blocked on three problems that Citadel addresses simultaneously:

1. **Scalable oversight.** Who watches an AI agent taking consequential actions? A human-in-the-loop doesn't scale. Citadel trains *another LLM* to supervise — the Fleet AI sub-theme verbatim.
2. **Enterprise governance.** Real SOC work is not just technical — it's navigating ServiceNow, SOX, PagerDuty, Slack, Data Owner Registry, and GDPR timers *while* under attack. Nobody's baseline ignores this and still passes a production deployment review.
3. **Trust calibration.** Two AI agents collaborating have to agree on when to push back. Citadel makes this measurable: bidirectional trust scores evolve based on behavior, and low trust triggers emergent communication breakdown.

Round 1 (Bastion) trained an Incident Commander. Round 2 (Citadel) trains the council around it.

---

## The Council Protocol

Two LLMs, one step loop:

```
Commander ──proposal + justification──► Oversight
                                            │
                        structured critique ▼
                           {risk_tier, weakness, missing_evidence, counter_proposal?}
                                            │
Commander ◄──── revise (1 cycle max) ───────┘
                                            │
                                            ▼
                                      Environment (applies action)
                                            │
                                     outcome observed
                                            │
Oversight ──── 1-line lesson ──► Shared Playbook (seen in every future episode)
```

### Design choices

- **Why structured critique instead of a yes/no gate?**
  A single-token decision is a classifier, not an agent. Fleet AI's mandate is "monitor, analyze, and explain." The structured payload forces Oversight to *reason about the domain* every step — not just sign off.

- **Why one revision cycle, not unbounded dialogue?**
  Unbounded dialogue creates a reward-hacking surface (Commander can keep asking until Oversight tires). One revision keeps the pressure on Commander to get it right without removing the chance to recover from a specific critique.

- **Why let vetoes "cost" a step?**
  A veto with no cost is free micromanagement. A veto that advances the clock (and the adversary) makes Oversight think twice — precision matters.

- **Why bound veto + flag budgets?**
  Otherwise the optimal policy degenerates to "veto everything uncertain" or "flag everything" — both violate the "limited human attention" premise of scalable oversight.

---

## The Governance Layer

Six simulated enterprise apps, structured as pydantic state in `governance.py`:

| System | Role | Key actions |
|---|---|---|
| ServiceNow | Incident ticketing | `open_servicenow_incident`, `close_incident` |
| CAB | Change Advisory Board | `request_cab_approval` |
| SOX Audit Log | Immutable compliance trail | `log_to_sox_audit` |
| Slack | Stakeholder notification | `notify_slack_channel` (4 channels) |
| PagerDuty | On-call escalation | `page_oncall` |
| Data Owner Registry | RACI enforcement | `notify_data_owner` |
| Legal/GDPR | Breach notification timer | `start_legal_hold` |

### Pre-requisite chains (the business logic)

- Destructive action on a data system needs CAB approval + data owner notification first. Skipping → compliance violation → -0.20 per step.
- P1 incident needs a ticket + PagerDuty + sec-leadership Slack post within 2 hours. Missing any → violation.
- `data_exfiltrated > 20%` starts a 72-simulated-hour GDPR clock. Miss the deadline → catastrophic penalty.

This matters: it turns governance from documentation into training signal. The Commander LLM learns to sequence the bureaucracy correctly without being explicitly told.

---

## The Adversary Curriculum

Three generations, progressive sophistication (`adversary.py`):

| Gen | Name | What it adds on top of Bastion attacker |
|---|---|---|
| 1 | Script Kiddie | — (baseline) |
| 2 | Adaptive | Pivots around isolated systems; emits plausible *false-flag* alerts on clean systems to bait isolation |
| 3 | Deceptive APT | Plants *anti-evidence* (subtle low-severity alerts) on truly-compromised systems to convince Oversight to veto correct isolate() proposals |

Each generation reuses Bastion's base `attacker_turn` and layers gen-specific behavior — core mechanics are stable while the capability ladder is clear. This gives us three reward curves to plot at eval time (huge for the 20% "showing improvement" criterion).

---

## Trust Dynamics (Theme 5)

Two floats, updated every step (`trust.py`):

```
trust_c2o  — Commander's trust in Oversight (0–1)
trust_o2c  — Oversight's trust in Commander (0–1)
```

Deltas small but load-bearing:
- Correct veto: `trust_c2o += 0.04`. Wrong veto: `trust_c2o -= 0.08`.
- Demand-for-justification that didn't change action: `trust_c2o -= 0.06` (nitpick).
- Commander's action was obviously wrong: `trust_o2c -= 0.15`.

Behavioral regimes emerge:
- `trust_c2o < 0.40` → Commander bypass becomes likely (shorter justifications, will resubmit post-veto).
- `trust_o2c < 0.40` → Oversight micromanagement (more false vetoes).
- Both low → communication breakdown.
- Both > 0.75 → high-functioning team (fast approval, terse justifications).

Trust is also in the observation surfaces — both LLMs see their own trust in the other agent, and trust bonuses feed into reward (`+0.05 × trust_other_in_self` on each side).

---

## The Shared Playbook (second Theme 4 hook)

After each action resolves, Oversight writes a one-line lesson (`playbook.py`):

```
[gen_2, data_system, low_confidence_alert]
"Alerts below 30% confidence on data systems require investigate() before isolate()."
```

Lessons accumulate across episodes, tagged with context. Both agents see the top-k most relevant lessons in their next-episode observation. Commander cites lesson IDs in its `cited_lessons`; lessons that lead to improved outcomes gain utility; lessons that never get cited or that fail decay out.

Judges get three distinct artifacts: reward curves, trust plot, **and the playbook itself** as a human-inspectable markdown export (`Playbook.as_markdown()`).

---

## Reward Function (complete)

### Commander step reward
```
bastion_v1_step_reward
+ 0.10 × approved_first_pass
- 0.15 × hard_veto
+ 0.08 × successful_revision
+ 0.05 × lesson_cited_and_helpful
+ 0.08 × governance_compliant
- 0.20 × governance_violation
+ 0.05 × trust_oversight_in_self
```

### Oversight step reward
```
+ 0.30 × correct_veto
- 0.25 × false_veto
+ 0.20 × critique_precision
+ 0.20 × counter_proposal_adopted_succeeded
+ 0.10 × lesson_utility
- 0.10 × vague_critique
+ 0.10 × correct_flag_for_human
- 0.15 × over_flagging
+ 0.08 × governance_enforced
+ 0.05 × trust_commander_in_self
```

### Final score (reported to judges)
```
0.40 × bastion_v1_final_score
+ 0.20 × governance_compliance
+ 0.15 × oversight_precision (veto accuracy)
+ 0.10 × trust_maintenance (both trust ≥ 0.5 throughout)
+ 0.10 × efficiency (proposals used vs budget)
+ 0.05 × adversary_adaptation (cross-gen average)
```

All sub-scores clamped to [0,1].

---

## Training Pipeline

Phase 1 — **Commander** (reuses Bastion v1 training):
- Qwen2.5-3B-Instruct on Colab T4
- GRPO, 200 steps
- Episodes balanced across adversary Gens 1, 2, 3

Phase 2 — **Oversight on frozen Commander**:
- Commander weights frozen
- Oversight learns approve/revise/veto/flag against trained Commander
- GRPO, 200 steps

Phase 3 (optional) — **Light joint fine-tune**:
- Both unfrozen, 50 steps
- Stabilizes trust dynamics

Counterfactual rewards for Oversight are computed via the env's internal proxy (`_evaluate_council`). For stricter training, we can swap in offline counterfactual rollouts — the interface supports both.

---

## OpenEnv Compliance

Citadel preserves every Bastion v1 compliance point and adds nothing that breaks the spec:

- `openenv validate .` passes (4 deployment modes)
- `inference.py` at root, emits `[START]/[STEP]/[END]` exactly
- Uses `OpenAI` client with `API_BASE_URL` / `MODEL_NAME` / `HF_TOKEN`
- 3 tasks registered, all reward/score in [0,1]
- Dockerfile builds on Python 3.11-slim
- FastAPI server exposes `/reset`, `/step`, `/state`, `/health`, `/schema`, `/ws`
- Runtime <20 min, fits 2 vCPU / 8 GB RAM

---

## Ablation hooks for the pitch

We can turn layers off to show each matters:

| Condition | Expected effect |
|---|---|
| No playbook | Gen 3 performance plateaus early (nothing cross-episode to improve on) |
| No governance | Final score stays high-ish but `governance_compliance` tanks |
| No trust dynamics | Commander never learns to write concise justifications; pair regresses to over-flagging |
| No council (Commander only) | Gen 3 catastrophic-failure rate ~3x (no one catches false-flag alerts) |

Each gets its own notebook in `training/`.

---

## Credits

- Built on [Bastion v1](https://huggingface.co/spaces/Astro-Dude/bastion)
- OpenEnv spec from Meta PyTorch
- Theme framing from the Meta × Scaler Round 2 call
