---
title: Citadel
emoji: 🏰
colorFrom: indigo
colorTo: red
sdk: docker
pinned: false
license: mit
app_port: 7860
tags:
  - openenv
  - multi-agent
  - scalable-oversight
  - cybersecurity
  - incident-response
  - enterprise-governance
  - reinforcement-learning
---

# Citadel — Multi-Agent AI Defense Council

> Bastion defended. Sentinel supervised. **Citadel is an LLM council** that governs, critiques, co-evolves, and earns trust.

Citadel is the Round 2 submission for the Meta PyTorch × Scaler OpenEnv Hackathon. It extends our Round 1 project (Bastion) into a full enterprise SOC environment: two LLMs cooperate through a structured critique protocol, under real enterprise governance constraints, against three generations of adversary, while maintaining bidirectional trust and a shared lessons playbook.

## Why a council, not a gate

Most multi-agent oversight work treats the reviewer as a one-token classifier (`approve`/`veto`). That's an oversight *gate*, not an overseer. Fleet AI's mandate — "monitor, **analyze, and explain** the behavior of other AI agents" — demands more. In Citadel, Oversight emits a structured critique every step:

```json
{
  "decision": "approve | revise | veto | flag_for_human",
  "risk_tier": 3,
  "weakness": "isolating database without investigation — alert confidence 22%",
  "missing_evidence": ["no log confirming lateral movement from db"],
  "counter_proposal": {"action": "investigate", "target": "database"},
  "lesson_text": "Alerts <30% confidence on data systems require investigate() before isolate()."
}
```

- On **REVISE**, Commander gets one chance to address the named weakness.
- On **VETO**, Commander must pick a different action next step.
- After every action resolves, Oversight writes a one-line lesson that both agents read in every future episode. Lessons accumulate, compete on utility, and decay — the council literally trains itself.

## Themes hit

| Theme | How Citadel covers it |
|---|---|
| **1 — Multi-Agent Interactions** | Commander + Oversight form an LLM council with separate reward functions |
| ↳ **Fleet AI (Scalable Oversight)** | Oversight analyzes & explains, not just monitors |
| **3.1 — Professional Tasks** | ServiceNow, SOX, PagerDuty, Slack, Data Owner Registry, GDPR timer — real enterprise pre-requisite chains |
| ↳ **Scaler AI Labs (Multi-App Enterprise)** | Governance layer is 6 interacting simulated enterprise systems |
| **4 — Self-Improvement** | Two hooks: (a) adversary curriculum (Gen 1/2/3), (b) shared playbook written by Oversight, cited by Commander |
| **5 — Wild Card** | Bidirectional trust dynamics — emergent communication breakdown / high-functioning regimes |

## The environment

- 8-system network (web_server, app_server, database, file_server, email_server, workstations, backup_server, firewall) with adjacency + criticality
- 18 Commander actions (10 incident response + 8 governance)
- 4 Oversight decisions with structured critique payload
- 3 tasks (easy_1 / medium_1 / hard_1), each with a default adversary generation
- 3 adversary generations — **script kiddie → adaptive → deceptive APT**
- SIEM alerts with MITRE ATT&CK mapping, false positives, false flags
- Partially observable; compromise status unknown until investigate/hunt

## Quickstart (local)

```bash
pip install -r requirements.txt
# Run the council over all three tasks with the default LLM:
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export HF_TOKEN=hf_xxx
python inference.py
```

If no Docker image / HF Space is reachable, `inference.py` falls back to an in-process `LocalEnv` — the validator flow still works end-to-end.

## Demo artifact (no-LLM playbook preview)

For judges who want to see the shared playbook without provisioning an LLM, we ship a deterministic baseline runner:

```bash
python scripts/demo_export.py
```

This drives a naive Commander (rotating actions across data and non-data systems) and a rule-based *teaching* Oversight (real critiques + situational `lesson_text`) over all three tasks across adversary Gens 1/2/3, then writes:

- [playbook_export.md](playbook_export.md) — human-readable playbook grouped by adversary generation, with per-lesson utility, wins/losses, citation count, and provenance (task + hour). The header table summarizes steps and data exfiltrated per run.
- [playbook_demo.json](playbook_demo.json) — raw lesson state, kept separate from the production `playbook.json` so the demo never clobbers a trained council's memory.

The export is the *floor* — it shows what the council captures with zero training. Trained weights produce richer, more cited lessons.

## Pre-submission validation

Citadel preserves all of Bastion's OpenEnv compliance:
- ✅ `openenv validate .` passes (4 deployment modes)
- ✅ `inference.py` in repo root with `[START]/[STEP]/[END]` stdout
- ✅ 3 tasks + graders (scores clamped to [0,1])
- ✅ Dockerfile builds (Python 3.11-slim base)
- ✅ `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN` env vars (OpenAI client)
- ✅ Runtime <20 min, fits 2 vCPU / 8 GB RAM

Run the hackathon validator:

```bash
./validate-submission.sh https://astro-dude-citadel.hf.space .
```

## File layout

```
Citadel/
├── models.py               # Pydantic action/obs/state types + new Council types
├── governance.py           # 6 enterprise-app simulators + pre-req + compliance score
├── trust.py                # Bidirectional trust dynamics (Theme 5)
├── playbook.py             # Shared lessons memory + as_markdown() export (Theme 4 hook)
├── adversary.py            # 3 adversary generations (Theme 4 curriculum)
├── dynamics.py             # Bastion attacker + team comms + forensic report
├── environment.py          # Two-agent council step loop
├── oversight_env.py        # Oversight-perspective wrapper for Phase 2 training
├── reward.py               # Commander / Oversight / Joint final score
├── baseline.py             # Commander baselines (no_op, naive) + Oversight baselines
├── tasks.py                # 3 scenarios, each with default_adversary_gen
├── client.py               # CitadelEnv OpenEnv client
├── inference.py            # Drives both LLMs through 3 tasks
├── server/app.py           # FastAPI server
├── scripts/demo_export.py  # Baseline runner that emits the judge artifact below
├── playbook_export.md      # Generated: human-readable playbook from baseline run
├── playbook_demo.json      # Generated: raw lesson state for the demo (not production)
├── Dockerfile, openenv.yaml, pyproject.toml, requirements.txt
└── training/               # Notebooks for Commander + Oversight + curriculum + trust
```

## Judging angles

- **40% Environment Innovation** — Council protocol + shared playbook + bidirectional trust are each novel; the combination is genuinely unpublished.
- **30% Storytelling** — Demo contrasts untrained pair (trust collapse, bypass, 60% data loss) vs trained pair (clean, governance-compliant, <10% data loss).
- **20% Showing Improvement** — Two reward curves, a 3×generation performance matrix, a trust-evolution plot, a growing playbook (see [playbook_export.md](playbook_export.md) for the untrained baseline).
- **10% Reward / Training Pipeline** — Coherent multi-layer reward with clear ablation hooks; two-phase training (freeze Commander → train Oversight) reuses proven Bastion v1 recipe.

---

Built on top of [Bastion v1](https://huggingface.co/spaces/Astro-Dude/bastion) (Round 1 submission).
