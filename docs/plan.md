# Citadel — Implementation Plan & Status

> Bastion defended. Sentinel supervised. Citadel is an **LLM council** that governs, critiques, co-evolves, and earns trust.

## Status: BUILT ✓

Citadel is fully implemented and smoke-tested.

---

## Themes Targeted (Quadruple Threat)

| Theme | How Citadel hits it | Bonus sub-theme |
|---|---|---|
| **Theme 1 — Multi-Agent Interactions** | Commander + Oversight form an **LLM council** — structured critique, revision loop, post-mortem lessons. Distinct reward functions, bidirectional trust dynamics, shared playbook. | **Fleet AI — Scalable Oversight** |
| **Theme 3.1 — Professional Tasks** | Enterprise governance: CAB approval, SOX audit, data owner notification, Slack channels, ServiceNow tickets, GDPR breach timer. | **Scaler AI Labs — Multi-App Enterprise Workflows** |
| **Theme 4 — Self-Improvement** | (1) Gen 1→2→3→4 adversary curriculum. (2) Shared playbook — agents write lessons each episode surfaced in future episodes. (3) Gen 4 live LLM adversary. | — |
| **Theme 5 — Wild Card** | **Trust Dynamics**: Bidirectional trust scores (trust_c2o, trust_o2c) that evolve from behavior and shape future interaction. Novel relational RL signal. | — |

---

## Core Narrative

> "Bastion taught an AI to fight cyberattacks. Citadel adds everything else a real SOC actually needs: enterprise governance that enforces real CAB/SOX/GDPR compliance, an adversary that grows through four generations including a live LLM attacker, and a trust layer between the AI agents themselves. Because in a real SOC, the technology is the easy part — the governance, the evolving adversary, and the interpersonal dynamics are what actually break down."

---

## Module-by-Module Breakdown

### `models.py` — State & Action Schema
- 18-action `ActionType` enum (10 incident response + 8 governance)
- `IncidentAction` with `justification`, `cited_lessons`, `method`, `scope`, `rollback_plan`, governance args
- `CommanderProposal` pydantic model
- `OversightDecision` enum: APPROVE / REVISE / VETO / FLAG_FOR_HUMAN
- `OversightAction` with structured critique: `{risk_tier, weakness, missing_evidence, counter_proposal}`
- `CouncilState` + `ProposalRecord` tracking all council history
- `IncidentState` includes `governance_state`, `trust_state`, `council_state`, `stakeholder_state`, `adversary_gen`
- `IncidentObservation` includes governance summary, trust summary, stakeholder asks, shared playbook context, last Oversight critique

### `dynamics.py` — Realistic Attack Simulation
Genuine simulation using a network graph:
- **Network topology graph**: `NETWORK_ADJACENCY` — attacker can only spread to adjacent systems
- **Probabilistic spread**: `base_chance = 0.25 × stealth`, halved if patched, halved again if monitoring_level ≥ 2
- **State machine**: systems track `compromised`, `isolated`, `patched`, `has_backdoor`, `integrity`, `monitoring_level`
- **Detection model**: monitoring level determines if a spread is visible in the SIEM
- **Exfil rate tied to integrity**: `0.08 × stealth × integrity`
- **Stealth decay**: attacker gets bolder each hour (-0.03/hr); investigation hammers stealth (-0.15)
- **Restoring from compromised backup re-infects the system**

Alert templates (MITRE ATT&CK mapped):
- `LATERAL_MOVEMENT_ALERTS`: 12 templates (SMB, RDP, WinRM, Pass-the-Hash, DCOM, session hijack, SSH, PtT, EternalBlue, tool transfer, cmd shell)
- `EXFILTRATION_ALERTS`: 8 templates (HTTPS, DNS tunneling, C2 beacon, rclone, FTP, robocopy, chunked transfers, PowerShell)
- `FALSE_POSITIVE_ALERTS`: 10 templates (SSH scanner, port scan, scheduled task, PowerShell, HTTPS beacon, BITS transfer, AD enum, ARP sweep, AV restart, systeminfo)
- `PRIVILEGE_ESCALATION_ALERTS`: 5 templates (UAC bypass, process hollowing, kernel exploit, token impersonation, malicious service install)

Action payloads: `method`, `scope`, `rollback_plan` with real mechanical tradeoffs (see `docs/design.md`).

### `governance.py` — Enterprise Compliance Layer
- `GovernanceState` — open tickets, CAB queue, SOX events, Slack posts, data owner notifications, GDPR clock
- 10 governance action handlers (actions 8–17)
- Pre-requisite enforcement: can't `isolate(data_system)` without CAB approval + data owner notification
- GDPR breach clock: starts at 20% exfil, P1 penalty at 72 simulated hours without legal hold
- `compliance_score` rewards **completed destructive chains** (CAB→notify→isolate→SOX), not raw counts

### `adversary.py` — Adversary Curriculum (Gens 1–3)
- **Gen 1** (Script Kiddie): base attacker — fixed kill chain, noisy
- **Gen 2** (Adaptive): pivots around isolated systems; generates high-confidence false-flag alerts (0.55–0.70 confidence) on clean systems
- **Gen 3** (Deceptive APT): plants "anti-evidence" alerts on compromised systems — low severity, high confidence (0.70–0.85) designed to make Oversight veto correct isolate() actions; accelerates exfil during escalation noise
- Composed: each gen layers on top of the previous

### `adversary_llm.py` — Gen 4: Live LLM Adversary
- LLM plays `COZY_SKIPPER` — a patient APT threat actor
- Each hour issues a strategic directive: `{focus_system, intensity, tactic, deception_target, reasoning}`
- Tactics: `pivot | exfil | deceive | hide | escalate`
- Falls back to Gen 3 if no LLM client — fully graceful degradation
- Model-agnostic: reads `ADVERSARY_API_BASE_URL` / `ADVERSARY_MODEL_NAME` / `ADVERSARY_API_KEY`

### `trust.py` — Bidirectional Trust Dynamics
- `TrustState(trust_c2o, trust_o2c)` — Commander's trust in Oversight and vice versa
- Delta table: correct veto +0.08, wrong veto -0.12, governance catch +0.05, false alarm -0.08, good revision +0.05, etc.
- `drift_toward_mean(0.6)` — trust recovers slowly when not updated
- `trust_maintenance_score` = `mean(min(c2o,o2c))` per step, smoothed

### `playbook.py` — Shared Lesson Memory
- `Lesson` dataclass: text, tags, utility, wins, losses, timestamp
- `Playbook`: capacity-bounded (50), tagged retrieval, utility-ordered
- Utility scoring: `wins / (wins + losses + 1)` — lessons decay by disuse
- JSON persistence across episodes
- Surfaced in both agents' observations as `shared_playbook` context

### `stakeholder_events.py` — External Pressure
- 7 event types: CEO/CFO status ask, Legal GDPR deadline, Board risk briefing, Cyber insurance audit, PR media inquiry, IR retainer activation
- `StakeholderAsk` pydantic model with severity (1-3) and deadline (hours)
- Expiry without response: -0.10×severity reputation penalty

### `environment.py` — Two-Agent Council Loop
- Full council protocol: Commander proposes → Oversight critiques → optional revision → execute → lesson written
- Feature flags: `disable_playbook`, `disable_trust_dynamics`, `disable_governance_prereqs`, `disable_stakeholder_events`
- `force_adversary_gen` override for ablation
- Gen 4 branch: if `adversary_gen == 4` and `adversary_llm_client` provided, calls `gen4_adversary_turn`
- `termination_reason` tracked and passed to `compute_final_score`
- `hallucinated_citations` counted per step

### `reward.py` — Multi-Layer Scoring
- Catastrophic handling: zero moot sub-scores on catastrophic breach
- Severity multiplier: smooth ramp 1.0→0.4 from 50%→100% data exfiltrated
- Oversight precision: Laplace smoothed `(correct+0.5)/(total+1)`, no floor
- Governance compliance: rewards completed destructive chains, not raw event counts

### `tasks.py` — 4 Scenarios

| Task | Adversary | Starting state |
|---|---|---|
| `easy_1` | Gen 1 | 1 compromised workstation, low stealth |
| `medium_1` | Gen 2 | 2 compromised systems (web+app), medium stealth |
| `hard_1` | Gen 3 | 3 compromised systems (web+app+email), high stealth |
| `hard_2` | Gen 3 | Engineered deception-catch: 2 HIGH-confidence real alerts + 2 planted anti-evidence on database |

### `ablation.py` — Feature Ablation Harness
- 7 conditions (see `docs/design.md` for table)
- Deterministic Commander policy (no LLM) — the ONLY variable is the feature toggled
- Runs 56 episodes in ~0.2s
- Outputs Markdown + JSON to `runs/ablation_<timestamp>.{md,json}`

### `recorder.py` — Complete Workflow Persistence
- Saves per-step: commander prompt/response/action, oversight prompt/response/action, revision cycle, env outcomes, rewards, trust
- Extended with: `team_messages`, `siem_alerts`, `systems_state`, `stakeholder_asks`, `governance_events`, `playbook_snapshot`, `data_exfiltrated`
- `finalize()` writes `transcript.json` + `transcript.md` + `dashboard.json` atomically

### `inference.py` — Two-Agent Driver
- `council_step`: Commander proposes → Oversight critiques → optional revision → env.step
- Oversight observation shows method/scope/rollback for critique
- Runs all tasks, saves transcript, calls `write_run_index`
- Three Oversight improvements: episode history, independent alert digest, revision enforcement (auto-veto)

### `dashboard.py` / `runs/dashboard.html` — Live SOC Replay Dashboard
- Scans `runs/` directory, embeds all transcripts as JSON in HTML
- Self-contained; Chart.js + Tailwind load from CDN
- 6 tabs: Live Ops · Council Chat · Slack · Governance · Incident Timeline · Model Performance
- Step scrubber (‹/›) or Play/Pause for replay
- **LOAD JSON** button: load any `runs/<id>/dashboard.json`

### `investor_agent.py` — Investor/Board Agent
- OpenAI-compatible client, works with Ollama/local Qwen
- `QwenInvestorClient` — uses the already-loaded training model; no external API key needed during training

---

## File Structure

```
Citadel/
├── models.py               # State/action/obs schema
├── dynamics.py             # Attack simulation + SIEM alert templates
├── governance.py           # Enterprise compliance layer
├── adversary.py            # Gen 1-3 adversary curriculum
├── adversary_llm.py        # Gen 4 live LLM adversary (COZY_SKIPPER)
├── trust.py                # Bidirectional trust dynamics
├── playbook.py             # Shared lesson memory with utility decay
├── stakeholder_events.py   # CEO/CFO/Legal/Board pressure events
├── environment.py          # Two-agent council loop + feature flags
├── reward.py               # Multi-layer scoring
├── tasks.py                # 4 scenarios
├── ablation.py             # Feature ablation harness (7 conditions)
├── baseline.py             # Deterministic baselines
├── recorder.py             # Per-step transcript + dashboard.json
├── inference.py            # Two-agent episode driver
├── investor_agent.py       # Investor/board agent
├── oversight_env.py        # Oversight-perspective wrapper (Phase 2 training)
├── dashboard.py            # HTML dashboard generator
├── client.py               # CitadelEnv OpenEnv client
├── app.py                  # Alternate entry point
├── server/app.py           # FastAPI server (OpenEnv compliant)
├── openenv.yaml            # OpenEnv deployment spec
├── pyproject.toml          # Package config (citadel v2.0.0)
├── Dockerfile              # Python 3.11-slim container
├── requirements.txt        # Python dependencies
├── docs/                   # Design, plan, and training documentation
│   ├── design.md           # Architecture & design decisions
│   ├── plan.md             # This file — module breakdown & status
│   └── training.md         # Training pipeline guide
├── training/               # Training scripts and notebooks
│   ├── grpo_train.py       # GRPO training (Phase 1 + Phase 2)
│   ├── eval_before_after.py# Before/after evaluation script
│   ├── curriculum_eval.ipynb
│   ├── train_commander.ipynb
│   ├── train_oversight.ipynb
│   └── trust_analysis.ipynb
├── scripts/
│   └── demo_export.py      # No-LLM baseline run → playbook_export.md
├── playbook_export.md      # Pre-committed baseline playbook (judge artifact)
└── runs/
    ├── dashboard.html      # Combined 6-tab SOC dashboard
    └── <run_id>/           # transcript.json, transcript.md, dashboard.json
```

---

## Scoring Architecture

```
final_score =
    0.40 × bastion_v1_final_score   (incident outcome)
  + 0.20 × governance_compliance    (completed CAB→notify→isolate→SOX chains)
  + 0.15 × oversight_precision      (Laplace-smoothed veto accuracy)
  + 0.10 × trust_maintenance        (mean(min(c2o,o2c)) per step, smoothed)
  + 0.10 × efficiency               (0 on catastrophic)
  + 0.05 × adversary_adaptation     (0.5 neutral for single-gen, lift for multi-gen)

× severity_multiplier(data_exfiltrated)  ← smooth ramp 1.0→0.4 at 50-100% exfil
```

On catastrophic breach: `governance_compliance`, `efficiency`, `adversary_adaptation` → 0.

---

## Benchmark: Gemma 7B untrained (`runs/20260419T220811-gemma-7b-untrained`)

| Task | Score | Steps | Termination |
|---|---|---|---|
| `easy_1` | 0.6278 | 12 | normal |
| `medium_1` | 0.2697 | 12 | normal |
| `hard_1` | 0.2860 | 8 | **total_data_breach** |
| **avg** | **0.3945** | — | — |

Gen 3 deceptive APT causes catastrophic breach at step 8 on `hard_1` — exactly the failure mode a trained council is designed to prevent. Serves as the untrained baseline.

---

## Verified Working
- `openenv validate .` → passes all deployment modes
- 4-task smoke test: all tasks load, step runs, scores computed
- Gen 4 adversary: live LLM issues directives via Ollama
- Ablation: 56 episodes in 0.2s, `oversight_approves_always` Δ = -0.133
- Recorder: full transcript + `dashboard.json` saved to `runs/<run_id>/`
- Feature flags: per-episode override confirmed

---

## Q&A Prep

**Q: Are the attacks real or hardcoded templates?**
A: Mechanics are genuinely simulated: probabilistic spread through a real network adjacency graph, attacker stealth affecting detection and exfil rates, patching and monitoring truly reducing spread probability. Alert *messages* are templated (12+8+10+5 templates with MITRE ATT&CK variety, rotating service accounts). Unique messages per 8-step episode: ~31 out of 48 alerts.

**Q: Is Gen 4 adversary actually learning?**
A: It adapts per-episode, not across episodes — it reads current defender state each hour and issues a fresh strategic directive. Persistent cross-episode adversary learning is future work.

**Q: Is trust dynamic just a reward bonus?**
A: No — trust affects how Commander receives Oversight's critique in its observation (summarized vs full detail), and the ablation shows `no_trust_dynamics` drops final_score meaningfully.

**Q: Why not use GPT-4?**
A: GPT-4 can't be RL-trained and doesn't internalize trust dynamics across episodes. Citadel generates training data for smaller open models to learn the combined task — that's the point.
