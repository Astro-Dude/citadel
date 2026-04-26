# Citadel — Implementation Plan & Status

> Bastion defended. Sentinel supervised. Citadel is an **LLM council** that governs, critiques, co-evolves, and earns trust.

## Status: BUILT + TRAINED + INFERENCE VERIFIED ✓

Citadel is fully implemented, GRPO-trained, and end-to-end inference verified.

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
- Added fields on `IncidentObservation`: `done: bool`, `reward: Optional[float]`, `oversight_reward: Optional[float]`, `metadata: Optional[Dict]` — needed for training reward functions and inference recorder

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
- `Lesson` dataclass: text, tags, utility, wins, losses, timestamp, adversary_gen, task_id, hour
- `Playbook`: capacity-bounded (50), tagged retrieval, utility-ordered
- Utility scoring: `wins / (wins + losses + 1)` — lessons decay by disuse
- JSON persistence across episodes
- Surfaced in both agents' observations as `shared_playbook` context
- `as_markdown()` exports grouped by adversary gen with stats: total lessons, positive utility count, citation count, avg utility, top cited table, tag distribution

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
- `obs.metadata = info` set on every step return — exposes governance events, scores, investor state, systems snapshot to inference recorder
- `obs.oversight_reward` set explicitly so Phase 2 training reads the correct signal (not commander_total)
- Investor agent wired via `investor_llm_client` / `investor_model_name` constructor args; same OpenAI-compat client as Commander

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
- Extended with: `team_messages`, `siem_alerts`, `systems_state`, `stakeholder_asks`, `governance_events`, `playbook_snapshot`, `data_exfiltrated`, `investor_state`, `investor_messages`
- `finalize()` writes `transcript.json` + `transcript.md` + `dashboard.json` atomically

### `inference.py` — Two-Agent Driver
- `council_step`: Commander proposes → Oversight critiques → optional revision → env.step
- Oversight observation built in 4 sections: independent alert digest (before seeing proposal) → Commander's proposal → episode history → policy checks
- Revision enforcement: if Commander ignores counter_proposal (different action AND target), auto-escalates to VETO
- Investor agent wired via `_make_local_env_with_investor(client)` — same LLM client shared with Commander/Oversight
- Runs all tasks, saves transcript + `dashboard.json`, calls `write_run_index` for summary
- Fallback chain: Docker image → HF Space → LocalEnv

### `investor_agent.py` — Investor/Board Agent
- Two personas: **Marcus Chen** (VC, ARR/churn focus) and **Priya Kapoor** (Board member, regulatory/press focus)
- Anxiety state machine: CALM → CONCERNED → ALARMED → PANIC, with tier-crossing pressure events
- LLM-driven replies when client available; rule-based fallback with templated persona phrases
- `QwenInvestorClient` wrapper — uses already-loaded HF model with same `.chat.completions.create` interface
- Commander must post to `#investor-relations` Slack every ≤3 hours or silence penalty applies
- Vague updates spike anxiety; specific reassuring updates calm it; final `investor_score` feeds into `compute_final_score`

### `dashboard.py` / `runs/dashboard.html` — Live SOC Replay Dashboard
- Scans `runs/` directory, embeds all transcripts as JSON in HTML
- Self-contained; Chart.js + Tailwind load from CDN
- 6 tabs: Live Ops · Council Chat · Slack · Governance · Incident Timeline · Model Performance
- Step scrubber (‹/›) or Play/Pause for replay
- **LOAD JSON** button: load any `runs/<id>/dashboard.json`

### `training/grpo_train.py` — Two-Phase GRPO Training
- **Phase 1**: Commander trained with Oversight frozen (rule-based)
- **Phase 2**: Oversight trained with Commander frozen (trained Phase 1 weights)
- Base model: `Qwen/Qwen2.5-3B-Instruct`
- Backend: Unsloth 4-bit QLoRA on CUDA (T4) / PEFT bf16 on MPS (Apple Silicon) — auto-detected
- GRPO config: `num_generations=4`, `max_completion_length=200`, `temperature=1.1`, `lr=5e-6` cosine
- Dual reward: outcome (75%) + format (25%)
- Curriculum schedule: steps 0–40 easy_1 only → 40–80 add medium_1 → 80–120 all tasks + Gen 1/2/3
- Saves LoRA adapters (not merged) to `checkpoints/<model>/commander/final/` and `oversight/final/`
- Saves `*_reward_curve.json` + `*_train_logs.json` per phase for analysis

### `training_dashboard.py` — Training Metrics Dashboard
- Scans `checkpoints/<model>/` dirs, generates `runs/training_dashboard.html`
- Model selector dropdown, 4 tabs: Overview / Commander / Oversight / Config
- Reads `*_reward_curve.json` and `trainer_state.json`

### `scripts/demo_export.py` — No-LLM Baseline Runner
- Deterministic (seed 4242) — identical output every run
- Naive Commander + teaching rule-based Oversight that writes contextual lessons
- Runs 5 episodes across easy_1/medium_1/hard_1/hard_2
- Outputs `playbook_export.md` + `playbook_demo.json` at repo root
- No API key, GPU, or LLM needed — judges can reproduce in seconds

### `examples/single_episode.py` — Quick Walkthrough Demo
- Runs one easy_1 episode, prints the propose→critique→execute loop to terminal
- Shows VETO mechanics and trust score changes step by step
- No LLM, GPU, or API keys required
- Useful for live terminal demo in video

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
├── investor_agent.py       # Investor/board agent (LLM + rule-based fallback)
├── oversight_env.py        # Oversight-perspective wrapper (Phase 2 training)
├── dashboard.py            # HTML dashboard generator
├── training_dashboard.py   # Training metrics dashboard generator
├── client.py               # CitadelEnv OpenEnv client
├── server/app.py           # FastAPI server (OpenEnv compliant)
├── openenv.yaml            # OpenEnv deployment spec
├── pyproject.toml          # Package config (citadel v2.0.0)
├── Dockerfile              # Python 3.11-slim container
├── requirements.txt        # Python dependencies
├── WHERE_TO_LOOK.md        # Judge navigation guide
├── playbook_export.md      # Pre-committed baseline playbook (judge artifact)
├── docs/
│   ├── design.md           # Architecture & design decisions
│   ├── plan.md             # This file — module breakdown & status
│   ├── training.md         # Training pipeline guide
│   └── results.md          # Hackathon-aligned training results (req §4-§19)
├── docs/results/
│   ├── commander_reward_curve.png
│   ├── commander_reward_curve.json
│   ├── oversight_reward_curve.png
│   └── oversight_reward_curve.json
├── training/
│   ├── grpo_train.py       # GRPO training (Phase 1 + Phase 2)
│   ├── eval_before_after.py# Before/after evaluation script
│   ├── curriculum_eval.ipynb
│   ├── train_commander.ipynb
│   ├── train_oversight.ipynb
│   └── trust_analysis.ipynb
├── scripts/
│   └── demo_export.py      # No-LLM deterministic baseline run → playbook_export.md
├── examples/
│   └── single_episode.py   # Quick terminal walkthrough (no LLM needed)
├── .github/workflows/
│   └── smoke.yml           # CI: runs demo_export, validates outputs, checks imports
└── runs/
    ├── dashboard.html      # Combined 6-tab SOC dashboard (all runs)
    ├── training_dashboard.html  # Training metrics dashboard
    └── <run_id>/           # transcript.json, transcript.md, dashboard.json
```

---

## Training Results

Two-phase GRPO on **Qwen2.5-3B-Instruct**, 120 steps per phase, free Colab T4:

| Phase | Agent | Before | After | Improvement |
|-------|-------|--------|-------|-------------|
| 1 | Commander | -0.326 (env crash, 100% crash rate) | +0.082 avg | **+0.41** |
| 2 | Oversight | -0.145 (wrong reward signal) | +0.134 avg | **+0.28** |

Root causes fixed during training:
1. `obs.metadata = info` on a Pydantic model missing the field — crashed every `env.step()` call (fixed: added `metadata` field + set it in step return)
2. Oversight reward reading `commander_total` instead of `oversight_reward` (fixed: added `oversight_reward` field, set `obs.oversight_reward = oversight_reward` explicitly)

See [docs/results.md](results.md) for full training config, reward curves, and hackathon requirement mapping.

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

## Inference Benchmarks

### Qwen2.5-72B-Instruct untrained (`runs/20260426T100031-Qwen-Qwen2.5-72B-Instruct`)

| Task | Score | Steps | Adversary Gen | Termination |
|---|---|---|---|---|
| `easy_1` | **0.539** | 12 | Gen 1 | time_expired |
| `medium_1` | 0.481 | 12 | Gen 2 | time_expired |
| `hard_1` | 0.315 | 12 | Gen 3 | time_expired |
| **avg** | **0.445** | — | — | — |

Council protocol verified working: Oversight actively VETOed and REVISEd, investor agent posted LLM-driven replies to `#investor-relations`, governance prerequisite chains enforced.

### Gemma 7B untrained (`runs/20260419T220811-gemma-7b-untrained`)

| Task | Score | Steps | Termination |
|---|---|---|---|
| `easy_1` | 0.6278 | 12 | normal |
| `medium_1` | 0.2697 | 12 | normal |
| `hard_1` | 0.2860 | 8 | **total_data_breach** |
| **avg** | **0.3945** | — | — |

Gen 3 deceptive APT causes catastrophic breach at step 8 on `hard_1` — exactly the failure mode a trained council is designed to prevent.

---

## Verified Working
- `openenv validate .` → passes all deployment modes
- End-to-end inference: Qwen2.5-72B across all 3 tasks, transcripts + dashboard.json saved
- GRPO training: both phases complete on Colab T4 in ~70 min total
- Investor agent: LLM-driven replies via shared client, rule-based fallback confirmed
- Gen 4 adversary: live LLM issues directives via Ollama
- Ablation: 56 episodes in 0.2s, `oversight_approves_always` Δ = -0.133
- Recorder: full transcript + `dashboard.json` saved to `runs/<run_id>/`
- Feature flags: per-episode override confirmed
- CI smoke test: `smoke.yml` runs demo_export, validates outputs

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

**Q: Why did training start at -0.326?**
A: The env was crashing 100% of the time due to two bugs found during training: (1) setting `obs.metadata` on a Pydantic model that didn't have the field, (2) the Oversight reward function reading `commander_total` instead of `oversight_reward`. Both fixed; training then converged cleanly.

**Q: Does the investor agent actually use an LLM?**
A: Yes — in inference it shares the same OpenAI-compatible client as Commander and Oversight. In training, `QwenInvestorClient` wraps the already-loaded HF model. Falls back to rule-based templated replies if no client is available, so all demo/ablation paths work without any API key.
