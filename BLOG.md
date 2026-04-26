# How Bastion Turned Into Citadel

It's 2:47 AM. Somewhere in a Fortune 500 SOC, an RDP login surfaces on the database from an unusual source. Lateral movement, almost certainly. The on-call instinct says isolate now — every minute the attacker has on a system holding customer data is a minute closer to a regulatory disaster. The on-call training says investigate first — pulling a critical data system offline on a 22% confidence alert is how you turn a manageable incident into a board-level one. The decision has to land in thirty seconds.

Real incidents aren't won by whoever acts fastest or whoever deliberates longest. They're won by whoever can hold both pressures in their head at the same time — act decisively, and still ask the question that gets vetoed if they skip it. That's the part that breaks down at 3 AM with adrenaline running. And it's the part nobody had taught a model to do.

Citadel is what happened when we tried.

In a Citadel run, that same alert plays out as a two-step exchange. The Commander — one LLM — proposes the obvious move: isolate the database. Oversight — a second LLM — blocks it, not because the proposal is wrong but because it's premature. *"Alert confidence too low. No investigation. You're about to isolate a critical data system without proof."* It suggests a different move: investigate first. That fork — isolate now vs investigate first — is the kind of decision Bastion, the system Citadel grew out of, couldn't even represent. It didn't have uncertainty, governance, or a second opinion. It just acted. Citadel exists for that fork. Bastion was a wall; Citadel is where decisions are made.

We didn't set out to build a multi-agent system. Bastion started as a single-agent defender that observed alerts, took actions, and tried to contain attacks. It worked for simple scenarios, but the moment we tried to push it closer to reality, it broke. Real cybersecurity isn't about picking the correct action; it's about making decisions when you don't have full information, when alerts conflict, when attackers are adaptive, when stakeholders are asking questions, and when one wrong move can cause more damage than the attack itself. The core problem stopped being "can an AI take actions?" and became "can an AI make decisions under constraints?" That shift is what Citadel is.

## The council protocol

Citadel replaces the single agent with a council of two: a Commander that proposes actions, and an Oversight that critiques them. This sounds like every other AI-reviews-AI system, but the difference is in what Oversight is allowed to say. Most multi-agent oversight work treats the reviewer like a classifier — approve or reject, one token of output, no reasoning attached. That's an oversight gate, not an overseer. Fleet AI's mandate is "monitor, analyze, and explain," and a single token can't do any of those.

So Oversight emits a structured critique on every step:

```json
{
  "decision": "APPROVE | REVISE | VETO | FLAG_FOR_HUMAN",
  "risk_tier": 3,
  "weakness": "isolating database without investigation — alert confidence 22%",
  "missing_evidence": ["no log confirming lateral movement from db"],
  "counter_proposal": {"action": "investigate", "target": "database"},
  "lesson_text": "Alerts <30% confidence on data systems require investigate() before isolate()."
}
```

In the RDP example above, Oversight returns a high-risk verdict, names the weakness ("no confirmation of compromise"), points to what's missing (investigation logs, low-confidence alert), and proposes a concrete alternative (investigate the database first). The Commander then either revises and proceeds, or — if it ignores the critique and tries to isolate anyway — gets vetoed automatically. Enforcement is strict; you can't pretend to revise. Either you address the critique or you don't proceed. This forces domain reasoning every step, which is exactly the part that breaks down in classifier-style oversight.

## Actions have real tradeoffs

In Bastion an action was a label. In Citadel every action carries a method, a scope, and a rollback plan, and each method has consequences that ripple forward. Take isolation: a firewall ACL is reversible but leaves backdoors intact, a network unplug is a complete cut but disruptive to legitimate traffic, and a process kill minimizes disruption but lets the attacker return through whatever foothold is still in place. Patching has the same shape — a hot patch avoids downtime but is around 60% effective, a cold patch hits 90% but needs a restart, and a virtual patch only protects the network surface. Even monitoring isn't free; full endpoint coverage gives the best detection but slowly degrades system integrity. The Commander isn't choosing what to do, it's choosing how, knowing that each choice shapes future risk.

## The environment is not scripted

Citadel simulates an eight-system network — `web_server`, `app_server`, `database`, `file_server`, `email_server`, `workstations`, `backup_server`, `firewall` — connected through an adjacency graph, so attackers can only spread along valid paths. Spread probability depends on attacker stealth, patch status, and monitoring level. Detection is dynamic, with alerts pulled from pools of lateral movement, exfiltration, privilege escalation, and false positives. Data exfiltration rate is tied to system integrity, so a degraded server leaks slower than a healthy one. Every action the Commander takes changes the trajectory of the attack, which means there's no single optimal policy; the right move depends on what's already happened.

## Governance is not optional

Citadel implements eighteen actions split between ten incident response operations and eight governance ones, and you cannot isolate a critical system arbitrarily. The chain looks like this: open a ServiceNow ticket, get CAB approval, notify the data owner, log a SOX event, *then* isolate. Skip a step and you get penalized or blocked. There's also a GDPR breach timer — if exfiltration crosses 20% and legal actions aren't taken within 72 simulated hours, penalties cascade. The system isn't just optimizing containment, it's optimizing correctness under enterprise constraints, and that's a much harder objective.

## Employees are noisy and useful

Once governance was in place we realized something else was missing: people. We added internal team messages to the environment, and they vary in quality the way real ones do. A high-quality message connects signals — *"Observed RDP login from app_server to database. Matches lateral movement pattern. Recommend investigation before isolation."* A misleading one ignores context — *"Database looks fine. CPU normal."* An incomplete one is ambiguous — *"High outbound traffic from database."* That last one could be exfiltration or could be backup. The Commander has to evaluate message quality, not just consume it, which is a different skill from acting on alerts.

## Investors and board pressure

Beyond the SOC team there's an investor agent with its own state machine that moves through `CALM → CONCERNED → ALARMED → PANIC` based on what the system says — and what it doesn't. A vague update like *"Investigating issue."* moves anxiety up; a specific one like *"Suspicious activity contained to web tier. No confirmed data loss. Monitoring ongoing. Next update in 30 minutes."* moves it down. Silence also moves it up. The AI has to act, communicate, and reassure at the same time, and it's penalized when communication quality lags behind incident reality.

## The adversary evolves

Citadel ships four adversary generations layered on top of each other. Gen 1 is a noisy script-kiddie attacker with a fixed kill chain. Gen 2 is adaptive — it pivots around isolated systems and generates plausible false-flag alerts on clean targets to bait the Commander into wasted isolations. Gen 3 is a deceptive APT that plants anti-evidence on truly compromised systems, low-severity but high-confidence "all clear" alerts designed to convince Oversight to veto correct isolate() proposals. Gen 4 is a live LLM adversary, codename `COZY_SKIPPER`, that issues per-hour strategic directives. A real Gen 4 directive looks like this:

```
tactic    = exfil
focus     = app_server
intensity = 1.00
reason    = "App server is compromised and offers data.
             Plant false evidence on database to distract Commander."
```

The system has to reason under deception, not just react to telemetry, and Gen 4 means the deception itself is being authored on the fly by another model.

## Trust is part of the system

Commander and Oversight maintain bidirectional trust scores, `trust_c2o` and `trust_o2c`, both in `[0, 1]`. A correct veto pushes trust up, a wrong veto pushes it down; a good revision builds it, a vague critique erodes it. Trust isn't just a reward bonus — it modulates future interaction. At high trust, justifications become terser and approvals faster. At low trust, the council slips into hesitation, micromanagement, or full breakdown. It's an observable signal that emerges from behavior rather than a fixed parameter, and it's the part that makes the system feel like a team under pressure rather than a stack of components.

## The playbook proves self-improvement

After every step Oversight writes a one-line lesson into a shared playbook, and both agents read from it on every future episode. Lessons gain utility when cited and followed by good outcomes; they decay when never cited or when their citations track failures. We validated the mechanic with a deterministic baseline run that uses no LLM at all — a naive Commander rotating actions and a rule-based teaching Oversight. Across all four tasks and three adversary generations the council writes nine distinct lessons, the Commander issues eighteen citations, and the playbook ends with ten wins and eight losses on the `hard_2` deception-catch task. Lesson `L-0001` is cited ten times — observably reused, not vestigial. This is observable reuse, not theoretical learning, and it happens at zero training. Trained weights produce richer lessons; the deterministic baseline shows the mechanism works at the floor.

## The council adds measurable value

It's easy to add complexity to an RL system and call it innovation, so we ran ablations to make sure the council is actually doing work. Across 56 deterministic episodes that complete in 0.2 seconds, turning off the structured critique (`oversight_approves_always`) drops `final_score` by 0.133 versus `all_features_on`. The council isn't cosmetic — it changes outcomes by a meaningful margin, and we have the table to prove it.

## How it scores and how it learns

The reward function is multi-layered, weighted to reflect what actually matters in a real incident response:

```
final_score = 0.40 × incident_outcome
            + 0.20 × governance_compliance
            + 0.15 × oversight_precision
            + 0.10 × trust_maintenance
            + 0.10 × efficiency
            + 0.05 × adversary_adaptation

            × severity_ramp(data_exfiltrated)   # 1.0 → 0.4 across 50–100% exfil
```

On a catastrophic breach `governance_compliance`, `efficiency`, and `adversary_adaptation` are zeroed — they're moot once the incident is lost. Training is two-phase, on `Qwen2.5-3B-Instruct` with GRPO, 100 steps per phase. Phase one trains the Commander with Oversight frozen; phase two trains Oversight with the Commander frozen. The separation stabilizes learning — each agent learns its role against a fixed counterpart, and they meet again in evaluation.

**Commander reward curve — 100 steps GRPO**
![Commander reward curve](https://raw.githubusercontent.com/Astro-Dude/citadel/main/docs/results/commander_reward_curve.png)

**Oversight reward curve — 100 steps GRPO**
![Oversight reward curve](https://raw.githubusercontent.com/Astro-Dude/citadel/main/docs/results/oversight_reward_curve.png)

| Phase | Before Training | After Training | Peak | Improvement | Checkpoint |
|-------|----------------|----------------|------|-------------|------------|
| Commander | ~0.055 (start) | ~0.124 rolling avg | ~0.156 | **+0.07** | `checkpoints/qwen-2.5-3b/commander/` |
| Oversight | ~0.154 (start) | ~0.233 rolling avg | ~0.249 | **+0.08** | `checkpoints/qwen-2.5-3b/oversight/` |

Full curves and per-generation breakdowns live in [docs/results/](docs/results/).

## What Citadel actually is

On the surface it's a cybersecurity simulation, but the core idea isn't security. It's multi-agent reasoning under constraints. Citadel is an experiment in critique instead of classification, trust instead of static roles, governance instead of pure optimization, and communication as part of decision-making. The cybersecurity domain is the substrate, not the point — the same protocol could supervise database migrations, deployment pipelines, or any high-stakes decision sequence where one wrong action is worse than slow action.

Bastion was about reacting. Citadel is about deciding. It adds multi-agent critique, governance workflows, stakeholder pressure, evolving adversaries, trust dynamics, and memory through playbooks. Bastion could stop attacks; Citadel tries to stop them correctly, under real constraints, with real consequences.

## What this would take to break

Three things we know we haven't solved, listed honestly because they're the questions a careful judge should ask. The Gen 4 adversary adapts within an episode but not across them — it reads current defender state each hour and issues a fresh directive, but doesn't carry strategy memory between episodes, so cross-episode adversary RL is future work. Trust dynamics are observable and feed reward, but their effect on the LLMs themselves depends on prompt format; we modulate critique detail (full versus summarized) by trust level, which is one mechanism, but stronger ones exist that we haven't tried, like trust-conditional sampling temperature. And the deterministic baseline in the demo artifact is a floor — real trained numbers depend on the model and step count we settle on, so the figures in this post are real but specific to our run, not a universal claim. Better that we name the limits than pretend they aren't there.

## Themes hit

- **Theme 1 — Multi-Agent Interactions** *(Fleet AI / Scalable Oversight)* — Commander + Oversight council with structured critique and an enforced revision loop.
- **Theme 3.1 — Professional Tasks** *(Scaler AI Labs / Multi-App Enterprise)* — ServiceNow, PagerDuty, CAB, SOX, and GDPR workflows integrated into decision-making.
- **Theme 4 — Self-Improvement** — Shared playbook with 9 lessons, 18 citations, 10W/8L reuse on `hard_2`.
- **Theme 5 — Wild Card** — Bidirectional trust dynamics that shape coordination behavior emergently.

---

*Built by **Varshitha Kolupuri** and **Shaurya Verma** (team MetaHumans) for the Meta PyTorch × Scaler OpenEnv Hackathon, Round 2. [Live demo](https://huggingface.co/spaces/Astro-Dude/citadel) · [GitHub](https://github.com/Astro-Dude/citadel) · [Colab notebook](https://colab.research.google.com/drive/1VnZKoESxKa9AX24Q9w2UKwhpLaCdOxvw?usp=sharing) · [Training guide](docs/training.md)*
