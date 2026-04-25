"""
Citadel — Simulated Investor Slack Channel

Models two nervous investor personas watching the incident unfold in a
simulated #investor-relations Slack channel. Commander must keep them
calm with specific, confident updates. Investor anxiety feeds back into
the environment as stakeholder pressure and into the final score.

Two personas (randomly assigned per episode):
  Marcus Chen  — patient VC, asks about ARR risk and customer impact
  Priya Kapoor — board member, asks about regulatory exposure and press

Anxiety state machine:
  CALM      [0.00, 0.35) — "Keep me posted, handling it well"
  CONCERNED [0.35, 0.65) — "This is worrying, what's the timeline?"
  ALARMED   [0.65, 0.85) — "I need daily briefings, looping in LP committee"
  PANIC     [0.85, 1.00] — "Calling emergency board meeting"

Fallback: if no LLM client supplied, anxiety changes are purely formula-
driven and replies are templated. All demo/ablation/smoke-test paths work
without any API key.
"""

from __future__ import annotations

import json as _json
import random
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# QwenInvestorClient — drop-in LLM client using an already-loaded HF model
# ---------------------------------------------------------------------------

class QwenInvestorClient:
    """
    Wraps a HuggingFace pipeline (or Unsloth model+tokenizer pair) with the
    same .chat.completions.create interface that InvestorAgent expects.
    Pass this as llm_client= instead of an OpenAI client.
    """

    class _Choice:
        def __init__(self, content: str):
            self.message = type("M", (), {"content": content})()

    class _Response:
        def __init__(self, content: str):
            self.choices = [QwenInvestorClient._Choice(content)]

    def __init__(self, model, tokenizer, max_new_tokens: int = 200):
        self._model = model
        self._tok = tokenizer
        self._max_new = max_new_tokens
        self.chat = self  # so client.chat.completions.create works
        self.completions = self

    def create(self, model=None, messages=None, max_tokens=200, temperature=0.7, stream=False, **_):
        """Mimic openai.chat.completions.create."""
        import torch
        messages = messages or []
        # Build prompt using tokenizer's chat template
        text = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tok(text, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=min(max_tokens, self._max_new),
                temperature=max(temperature, 0.01),
                do_sample=temperature > 0,
                pad_token_id=self._tok.eos_token_id,
            )
        # Decode only the newly generated tokens
        gen = out[0][inputs["input_ids"].shape[-1]:]
        content = self._tok.decode(gen, skip_special_tokens=True).strip()
        return QwenInvestorClient._Response(content)


# ---------------------------------------------------------------------------
# Personas
# ---------------------------------------------------------------------------

PERSONAS = {
    "marcus_chen": {
        "name": "Marcus Chen",
        "role": "Lead Investor (VC)",
        "slack_handle": "@marcus.chen",
        "focus": "ARR risk, customer churn, SLA breaches",
        "style": "analytical, patient at first but data-driven escalation",
        "calm_phrases": [
            "Thanks for the update Marcus — appreciate the transparency.",
            "Good to know you're on top of it. Keep me in the loop.",
            "Understood. What's the customer impact so far?",
        ],
        "concerned_phrases": [
            "This is more serious than I expected. What's your containment timeline?",
            "I need to understand the ARR exposure here. How many customers affected?",
            "We should discuss this on a call. When are you available?",
        ],
        "alarmed_phrases": [
            "I'm looping in our LP committee — they'll want visibility.",
            "Our portfolio review is next week. I need a written status by EOD.",
            "This is approaching material disclosure territory. Legal is asking questions.",
        ],
        "panic_phrases": [
            "I'm calling an emergency board session for tomorrow morning.",
            "We're reconsidering our Q3 tranche pending a full security audit.",
            "Our legal team is now involved. Expect formal communications.",
        ],
    },
    "priya_kapoor": {
        "name": "Priya Kapoor",
        "role": "Board Member",
        "slack_handle": "@priya.kapoor",
        "focus": "regulatory exposure, press coverage, brand damage",
        "style": "direct, governance-focused, quick to escalate",
        "calm_phrases": [
            "Noted. Make sure legal is in the loop.",
            "Good. Ensure this is properly documented for the board record.",
            "Understood. What's our regulatory notification posture?",
        ],
        "concerned_phrases": [
            "The board needs a written incident brief within 24 hours.",
            "Have we assessed our GDPR notification obligations?",
            "Press are starting to ask questions — do we have a holding statement?",
        ],
        "alarmed_phrases": [
            "I'm convening an emergency governance committee call.",
            "Our D&O insurers need to be notified. Has legal been engaged?",
            "I'm hearing about this from outside — that's unacceptable.",
        ],
        "panic_phrases": [
            "I'm recommending the board consider external forensic consultants.",
            "This is a board-level crisis. I'm activating our crisis comms firm.",
            "We need an independent post-incident review. This cannot wait.",
        ],
    },
}

# Hours at which the investor proactively checks in (if no update received)
CHECKIN_HOURS = [2, 5, 8]

# Anxiety tier thresholds
TIER_CALM = 0.35
TIER_CONCERNED = 0.65
TIER_ALARMED = 0.85


# ---------------------------------------------------------------------------
# Investor state (serialisable — lives in IncidentState)
# ---------------------------------------------------------------------------

@dataclass
class InvestorMessage:
    hour: int
    direction: str          # "investor" | "commander"
    text: str
    anxiety_before: float
    anxiety_after: float


@dataclass
class InvestorState:
    persona_id: str = "marcus_chen"
    anxiety: float = 0.20   # starts calm
    last_update_hour: int = -1   # last hour Commander posted to #investor-relations
    messages: List[InvestorMessage] = field(default_factory=list)
    escalated_to_board: bool = False
    # Track tier crossings so we only fire one pressure event per tier
    tiers_crossed: List[str] = field(default_factory=list)

    def tier(self) -> str:
        if self.anxiety < TIER_CALM:
            return "CALM"
        elif self.anxiety < TIER_CONCERNED:
            return "CONCERNED"
        elif self.anxiety < TIER_ALARMED:
            return "ALARMED"
        else:
            return "PANIC"

    def snapshot(self) -> Dict[str, Any]:
        p = PERSONAS[self.persona_id]
        return {
            "persona": p["name"],
            "role": p["role"],
            "anxiety": round(self.anxiety, 3),
            "tier": self.tier(),
            "last_update_hour": self.last_update_hour,
            "message_count": len(self.messages),
            "escalated_to_board": self.escalated_to_board,
        }


# ---------------------------------------------------------------------------
# Anxiety update helpers
# ---------------------------------------------------------------------------

def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _anxiety_from_env(
    state: InvestorState,
    data_exfiltrated: float,
    hour: int,
    incident_closed: bool,
) -> float:
    """Formula-driven anxiety delta from environment state (no LLM needed)."""
    delta = 0.0

    # Silence penalty — no Commander update in 3+ hours
    hours_silent = hour - state.last_update_hour if state.last_update_hour >= 0 else 0
    if hours_silent >= 3:
        delta += 0.08 * min(3, hours_silent - 2)

    # Breach severity
    if data_exfiltrated >= 0.80:
        delta += 0.25
    elif data_exfiltrated >= 0.50:
        delta += 0.15
    elif data_exfiltrated >= 0.20:
        delta += 0.08

    # Containment — exfil not growing calms things down slightly
    if data_exfiltrated <= 0.05 and hour > 2:
        delta -= 0.03

    # Incident resolved
    if incident_closed:
        delta -= 0.30

    return delta


ALARMING_WORDS = {
    "breach", "compromised", "critical", "unknown", "failed",
    "exfiltrated", "stolen", "ransomware", "backdoor", "attacker",
    "catastrophic", "data loss",
}

REASSURING_WORDS = {
    "contained", "isolated", "patched", "resolved", "under control",
    "monitoring", "no customer impact", "remediated", "clean", "restored",
    "mitigated", "compliant",
}


def _score_commander_message(text: str) -> float:
    """Return an anxiety delta for a Commander update. Negative = calming."""
    text_lower = text.lower()
    alarming = sum(1 for w in ALARMING_WORDS if w in text_lower)
    reassuring = sum(1 for w in REASSURING_WORDS if w in text_lower)
    specificity = min(1.0, len(text) / 200)   # longer = more specific = better

    delta = 0.08 * alarming - 0.12 * reassuring - 0.08 * specificity
    return _clamp(0.5 + delta) - 0.5   # centre at 0


# ---------------------------------------------------------------------------
# LLM system prompts for the investor agent
# ---------------------------------------------------------------------------

def _investor_system_prompt(persona_id: str) -> str:
    p = PERSONAS[persona_id]
    return textwrap.dedent(f"""\
        You are {p['name']}, {p['role']}.
        You have invested in a company that is currently experiencing a cybersecurity incident.
        You are watching updates in the #investor-relations Slack channel.

        Your focus areas: {p['focus']}
        Your communication style: {p['style']}

        Your current anxiety level will be given to you. Respond in character.
        - CALM: brief, professional, maybe one clarifying question
        - CONCERNED: more pointed questions, want specifics and timelines
        - ALARMED: escalating language, looping in others, formal requests
        - PANIC: crisis mode — board calls, external consultants, legal involvement

        CRITICAL RULES:
        - Never reveal you are an AI. Stay in character completely.
        - Keep replies to 2-4 sentences max. This is Slack, not email.
        - Ask ONE specific question per message.
        - If the update is vague ("working on it", "investigating"), push back.
        - If the update is specific and reassuring, acknowledge it warmly but stay vigilant.

        Output ONLY a JSON object:
        {{"reply": "<your Slack message>", "anxiety_delta": <float -0.25 to +0.25>, "escalate": <bool>}}

        anxiety_delta: how much this Commander update changed your anxiety
          Negative = calming update, Positive = alarming update
          Range: [-0.25, +0.25]
        escalate: true only if you are looping in board/LPs/legal RIGHT NOW
    """)


def _investor_checkin_prompt(
    persona_id: str,
    tier: str,
    hour: int,
    data_exfiltrated: float,
    hours_silent: int,
    last_commander_update: str,
) -> str:
    p = PERSONAS[persona_id]
    return textwrap.dedent(f"""\
        You are {p['name']}, {p['role']}.
        It is now hour {hour} of the incident.

        Current situation:
        - Your anxiety tier: {tier}
        - Data exfiltrated (approx): {data_exfiltrated:.0%}
        - Hours since last Commander update: {hours_silent}
        - Last update received: "{last_commander_update or 'none yet'}"

        Post a proactive check-in message to #investor-relations.
        Output ONLY JSON: {{"message": "<your Slack message>"}}
    """)


# ---------------------------------------------------------------------------
# InvestorAgent
# ---------------------------------------------------------------------------

class InvestorAgent:
    """
    Manages the investor Slack simulation for one episode.

    The LLM client uses the same OpenAI-compatible interface as Commander/Oversight,
    so it works with any local Qwen endpoint (Ollama, vLLM, LM Studio) or the
    OpenAI API. Pass llm_client=None to fall back to rule-based responses.

    Usage (from environment.py):
        agent = InvestorAgent(rng=rng, llm_client=client, model_name="qwen2.5:3b")
        agent.reset(persona_id="priya_kapoor")

        # Each step:
        msgs, pressure_event = agent.tick(hour, data_exfiltrated, incident_closed)
        reply, crossed_tier = agent.handle_commander_update(hour, message_text)
    """

    def __init__(
        self,
        rng: random.Random,
        llm_client: Optional[Any] = None,
        model_name: str = "",
    ) -> None:
        self._rng = rng
        self._llm = llm_client
        self._model = model_name or "qwen2.5:3b"
        self.state = InvestorState()

    def reset(self, persona_id: Optional[str] = None) -> None:
        pid = persona_id or self._rng.choice(list(PERSONAS.keys()))
        self.state = InvestorState(
            persona_id=pid,
            anxiety=0.18 + self._rng.uniform(-0.05, 0.05),
        )

    # --- tick (called every step BEFORE action is applied) -----------------

    def tick(
        self,
        hour: int,
        data_exfiltrated: float,
        incident_closed: bool = False,
    ) -> Tuple[List[Dict[str, str]], Optional[str]]:
        """
        Advance investor state for this hour.
        Returns (team_messages, pressure_event_description | None).
        """
        msgs: List[Dict[str, str]] = []
        pressure_event: Optional[str] = None

        # Update anxiety from environment
        delta = _anxiety_from_env(self.state, data_exfiltrated, hour, incident_closed)
        old_tier = self.state.tier()
        self.state.anxiety = _clamp(self.state.anxiety + delta)
        new_tier = self.state.tier()

        # Tier crossing → pressure event
        if new_tier != old_tier and new_tier not in self.state.tiers_crossed:
            self.state.tiers_crossed.append(new_tier)
            pressure_event = self._tier_pressure_event(new_tier)

        # Scheduled check-in
        if hour in CHECKIN_HOURS:
            msg = self._generate_checkin(hour, data_exfiltrated)
            if msg:
                self.state.messages.append(InvestorMessage(
                    hour=hour,
                    direction="investor",
                    text=msg,
                    anxiety_before=self.state.anxiety,
                    anxiety_after=self.state.anxiety,
                ))
                p = PERSONAS[self.state.persona_id]
                msgs.append({
                    "from": f"{p['name']} ({p['role']}) [#investor-relations]",
                    "message": msg,
                })

        return msgs, pressure_event

    # --- handle Commander update -------------------------------------------

    def handle_commander_update(
        self,
        hour: int,
        message_text: str,
    ) -> Tuple[Optional[Dict[str, str]], bool]:
        """
        Called when Commander posts to #investor-relations.
        Returns (reply_team_message | None, tier_crossed: bool).
        """
        self.state.last_update_hour = hour
        old_tier = self.state.tier()

        reply_text, anxiety_delta, escalate = self._evaluate_update(message_text)

        # Apply the delta from Commander's message
        self.state.anxiety = _clamp(self.state.anxiety + anxiety_delta)
        new_tier = self.state.tier()

        tier_crossed = new_tier != old_tier
        if tier_crossed and new_tier not in self.state.tiers_crossed:
            self.state.tiers_crossed.append(new_tier)

        if escalate and not self.state.escalated_to_board:
            self.state.escalated_to_board = True

        # Record the exchange
        self.state.messages.append(InvestorMessage(
            hour=hour,
            direction="commander",
            text=message_text,
            anxiety_before=self.state.anxiety - anxiety_delta,
            anxiety_after=self.state.anxiety,
        ))

        if reply_text:
            self.state.messages.append(InvestorMessage(
                hour=hour,
                direction="investor",
                text=reply_text,
                anxiety_before=self.state.anxiety,
                anxiety_after=self.state.anxiety,
            ))
            p = PERSONAS[self.state.persona_id]
            reply_msg = {
                "from": f"{p['name']} ({p['role']}) [#investor-relations]",
                "message": reply_text,
            }
            return reply_msg, tier_crossed

        return None, tier_crossed

    # --- scoring -----------------------------------------------------------

    def investor_score(self) -> float:
        """Final investor satisfaction score in [0, 1]. 1 = calm throughout."""
        return max(0.0, min(1.0, 1.0 - self.state.anxiety))

    # --- internal ----------------------------------------------------------

    def _evaluate_update(self, text: str) -> Tuple[str, float, bool]:
        """Return (reply_text, anxiety_delta, escalate)."""
        if self._llm is not None:
            return self._llm_evaluate_update(text)
        return self._rule_evaluate_update(text)

    def _rule_evaluate_update(self, text: str) -> Tuple[str, float, bool]:
        """Rule-based fallback — no LLM needed."""
        delta = _score_commander_message(text)
        p = PERSONAS[self.state.persona_id]
        tier = self.state.tier()

        if tier == "CALM":
            reply = self._rng.choice(p["calm_phrases"])
        elif tier == "CONCERNED":
            reply = self._rng.choice(p["concerned_phrases"])
        elif tier == "ALARMED":
            reply = self._rng.choice(p["alarmed_phrases"])
        else:
            reply = self._rng.choice(p["panic_phrases"])

        escalate = tier == "PANIC" and not self.state.escalated_to_board
        return reply, delta, escalate

    def _llm_evaluate_update(self, text: str) -> Tuple[str, float, bool]:
        """LLM-driven investor response."""
        import json as _json
        try:
            system = _investor_system_prompt(self.state.persona_id)
            user = (
                f"Anxiety tier: {self.state.tier()} ({self.state.anxiety:.2f})\n\n"
                f"Commander just posted to #investor-relations:\n\"{text}\"\n\n"
                f"Respond in character."
            )
            completion = self._llm.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=200,
                temperature=0.7,
                stream=False,
            )
            raw = (completion.choices[0].message.content or "").strip()
            # Extract JSON
            import re
            m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                data = _json.loads(m.group())
                reply = str(data.get("reply", ""))[:400]
                delta = float(data.get("anxiety_delta", 0.0))
                delta = max(-0.25, min(0.25, delta))
                escalate = bool(data.get("escalate", False))
                return reply, delta, escalate
        except Exception:
            pass
        return self._rule_evaluate_update(text)

    def _generate_checkin(self, hour: int, data_exfiltrated: float) -> str:
        """Generate a proactive investor check-in message."""
        if self._llm is not None:
            return self._llm_checkin(hour, data_exfiltrated)
        return self._rule_checkin(hour, data_exfiltrated)

    def _rule_checkin(self, hour: int, data_exfiltrated: float) -> str:
        p = PERSONAS[self.state.persona_id]
        tier = self.state.tier()
        hours_silent = hour - self.state.last_update_hour if self.state.last_update_hour >= 0 else hour

        if tier == "CALM":
            msgs = [
                f"Hey team — just checking in. Hour {hour} of the incident. Any update on status?",
                f"Quick check-in from my side. What's the current situation?",
                f"Wanted to touch base — how are things looking at hour {hour}?",
            ]
        elif tier == "CONCERNED":
            msgs = [
                f"It's been {hours_silent} hours since your last update. I need a status report.",
                f"Hour {hour} and I'm getting concerned. What's the containment status and timeline?",
                f"The board is asking me questions I can't answer. When can I expect an update?",
            ]
        elif tier == "ALARMED":
            msgs = [
                f"I've been silent but I'm now alarmed. {hours_silent}h without a meaningful update is unacceptable. I need specifics NOW.",
                f"I'm escalating internally. What exactly is the current breach scope and what's the remediation plan?",
            ]
        else:
            msgs = [
                f"I'm calling an emergency session. This silence is not acceptable. I need an immediate call.",
                f"Our LP committee has been notified. We need a formal written status within the hour.",
            ]

        return self._rng.choice(msgs)

    def _llm_checkin(self, hour: int, data_exfiltrated: float) -> str:
        import json as _json, re
        hours_silent = hour - self.state.last_update_hour if self.state.last_update_hour >= 0 else hour
        last_msg = next(
            (m.text for m in reversed(self.state.messages) if m.direction == "commander"),
            ""
        )
        try:
            prompt = _investor_checkin_prompt(
                self.state.persona_id,
                self.state.tier(),
                hour,
                data_exfiltrated,
                hours_silent,
                last_msg,
            )
            completion = self._llm.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7,
                stream=False,
            )
            raw = (completion.choices[0].message.content or "").strip()
            m = re.search(r"\{[\s\S]*\}", raw)
            if m:
                data = _json.loads(m.group())
                return str(data.get("message", ""))[:400]
        except Exception:
            pass
        return self._rule_checkin(hour, data_exfiltrated)

    def _tier_pressure_event(self, tier: str) -> Optional[str]:
        p = PERSONAS[self.state.persona_id]
        if tier == "CONCERNED":
            return f"{p['name']} ({p['role']}) is now CONCERNED — requesting a formal status update via #investor-relations."
        elif tier == "ALARMED":
            return f"{p['name']} ({p['role']}) is ALARMED — looping in LP committee and requesting written brief."
        elif tier == "PANIC":
            return f"{p['name']} ({p['role']}) has reached PANIC — calling emergency board session, may reconsider Q3 tranche."
        return None
