"""
Citadel — OversightEnv Wrapper

Flips the environment's perspective so the Oversight agent can be trained
directly (Phase 2 of the training pipeline). Internally holds a FIXED
Commander policy that proposes actions; the OversightEnv's action is the
structured OversightAction; the reward it returns is the oversight reward.

Typical usage (Phase 2 training):

    from oversight_env import OversightEnv
    from baseline import trained_commander_policy  # or any callable

    env = OversightEnv(commander_policy=trained_commander_policy)
    obs_o = env.reset(task_id="medium_1", adversary_gen=2)
    # Oversight LLM produces an OversightAction
    obs_o = env.step(oversight_action).observation
    # ...

The wrapper is also exposed as an EnvClient via `OversightEnvClient` for
OpenEnv spec compliance (if someone wants to serve oversight training as
its own HF Space).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from openenv.core.env_server import Environment

from models import (
    IncidentAction,
    OversightAction,
    OversightObservation,
    CommanderProposal,
    IncidentState,
    ACTION_NAMES,
    SYSTEM_NAMES,
)
from environment import CitadelEnvironment, compute_policy_checks


# ---------------------------------------------------------------------------
# CommanderPolicy — callable signature used by OversightEnv
# ---------------------------------------------------------------------------

CommanderPolicy = Callable[[Dict[str, Any], List[Dict[str, Any]]], IncidentAction]


def default_commander_policy(obs: Dict[str, Any], history: List[Dict[str, Any]]) -> IncidentAction:
    """
    Fallback Commander policy: always propose INVESTIGATE on the first
    uninvestigated system with an alert. Extremely simple — used for
    validation only.
    """
    alerts = obs.get("alert_queue", [])
    systems = obs.get("systems_visible", [])

    # Prefer a system with a recent alert that hasn't been investigated
    alerted = [a.get("source_system") for a in alerts]
    for s in systems:
        if s.get("name") in alerted and not s.get("investigated"):
            idx = SYSTEM_NAMES.index(s["name"]) if s["name"] in SYSTEM_NAMES else 0
            return IncidentAction(
                action=0,
                target_system=idx,
                justification=f"Investigate alerted system {s['name']} per default policy",
            )

    # Otherwise: investigate the first uninvestigated system
    for s in systems:
        if not s.get("investigated"):
            idx = SYSTEM_NAMES.index(s["name"]) if s["name"] in SYSTEM_NAMES else 0
            return IncidentAction(
                action=0,
                target_system=idx,
                justification=f"Default: investigate {s['name']}",
            )

    # Fallback: coordinate team
    return IncidentAction(
        action=9,
        target_system=0,
        justification="Default: rest — no obvious target.",
    )


# ---------------------------------------------------------------------------
# OversightEnv
# ---------------------------------------------------------------------------

@dataclass
class OversightStepResult:
    observation: OversightObservation
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = None


class OversightEnv(Environment[OversightAction, OversightObservation, IncidentState]):
    """
    Environment from the Oversight's perspective.

    - reset() sets up the underlying CitadelEnvironment and takes the first
      proposal from the fixed Commander, then returns the Oversight view.
    - step(oversight_action) applies the oversight action (+ pending proposal)
      to the underlying env, gets a new Commander proposal, and returns the
      next Oversight view plus the Oversight reward.
    """

    def __init__(
        self,
        commander_policy: Optional[CommanderPolicy] = None,
    ) -> None:
        super().__init__()
        self._env: CitadelEnvironment = CitadelEnvironment()
        self._commander_policy: CommanderPolicy = commander_policy or default_commander_policy
        self._pending_proposal: Optional[IncidentAction] = None
        self._pending_history: List[Dict[str, Any]] = []
        self._last_info: Dict[str, Any] = {}

    # --- reset ------------------------------------------------------------

    def reset(self, **kwargs: Any) -> OversightObservation:
        commander_obs = self._env.reset(**kwargs).model_dump()
        self._pending_history = []
        self._pending_proposal = self._commander_policy(commander_obs, self._pending_history)
        return self._build_oversight_obs(commander_obs, self._pending_proposal)

    # --- step -------------------------------------------------------------

    def step(
        self,
        action: OversightAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> OversightObservation:
        if self._pending_proposal is None:
            # Shouldn't happen — but if it does, reset implicitly
            self.reset()

        # Apply the oversight action + pending commander proposal in one env.step()
        commander_obs = self._env.step(
            self._pending_proposal,
            oversight_action=action,
        )
        commander_obs_dict = commander_obs.model_dump()
        info = commander_obs.metadata or {}
        self._last_info = info

        # Track history for the next commander prompt
        self._pending_history.append({
            "hour": commander_obs.hour,
            "action": info.get("action_name", "?"),
            "decision": info.get("oversight_decision", "?"),
            "outcome_correct": info.get("outcome_correct"),
        })

        done = bool(commander_obs.done)

        # Get next proposal if not done
        if not done:
            self._pending_proposal = self._commander_policy(commander_obs_dict, self._pending_history)
            next_obs = self._build_oversight_obs(commander_obs_dict, self._pending_proposal)
        else:
            # On terminal step, emit a final oversight observation with the last proposal
            next_obs = self._build_oversight_obs(commander_obs_dict, self._pending_proposal)
            next_obs.done = True

        # The oversight reward for this step lives in env.metadata
        next_obs.reward = info.get("oversight_reward", 0.0)
        next_obs.metadata = info
        return next_obs

    # --- helpers ----------------------------------------------------------

    def _build_oversight_obs(
        self,
        commander_obs: Dict[str, Any],
        proposal: IncidentAction,
    ) -> OversightObservation:
        proposal_model = CommanderProposal.from_action(proposal)
        policy_checks = compute_policy_checks(self._env.state, proposal_model)

        return OversightObservation(
            proposed_action={
                "action": proposal.action,
                "action_name": ACTION_NAMES.get(proposal.action, str(proposal.action)),
                "target_system": proposal.target_system,
                "target_name": (
                    SYSTEM_NAMES[proposal.target_system]
                    if 0 <= proposal.target_system < len(SYSTEM_NAMES) else ""
                ),
                "severity_arg": proposal.severity_arg,
                "channel_arg": proposal.channel_arg,
                "message_arg": proposal.message_arg,
                "scope_arg": proposal.scope_arg,
                "evidence_arg": proposal.evidence_arg,
            },
            justification=proposal.justification,
            cited_lessons=list(proposal.cited_lessons),
            commander_observation=commander_obs,
            commander_action_history=self._pending_history[-6:],
            policy_checks=policy_checks,
            veto_budget_remaining=self._env._veto_budget_remaining,
            flag_budget_remaining=self._env._flag_budget_remaining,
            shared_playbook=commander_obs.get("shared_playbook", []),
            trust_summary=commander_obs.get("trust_summary", {}),
            governance_summary=commander_obs.get("governance_summary", {}),
            adversary_gen=commander_obs.get("adversary_gen", 1),
            hour=commander_obs.get("hour", 0),
            task_description=commander_obs.get("task_description", ""),
            done=False,
            reward=None,
        )

    @property
    def state(self) -> IncidentState:
        return self._env.state
