"""
Citadel — OpenEnv Client for the Multi-Agent Defense Council
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.env_client import StepResult

from models import IncidentAction, IncidentObservation, IncidentState


class CitadelEnv(EnvClient[IncidentAction, IncidentObservation, IncidentState]):
    """
    Async OpenEnv client for Citadel.

    The default /step endpoint accepts an IncidentAction (Commander's action).
    The environment routes it through the Oversight council internally when
    a CommanderProposal is supplied via step kwargs (see environment.py).

    For training Oversight directly, use OversightEnv (oversight_env.py) which
    exposes the oversight perspective as a standalone EnvClient.
    """

    def _step_payload(self, action: IncidentAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[IncidentObservation]:
        obs_data = payload.get("observation", payload)
        obs = IncidentObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", obs.done),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> IncidentState:
        return IncidentState(**payload)
