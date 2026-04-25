"""
Stub for openenv.core.env_server — replaces the Meta OpenEnv SDK dependency.
Action, Observation, State are Pydantic BaseModels so all subclasses
that use Field() and model_dump() work correctly.
"""
from __future__ import annotations
from typing import Generic, TypeVar
from pydantic import BaseModel, ConfigDict

A = TypeVar("A")
O = TypeVar("O")
S = TypeVar("S")


class Action(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Observation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class State(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class Environment(Generic[A, O, S]):
    """Minimal base class — matches the interface CitadelEnvironment expects."""

    def reset(self, **kwargs):
        raise NotImplementedError

    def step(self, action: A) -> O:
        raise NotImplementedError

    def render(self):
        pass
