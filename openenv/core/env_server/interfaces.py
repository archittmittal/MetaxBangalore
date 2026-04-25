"""OpenEnv interfaces -- minimal shim for HF Space deployment."""

import inspect
from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

from .types import Action, Observation, State

ActT = TypeVar("ActT", bound=Action)
ObsT = TypeVar("ObsT", bound=Observation)
StateT = TypeVar("StateT", bound=State)


class Environment(ABC, Generic[ActT, ObsT, StateT]):
    """Base class for all environment servers."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(self, transform=None, rubric=None):
        self.transform = transform
        self.rubric = rubric

    @abstractmethod
    def reset(self, seed=None, episode_id=None, **kwargs):
        pass

    @abstractmethod
    def step(self, action, timeout_s=None, **kwargs):
        pass

    @property
    @abstractmethod
    def state(self):
        pass

    def close(self):
        pass
