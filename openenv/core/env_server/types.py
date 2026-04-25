"""OpenEnv base types -- minimal shim for HF Space deployment."""

from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class Action(BaseModel):
    """Base class for all environment actions."""
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the action"
    )


class Observation(BaseModel):
    """Base class for all environment observations."""
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
    done: bool = Field(default=False, description="Whether the episode has terminated")
    reward: float = Field(default=0.0, description="Reward signal from the last action")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the observation"
    )


class State(BaseModel):
    """Base class for environment state."""
    model_config = ConfigDict(
        extra="allow",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )
    episode_id: Optional[str] = Field(default=None, description="Unique identifier for the current episode")
    step_count: int = Field(default=0, ge=0, description="Number of steps taken in the current episode")
