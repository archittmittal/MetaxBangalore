"""
ConflictEnv -- Pydantic Action / Observation Schemas
====================================================
Type-safe contract between the LLM agent and the environment.
Extends OpenEnv base classes for protocol compliance.
"""

from typing import Dict, Any, List, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


# ---------------------------------------------------------------------------
#  Valid commands the agent can issue
# ---------------------------------------------------------------------------
VALID_COMMANDS = {
    "reschedule",       # Move an event to a new time slot
    "draft_message",    # Send a message to an actor (tone matters!)
    "cancel",           # Cancel an event entirely
    "query_preference", # Ask an actor about their preferred times
    "escalate",         # Escalate a conflict to a human (failure state)
    "confirm",          # Lock in a resolved event
    "resolve",          # Declare the episode complete
}


class ConflictAction(Action):
    """An action taken by the LLM agent to resolve scheduling conflicts."""

    command: str = Field(
        ...,
        description=(
            "The conflict-resolution command to execute. "
            "One of: reschedule, draft_message, cancel, query_preference, "
            "escalate, confirm, resolve."
        ),
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Command-specific parameters. Examples:\n"
            "  reschedule: {event_id: str, new_slot: str}\n"
            "  draft_message: {actor_id: str, tone: str, content: str}\n"
            "  cancel: {event_id: str}\n"
            "  query_preference: {actor_id: str}\n"
            "  escalate: {conflict_id: str}\n"
            "  confirm: {event_id: str}\n"
            "  resolve: {}"
        ),
    )


class ConflictObservation(Observation):
    """The observation returned to the agent after each step."""

    # --- Calendar state (schema varies by drift version!) ---
    calendar: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current calendar state. Structure changes with schema drift.",
    )

    # --- Conflict tracking ---
    active_conflicts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of unresolved scheduling conflicts.",
    )

    # --- Actor information ---
    actor_profiles: Dict[str, Any] = Field(
        default_factory=dict,
        description="Known actor preferences and metadata.",
    )

    # --- Policy rules (also subject to drift) ---
    policy_rules: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current cancellation/rebooking/venue policies.",
    )

    # --- Communication ---
    pending_messages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Messages drafted but not yet sent.",
    )

    # --- Environment metadata ---
    schema_version: str = Field(
        "v1",
        description="Current schema version hint (v1, v2, or v3).",
    )
    step_count: int = Field(0, description="Current step number in the episode.")
    max_steps: int = Field(15, description="Maximum steps allowed this episode.")
    difficulty: str = Field("easy", description="Current difficulty tier.")
    scenario_name: str = Field("", description="Name of the active scenario archetype.")

    # --- Reward & termination ---
    reward: float = Field(0.0, description="Reward delta for the latest step.")
    done: bool = Field(False, description="Whether the episode is complete.")
    feedback: str = Field(
        "",
        description="Natural-language feedback on the last action taken.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional technical metadata for the observation.",
    )


class ConflictState(State):
    """The internal state of the environment, used for protocol introspection."""

    obs: ConflictObservation = Field(..., description="The latest observation.")
    schema_v: str = Field("v1", description="Current drift version.")
    difficulty: str = Field("easy", description="Current difficulty.")

