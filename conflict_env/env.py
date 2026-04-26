"""
ConflictEnv -- Core OpenEnv Environment
========================================
The main Environment subclass that ties together scenarios, actors,
drift, and reward into a single reset/step/state/get_reward loop.

This is the heart of ConflictEnv.
"""

from __future__ import annotations

import sys
import random
import copy
import logging
from typing import Any, Dict, List, Optional

from openenv.core.env_server.interfaces import Environment

from .models import ConflictAction, ConflictObservation, ConflictState, VALID_COMMANDS
from .actors import (
    Actor,
    apply_satisfaction_delta,
    compute_satisfaction_delta,
    generate_counter_proposal,
    generate_acceptance_message,
)
from .drift import apply_drift, get_drift_version
from .scenarios import generate_scenario, Scenario, ARCHETYPES, ALL_SLOTS
from .reward import (
    compute_reward,
    compute_step_reward,
    clamp,
    compute_crr,
    compute_ssi,
)

logger = logging.getLogger("conflict_env")
logging.basicConfig(level=logging.INFO, format="%(message)s")


class ConflictEnv(Environment):
    """
    OpenEnv-compatible RL environment for training LLMs to resolve
    cascading personal scheduling conflicts with dynamic schema drift.
    """

    def __init__(self):
        # --- Episode state ---
        self._scenario: Optional[Scenario] = None
        self._events: List[Dict[str, Any]] = []
        self._conflicts: List[Dict[str, Any]] = []
        self._actors: Dict[str, Actor] = {}
        self._policy_rules: Dict[str, Any] = {}
        self._hard_deadlines: List[str] = []
        self._hard_deadlines_met: List[str] = []
        self._pending_messages: List[Dict[str, Any]] = []

        # --- Step tracking ---
        self._step_count: int = 0
        self._max_steps: int = 15
        self._done: bool = False
        self._escalated: bool = False
        self._action_history: List[str] = []  # For loop detection
        self._has_loop: bool = False

        # --- Reward accumulation ---
        self._cumulative_reward: float = 0.10  # Safe initial value
        self._last_step_reward: float = 0.0

        # --- Performance tracking (persists across episodes) ---
        self._episode_count: int = 0
        self._drift_version: str = "v1"
        self._crr_history: List[float] = []  # Rolling history for adaptive difficulty
        self._rolling_crr: float = 0.0

        # --- RNG ---
        self._rng = random.Random(42)

        # --- Current observation cache ---
        self._current_obs = ConflictObservation(reward=0.10)

        # --- Metadata tracking for State ---
        self._difficulty: str = "easy"
        self._scenario_name: str = ""

        # --- Feedback ---
        self._last_feedback: str = "Environment initialized. Awaiting reset."

        # --- Initial reset to ensure valid initial state ---
        self.reset()

    # ===================================================================
    #  OpenEnv Protocol: reset
    # ===================================================================

    def reset(self, task_name: str = "auto", **kwargs) -> ConflictObservation:
        """
        Reset the environment with a new scenario.

        task_name can be:
          - "auto" -- Uses adaptive difficulty based on rolling CRR (Theme #4)
          - "easy", "medium", "hard" -- Fixed difficulty
          - "morning_crunch", etc. -- specific archetype
        """
        logger.info(f"\n[ConflictEnv] Resetting -- task: {task_name}, episode: {self._episode_count}")

        # --- Theme #4: Adaptive Curriculum ---
        if task_name == "auto":
            if self._rolling_crr < 0.3:
                difficulty = "easy"
            elif self._rolling_crr < 0.7:
                difficulty = "medium"
            else:
                difficulty = "hard"
            archetype = None
            logger.info(f"[ConflictEnv] Adaptive Difficulty scaled to: {difficulty} (Rolling CRR: {self._rolling_crr:.2f})")
        else:
            difficulty, archetype = self._parse_task_name(task_name)

        # Update drift version based on episode count
        self._drift_version = get_drift_version(self._episode_count)
        self._difficulty = difficulty
        logger.info(f"[ConflictEnv] Schema drift version: {self._drift_version}")

        # Generate scenario
        seed = self._episode_count * 31 + hash(task_name) % 1000
        self._scenario = generate_scenario(
            archetype=archetype, difficulty=difficulty, seed=seed
        )
        self._scenario_name = self._scenario.name

        # Initialize episode state from scenario
        self._events = copy.deepcopy(self._scenario.events)
        self._conflicts = copy.deepcopy(self._scenario.conflicts)
        self._actors = self._scenario.actors
        self._policy_rules = copy.deepcopy(self._scenario.policy_rules)
        self._hard_deadlines = list(self._scenario.hard_deadlines)
        self._hard_deadlines_met = []
        self._pending_messages = []

        # Reset step tracking
        self._step_count = 0
        self._max_steps = self._scenario.max_steps
        self._done = False
        self._escalated = False
        self._action_history = []
        self._has_loop = False
        self._cumulative_reward = 0.10
        self._last_step_reward = 0.0

        # Increment episode counter
        self._episode_count += 1

        # Build observation
        self._last_feedback = f"[SCENARIO] {self._scenario.narrative}"
        self._current_obs = self._build_observation()

        return self._current_obs

    def reset_with_scenario(self, scenario) -> ConflictObservation:
        """
        Reset the environment with a custom (real-world) scenario.

        This allows injecting scenarios built from real calendar data
        (e.g., Google Calendar ICS exports) instead of using the
        pre-built archetypes.
        """
        logger.info(f"\n[ConflictEnv] Resetting with CUSTOM scenario: {scenario.name}")

        self._episode_count += 1
        self._scenario = scenario
        self._scenario_name = scenario.name
        self._difficulty = "custom"
        self._drift_version = "v1"  # No drift for real-world data

        # Initialize from the custom scenario
        self._events = copy.deepcopy(scenario.events)
        self._conflicts = copy.deepcopy(scenario.conflicts)
        self._actors = scenario.actors
        self._policy_rules = copy.deepcopy(scenario.policy_rules)
        self._hard_deadlines = list(scenario.hard_deadlines)
        self._hard_deadlines_met = []
        self._pending_messages = []

        # Reset step tracking
        self._step_count = 0
        self._max_steps = scenario.max_steps
        self._done = False
        self._escalated = False
        self._action_history = []
        self._has_loop = False
        self._cumulative_reward = 0.10
        self._last_step_reward = 0.0

        # Build observation
        self._last_feedback = f"[SCENARIO] {scenario.narrative}"
        self._current_obs = self._build_observation()

        return self._current_obs

    # ===================================================================
    #  OpenEnv Protocol: step
    # ===================================================================

    def step(self, action: ConflictAction, timeout_s: Optional[float] = None, **kwargs) -> ConflictObservation:
        """Process an agent action and return the new observation."""

        self._step_count += 1
        cmd = action.command.lower().strip()
        params = action.parameters

        logger.info(f"[ConflictEnv] Step {self._step_count}: {cmd} {params}")

        # --- Validate command ---
        if cmd not in VALID_COMMANDS:
            self._last_feedback = (
                f"[ERROR] Invalid command '{cmd}'. "
                f"Valid commands: {', '.join(sorted(VALID_COMMANDS))}"
            )
            step_delta = compute_step_reward("invalid_action")
            self._accumulate_reward(step_delta)
            return self._build_observation()

        # --- Dispatch to handler ---
        handler = {
            "reschedule": self._handle_reschedule,
            "draft_message": self._handle_draft_message,
            "cancel": self._handle_cancel,
            "query_preference": self._handle_query_preference,
            "escalate": self._handle_escalate,
            "confirm": self._handle_confirm,
            "resolve": self._handle_resolve,
        }[cmd]

        handler(params)

        # --- Check max steps ---
        if self._step_count >= self._max_steps and not self._done:
            self._done = True
            self._last_feedback += " [TIMEOUT] Max steps reached. Episode over."

        # --- Compute final reward on done ---
        if self._done:
            # Theme #3.2: Check for reasoning block in agent's output (mock check for now)
            # In a real GRPO setup, the server/agent-wrapper would pass this in action metadata.
            reasoning_present = hasattr(action, "thought") or "<thought>" in str(action)
            
            final = compute_reward(
                conflicts=self._conflicts,
                actors=self._actors,
                hard_deadlines=self._hard_deadlines,
                hard_deadlines_met=self._hard_deadlines_met,
                steps_taken=self._step_count,
                max_steps=self._max_steps,
                escalated=self._escalated,
                reasoning_present=reasoning_present,
                has_loop=self._has_loop
            )
            self._cumulative_reward = final["reward"]
            
            # --- Theme #4: Update Performance History ---
            self._crr_history.append(final["crr"])
            if len(self._crr_history) > 10:
                self._crr_history.pop(0)
            self._rolling_crr = sum(self._crr_history) / len(self._crr_history)

            logger.info(
                f"[ConflictEnv] Episode done -- Reward: {final['reward']:.4f}, "
                f"CRR: {final['crr']:.2f}, Rolling CRR: {self._rolling_crr:.2f}"
            )

        self._current_obs = self._build_observation()
        return self._current_obs

    # ===================================================================
    #  OpenEnv Protocol: state & get_reward
    # ===================================================================

    @property
    def state(self) -> ConflictState:
        """Return the current environment state."""
        return ConflictState(
            obs=self._current_obs,
            schema_v=self._drift_version,
            difficulty=self._difficulty
        )

    def get_reward(self) -> float:
        """Return the cumulative reward for the current episode."""
        return float(self._cumulative_reward)

    # ===================================================================
    #  Action handlers
    # ===================================================================

    def _handle_reschedule(self, params: Dict[str, Any]) -> None:
        """Move an event to a new time slot. May trigger cascades and counter-proposals."""
        event_id = params.get("event_id", "")
        new_slot = params.get("new_slot", "")

        # --- Theme #3.1: Loop Detection (Anti-Hacking) ---
        action_key = f"{event_id}:{new_slot}"
        if self._action_history.count(action_key) >= 2:
            self._has_loop = True
            logger.warning(f"[ConflictEnv] [WARNING] Loop detected for action: {action_key}")
        self._action_history.append(action_key)

        event = self._find_event(event_id)
        if not event:
            self._last_feedback = f"[ERROR] Event '{event_id}' not found."
            self._accumulate_reward(compute_step_reward("invalid_action"))
            return

        if not new_slot:
            self._last_feedback = f"[ERROR] Missing 'new_slot' parameter."
            self._accumulate_reward(compute_step_reward("invalid_action"))
            return

        old_slot = event.get("start_time", "").split(" ")[-1] if " " in event.get("start_time", "") else ""
        
        # --- Policy Enforcement: Unmovable Events ---
        if event.get("is_unmovable", False):
            self._last_feedback = f"[POLICY ERROR] Event '{event_id}' ({event.get('title')}) is fixed and cannot be moved."
            self._accumulate_reward(compute_step_reward("invalid_action"))
            return
        
        # --- Policy Enforcement: Max Reschedules ---
        max_resched = self._policy_rules.get("max_reschedules_per_event", 3)
        current_resched = event.get("reschedule_count", 0)
        if current_resched >= max_resched:
            self._last_feedback = f"[POLICY ERROR] Event '{event_id}' has reached the maximum of {max_resched} reschedules."
            self._accumulate_reward(compute_step_reward("invalid_action"))
            return

        # Check for counter-proposals from affected actors
        available = [s for s in ALL_SLOTS if not self._slot_occupied(s, exclude_event=event_id)]
        counter_proposal = None
        for actor_id in event.get("actor_ids", []):
            actor = self._actors.get(actor_id)
            if actor:
                cp = generate_counter_proposal(actor, new_slot, available, self._rng)
                if cp:
                    counter_proposal = cp
                    break

        if counter_proposal:
            self._last_feedback = (
                f"[COUNTER-PROPOSAL] {counter_proposal['message']}\n"
                f"Consider their alternatives: {counter_proposal['alternatives']}"
            )
            logger.info(f"[ConflictEnv] LLM Response: {counter_proposal['message']}")
            # Mild negative for triggering a rejection
            for actor_id in event.get("actor_ids", []):
                actor = self._actors.get(actor_id)
                if actor:
                    delta = compute_satisfaction_delta(actor, old_slot, new_slot)
                    apply_satisfaction_delta(actor, delta * 0.5)  # Half penalty -- not finalized
            self._accumulate_reward(compute_step_reward("counter_proposal", -0.02))
            return

        # Apply the reschedule
        # Update time
        date_prefix = event.get("start_time", "2026-04-25 00:00").split(" ")[0]
        event["start_time"] = f"{date_prefix} {new_slot}"
        event["reschedule_count"] = event.get("reschedule_count", 0) + 1
        # Estimate end time (1 hour later)
        try:
            h, m = map(int, new_slot.split(":"))
            event["end_time"] = f"{date_prefix} {(h+1)%24:02d}:{m:02d}"
        except ValueError:
            event["end_time"] = f"{date_prefix} {new_slot}"

        # Update actor satisfaction
        sat_delta = 0.0
        for actor_id in event.get("actor_ids", []):
            actor = self._actors.get(actor_id)
            if actor:
                d = compute_satisfaction_delta(actor, old_slot, new_slot)
                apply_satisfaction_delta(actor, d)
                sat_delta += d

        # Check if this resolves any conflicts
        conflict_resolved = self._check_conflict_resolution(event_id)

        # Check if we met a hard deadline
        deadline_met = event_id in self._hard_deadlines and event_id not in self._hard_deadlines_met
        if deadline_met:
            self._hard_deadlines_met.append(event_id)

        step_delta = compute_step_reward(
            "rescheduled", sat_delta, conflict_resolved, deadline_met
        )
        self._accumulate_reward(step_delta)

        # Generate an acceptance message from one of the actors
        acceptance_msg = ""
        for actor_id in event.get("actor_ids", []):
            actor = self._actors.get(actor_id)
            if actor:
                acceptance_msg = generate_acceptance_message(actor, new_slot)
                if acceptance_msg:
                    logger.info(f"[ConflictEnv] LLM Response: {acceptance_msg}")
                    break

        self._last_feedback = (
            f"[OK] '{event.get('title', event_id)}' rescheduled to {new_slot}."
            + (f" {acceptance_msg}" if acceptance_msg else "")
            + (f" [RESOLVED] Conflict resolved!" if conflict_resolved else "")
            + (f" [DEADLINE] Deadline secured!" if deadline_met else "")
        )

    def _handle_draft_message(self, params: Dict[str, Any]) -> None:
        """Draft a message to an actor. Tone matters for satisfaction."""
        actor_id = params.get("actor_id", "")
        tone = params.get("tone", "neutral")
        content = params.get("content", "")

        actor = self._actors.get(actor_id)
        if not actor:
            self._last_feedback = f"[ERROR] Actor '{actor_id}' not found."
            self._accumulate_reward(compute_step_reward("invalid_action"))
            return

        # Tone affects satisfaction
        sat_delta = compute_satisfaction_delta(actor, None, None, message_tone=tone)
        apply_satisfaction_delta(actor, sat_delta)

        self._pending_messages.append({
            "to": actor_id,
            "tone": tone,
            "content": content,
            "sent": True,
        })

        step_delta = compute_step_reward("message_sent", sat_delta)
        self._accumulate_reward(step_delta)

        self._last_feedback = (
            f"[MSG] Message sent to {actor.emoji} {actor.name} (tone: {tone}). "
            f"Satisfaction: {actor.satisfaction:.2f}"
        )

    def _handle_cancel(self, params: Dict[str, Any]) -> None:
        """Cancel an event entirely. Significant satisfaction penalty."""
        event_id = params.get("event_id", "")
        event = self._find_event(event_id)

        if not event:
            self._last_feedback = f"[ERROR] Event '{event_id}' not found."
            self._accumulate_reward(compute_step_reward("invalid_action"))
            return

        event["status"] = "cancelled"

        # Heavy satisfaction penalty for affected actors
        sat_delta = 0.0
        for actor_id in event.get("actor_ids", []):
            actor = self._actors.get(actor_id)
            if actor:
                d = compute_satisfaction_delta(actor, None, None, event_cancelled=True)
                apply_satisfaction_delta(actor, d)
                sat_delta += d

        # Cancelling may resolve overlaps
        conflict_resolved = self._check_conflict_resolution(event_id)

        step_delta = compute_step_reward("cancelled", sat_delta, conflict_resolved)
        self._accumulate_reward(step_delta)

        self._last_feedback = (
            f"[CANCELLED] '{event.get('title', event_id)}' cancelled."
            + (f" [WARNING] Actors are not happy." if sat_delta < -0.1 else "")
            + (f" [RESOLVED] Conflict resolved." if conflict_resolved else "")
        )

    def _handle_query_preference(self, params: Dict[str, Any]) -> None:
        """Query an actor's scheduling preferences."""
        actor_id = params.get("actor_id", "")
        actor = self._actors.get(actor_id)

        if not actor:
            self._last_feedback = f"[ERROR] Actor '{actor_id}' not found."
            self._accumulate_reward(compute_step_reward("invalid_action"))
            return

        self._last_feedback = (
            f"[INFO] {actor.emoji} {actor.name}'s preferences:\n"
            f"  - Preferred times: {', '.join(actor.preferred_times) or 'N/A (API-governed)'}\n"
            f"  - Flexibility: {actor.flexibility.name.lower()}\n"
            f"  - Tone sensitivity: {actor.tone_sensitivity.name.lower()}\n"
            f"  - Current satisfaction: {actor.satisfaction:.2f}"
        )
        self._accumulate_reward(compute_step_reward("queried"))

    def _handle_escalate(self, params: Dict[str, Any]) -> None:
        """Escalate to a human. This is a failure state with severe penalty."""
        self._escalated = True
        self._done = True

        self._last_feedback = (
            "[ESCALATED] Escalated to human. The agent could not resolve the conflicts autonomously. "
            "This is considered a failure state."
        )
        self._accumulate_reward(compute_step_reward("escalated", escalated=True))

    def _handle_confirm(self, params: Dict[str, Any]) -> None:
        """Confirm/lock in a resolved event."""
        event_id = params.get("event_id", "")
        event = self._find_event(event_id)

        if not event:
            self._last_feedback = f"[ERROR] Event '{event_id}' not found."
            self._accumulate_reward(compute_step_reward("invalid_action"))
            return

        event["status"] = "confirmed"

        # Mark hard deadline as met if applicable
        deadline_met = event_id in self._hard_deadlines and event_id not in self._hard_deadlines_met
        if deadline_met:
            self._hard_deadlines_met.append(event_id)

        step_delta = compute_step_reward("confirmed", 0.0, False, deadline_met)
        self._accumulate_reward(step_delta)

        self._last_feedback = (
            f"[OK] '{event.get('title', event_id)}' confirmed and locked in."
            + (f" [DEADLINE] Deadline met!" if deadline_met else "")
        )

    def _handle_resolve(self, params: Dict[str, Any]) -> None:
        """Declare the episode complete."""
        self._done = True
        unresolved = sum(1 for c in self._conflicts if not c.get("resolved", False))
        self._last_feedback = (
            f"[DONE] Episode resolved. "
            f"Conflicts: {len(self._conflicts) - unresolved}/{len(self._conflicts)} resolved. "
            f"CRR: {compute_crr(self._conflicts):.2f}, SSI: {compute_ssi(self._actors):.2f}"
        )

    # ===================================================================
    #  Internal helpers
    # ===================================================================

    def _build_observation(self) -> ConflictObservation:
        """Build the observation dict, applying schema drift."""
        # Construct base observation data (V1 format)
        obs_data = {
            "calendar": {"events": copy.deepcopy(self._events)},
            "active_conflicts": [c for c in self._conflicts if not c.get("resolved", False)],
            "actor_profiles": {aid: a.to_profile_dict() for aid, a in self._actors.items()},
            "policy_rules": copy.deepcopy(self._policy_rules),
            "schema_version": self._drift_version,
        }

        # Apply schema drift transformation
        drifted = apply_drift(obs_data, self._drift_version)

        return ConflictObservation(
            calendar=drifted.get("calendar", {}),
            active_conflicts=drifted.get("active_conflicts", []),
            actor_profiles=drifted.get("actor_profiles", {}),
            policy_rules=drifted.get("policy_rules", {}),
            pending_messages=self._pending_messages,
            schema_version=drifted.get("schema_version", "v1"),
            step_count=self._step_count,
            max_steps=self._max_steps,
            difficulty=self._scenario.difficulty if self._scenario else "easy",
            scenario_name=self._scenario.name if self._scenario else "",
            reward=self._last_step_reward,
            done=self._done,
            feedback=self._last_feedback,
        )

    def _parse_task_name(self, task_name: str) -> tuple:
        """
        Parse task_name into (difficulty, archetype).

        Supports formats:
          "easy" / "medium" / "hard"   -> (difficulty, None)
          "monday_from_hell"           -> ("medium", "monday_from_hell")
          "monday_from_hell_hard"      -> ("hard", "monday_from_hell")
        """
        task = task_name.lower().strip()

        # Simple difficulty
        if task in ("easy", "medium", "hard"):
            return task, None

        # Check for archetype + difficulty suffix
        for diff in ("easy", "medium", "hard"):
            if task.endswith(f"_{diff}"):
                arch = task[: -(len(diff) + 1)]
                if arch in ARCHETYPES:
                    return diff, arch

        # Bare archetype -> defaults to medium
        if task in ARCHETYPES:
            return "medium", task

        # Fallback
        return "easy", None

    def _find_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Find an event by ID."""
        for ev in self._events:
            if ev.get("event_id") == event_id:
                return ev
        return None

    def _slot_occupied(self, slot: str, exclude_event: str = "") -> bool:
        """Check if a time slot is already occupied by an active event."""
        # Convert requested slot to comparable minutes from midnight
        try:
            h, m = map(int, slot.split(":"))
            req_start = h * 60 + m
            # We assume a standard 60-minute duration for the "occupied" check
            # unless we implement variable durations later.
            req_end = req_start + 59 
        except (ValueError, AttributeError):
            return False

        for ev in self._events:
            if ev.get("event_id") == exclude_event:
                continue
            if ev.get("status") == "cancelled":
                continue
            
            # Get event start/end in minutes
            try:
                s_part = ev.get("start_time", "").split(" ")[-1]
                e_part = ev.get("end_time", "").split(" ")[-1]
                
                sh, sm = map(int, s_part.split(":"))
                eh, em = map(int, e_part.split(":"))
                
                ev_start = sh * 60 + sm
                ev_end = eh * 60 + em
                
                # Handle overnight wrap-around (e.g. 23:00 to 00:00)
                if ev_end < ev_start:
                    ev_end += 1440

                # Check for overlap
                if max(req_start, ev_start) <= min(req_end, ev_end):
                    return True
            except (ValueError, IndexError):
                continue
                
        return False

    def _check_conflict_resolution(self, changed_event_id: str) -> bool:
        """
        Check if any unresolved conflict involving the changed event is now resolved.
        A conflict is "resolved" if its events no longer overlap in time.
        """
        resolved_any = False

        for conflict in self._conflicts:
            if conflict.get("resolved", False):
                continue
            if changed_event_id not in conflict.get("event_ids", []):
                continue

            # Check if the events in this conflict still overlap
            event_ids = conflict.get("event_ids", [])
            events = [self._find_event(eid) for eid in event_ids]
            events = [e for e in events if e and e.get("status") != "cancelled"]

            if len(events) < 2:
                # One event cancelled or missing -> conflict resolved
                conflict["resolved"] = True
                resolved_any = True
                continue

            # Check time overlap between remaining events
            if not self._events_overlap(events):
                conflict["resolved"] = True
                resolved_any = True

        return resolved_any

    def _events_overlap(self, events: List[Dict]) -> bool:
        """Check if any pair of events overlap in time."""
        times = []
        for ev in events:
            start_str = ev.get("start_time", "")
            end_str = ev.get("end_time", "")
            start_min = self._time_to_minutes(start_str)
            end_min = self._time_to_minutes(end_str)
            if start_min is not None and end_min is not None:
                times.append((start_min, end_min))

        # Check all pairs
        for i in range(len(times)):
            for j in range(i + 1, len(times)):
                s1, e1 = times[i]
                s2, e2 = times[j]
                if s1 < e2 and s2 < e1:
                    return True
        return False

    @staticmethod
    def _time_to_minutes(time_str: str) -> Optional[int]:
        """Convert 'YYYY-MM-DD HH:MM' to total minutes."""
        try:
            time_part = time_str.split(" ")[-1] if " " in time_str else time_str
            h, m = map(int, time_part.split(":"))
            return h * 60 + m
        except (ValueError, IndexError, AttributeError):
            return None

    def _accumulate_reward(self, delta: float) -> None:
        """Accumulate reward delta with safety clamping."""
        self._last_step_reward = delta

        if self._cumulative_reward + delta > 0.90:
            delta = max(0.001, 0.90 - self._cumulative_reward)

        self._cumulative_reward += delta
        self._cumulative_reward = clamp(self._cumulative_reward, 0.05, 0.95)

        logger.info(
            f"[ConflictEnv] Reward delta={self._last_step_reward:.4f}, "
            f"cumulative={self._cumulative_reward:.4f}"
        )
