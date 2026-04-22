"""
ConflictEnv -- Actor System
===========================
8 actor archetypes with priority weights, flexibility levels,
tone sensitivity, and counter-proposal generation.

Actors are the "multi-agent" dimension of ConflictEnv (Theme 1).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
#  Enums
# ---------------------------------------------------------------------------

class Flexibility(Enum):
    VERY_LOW = 0.1   # Doctor -- almost never moves
    LOW = 0.3        # Boss, Client -- hard to move
    MEDIUM = 0.5     # Spouse, School
    HIGH = 0.8       # Vendor, Friend
    API = 0.0        # Airline -- governed by API rules, not negotiation


class ToneSensitivity(Enum):
    LOW = 0.2        # Boss, Doctor -- tolerates blunt
    MEDIUM = 0.5     # Client, School, Friend
    HIGH = 0.9       # Spouse -- needs warmth and care
    NONE = 0.0       # Airline -- automated system


# ---------------------------------------------------------------------------
#  Actor dataclass
# ---------------------------------------------------------------------------

@dataclass
class Actor:
    """A person or service the agent must negotiate with."""

    actor_id: str
    name: str
    role: str                     # boss, spouse, client, doctor, school, vendor, friend, airline
    emoji: str
    priority_weight: float        # 0.0-1.0  (boss=0.95, friend=0.50)
    flexibility: Flexibility
    tone_sensitivity: ToneSensitivity
    preferred_times: List[str] = field(default_factory=list)
    satisfaction: float = 1.0     # starts at 1.0, degrades with bad actions
    counter_proposals_made: int = 0

    def to_profile_dict(self) -> Dict:
        """Serialize for the observation (schema V1 format)."""
        return {
            "actor_id": self.actor_id,
            "name": self.name,
            "role": self.role,
            "emoji": self.emoji,
            "priority_weight": self.priority_weight,
            "flexibility": self.flexibility.name.lower(),
            "tone_sensitivity": self.tone_sensitivity.name.lower(),
            "preferred_times": self.preferred_times,
            "satisfaction": round(self.satisfaction, 3),
        }


# ---------------------------------------------------------------------------
#  Default actor catalogue
# ---------------------------------------------------------------------------

def _default_actors() -> Dict[str, Actor]:
    """Create the canonical set of 8 actor archetypes."""
    return {
        "boss": Actor(
            actor_id="boss", name="Rajesh Mehta", role="boss", emoji="[BOSS]",
            priority_weight=0.95, flexibility=Flexibility.LOW,
            tone_sensitivity=ToneSensitivity.LOW,
            preferred_times=["09:00", "10:00", "14:00"],
        ),
        "spouse": Actor(
            actor_id="spouse", name="Ananya", role="spouse", emoji="[SPOUSE]",
            priority_weight=0.90, flexibility=Flexibility.MEDIUM,
            tone_sensitivity=ToneSensitivity.HIGH,
            preferred_times=["12:00", "18:00", "19:00", "20:00"],
        ),
        "client": Actor(
            actor_id="client", name="Sarah Chen", role="client", emoji="[CLIENT]",
            priority_weight=0.85, flexibility=Flexibility.LOW,
            tone_sensitivity=ToneSensitivity.MEDIUM,
            preferred_times=["10:00", "11:00", "15:00"],
        ),
        "doctor": Actor(
            actor_id="doctor", name="Dr. Kapoor", role="doctor", emoji="[DOCTOR]",
            priority_weight=0.80, flexibility=Flexibility.VERY_LOW,
            tone_sensitivity=ToneSensitivity.LOW,
            preferred_times=["09:30", "14:30"],
        ),
        "school": Actor(
            actor_id="school", name="Delhi Public School", role="school", emoji="[SCHOOL]",
            priority_weight=0.75, flexibility=Flexibility.MEDIUM,
            tone_sensitivity=ToneSensitivity.MEDIUM,
            preferred_times=["08:00", "15:00", "16:00"],
        ),
        "vendor": Actor(
            actor_id="vendor", name="Priya Logistics", role="vendor", emoji="[VENDOR]",
            priority_weight=0.60, flexibility=Flexibility.HIGH,
            tone_sensitivity=ToneSensitivity.LOW,
            preferred_times=["10:00", "11:00", "14:00", "15:00", "16:00"],
        ),
        "friend": Actor(
            actor_id="friend", name="Arjun", role="friend", emoji="[FRIEND]",
            priority_weight=0.50, flexibility=Flexibility.HIGH,
            tone_sensitivity=ToneSensitivity.MEDIUM,
            preferred_times=["17:00", "18:00", "19:00", "20:00", "21:00"],
        ),
        "airline": Actor(
            actor_id="airline", name="IndiGo Airlines", role="airline", emoji="[AIRLINE]",
            priority_weight=0.40, flexibility=Flexibility.API,
            tone_sensitivity=ToneSensitivity.NONE,
            preferred_times=[],  # API-governed
        ),
    }


ALL_ACTORS = _default_actors()


def get_actors_for_difficulty(difficulty: str) -> Dict[str, Actor]:
    """
    Return a fresh copy of actors appropriate for the difficulty tier.

    Easy:   3 actors (boss, spouse, doctor)
    Medium: 5 actors (+ client, vendor)
    Hard:   7+ actors (+ school, friend, airline)
    """
    all_actors = _default_actors()

    if difficulty == "easy":
        keys = ["boss", "spouse", "doctor"]
    elif difficulty == "medium":
        keys = ["boss", "spouse", "client", "doctor", "vendor"]
    else:  # hard
        keys = list(all_actors.keys())  # all 8

    return {k: all_actors[k] for k in keys}


# ---------------------------------------------------------------------------
#  Satisfaction computation
# ---------------------------------------------------------------------------

def compute_satisfaction_delta(
    actor: Actor,
    old_slot: Optional[str],
    new_slot: Optional[str],
    message_tone: Optional[str] = None,
    event_cancelled: bool = False,
) -> float:
    """
    Compute how much an actor's satisfaction changes from a single action.

    Returns a delta in [-1.0, +0.3] range.  Negative = actor unhappy.
    """
    delta = 0.0

    # --- Cancel penalty (proportional to priority) ---
    if event_cancelled:
        delta -= 0.4 * actor.priority_weight
        return round(delta, 4)

    # --- Time displacement penalty ---
    if old_slot and new_slot and old_slot != new_slot:
        displacement = _time_distance(old_slot, new_slot)
        flex_factor = 1.0 - actor.flexibility.value  # less flexible = more penalty
        delta -= displacement * flex_factor * 0.15

    # --- Tone mismatch penalty ---
    if message_tone:
        tone_score = _tone_score(message_tone)
        mismatch = max(0, actor.tone_sensitivity.value - tone_score)
        delta -= mismatch * 0.3

    # --- Preference match bonus ---
    if new_slot and new_slot in actor.preferred_times:
        delta += 0.15

    return round(max(-1.0, min(0.3, delta)), 4)


def apply_satisfaction_delta(actor: Actor, delta: float) -> None:
    """Apply a satisfaction delta, clamping to [0.0, 1.0]."""
    actor.satisfaction = round(max(0.0, min(1.0, actor.satisfaction + delta)), 4)


# ---------------------------------------------------------------------------
#  Counter-proposal generation  (Multi-Agent Theme)
# ---------------------------------------------------------------------------

def generate_counter_proposal(
    actor: Actor,
    proposed_slot: str,
    available_slots: List[str],
    rng: Optional[random.Random] = None,
) -> Optional[Dict]:
    """
    An actor may reject a proposed time and counter-propose alternatives.

    Returns None if they accept, or a dict with alternatives if they reject.
    Probability of rejection = 1 - flexibility_value.
    """
    _rng = rng or random.Random()

    # API-governed actors don't negotiate
    if actor.flexibility == Flexibility.API:
        return None

    rejection_prob = 1.0 - actor.flexibility.value
    if _rng.random() > rejection_prob:
        return None  # Actor accepts

    # Pick 1-2 alternatives from preferred times that are also available
    alternatives = [t for t in actor.preferred_times if t in available_slots and t != proposed_slot]

    if not alternatives:
        # Fall back to random available slots
        alternatives = [s for s in available_slots if s != proposed_slot]

    if not alternatives:
        return None  # No alternatives -- forced acceptance

    chosen = _rng.sample(alternatives, min(2, len(alternatives)))
    actor.counter_proposals_made += 1

    return {
        "actor_id": actor.actor_id,
        "actor_name": actor.name,
        "rejected_slot": proposed_slot,
        "alternatives": chosen,
        "message": (
            f"{actor.emoji} {actor.name}: \"{proposed_slot} doesn't work for me. "
            f"How about {' or '.join(chosen)}?\""
        ),
    }


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _time_distance(slot_a: str, slot_b: str) -> float:
    """
    Compute normalized distance between two time slots (HH:MM strings).
    Returns 0.0 (same) to 1.0 (~12 hours apart).
    """
    try:
        ha, ma = map(int, slot_a.split(":"))
        hb, mb = map(int, slot_b.split(":"))
        diff_minutes = abs((ha * 60 + ma) - (hb * 60 + mb))
        return min(diff_minutes / 720.0, 1.0)  # 720 min = 12 hours
    except (ValueError, AttributeError):
        return 0.5  # Unknown -- moderate penalty


def _tone_score(tone: str) -> float:
    """Map a tone label to a warmth score [0, 1]."""
    tones = {
        "warm": 0.9,
        "friendly": 0.8,
        "professional": 0.6,
        "neutral": 0.5,
        "direct": 0.4,
        "blunt": 0.2,
        "cold": 0.1,
    }
    return tones.get(tone.lower(), 0.5)
