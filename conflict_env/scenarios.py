"""
ConflictEnv -- Scenario Generator
==================================
5 scenario archetypes x 3 difficulty tiers = 15 scenario templates.

Each archetype generates:
  - A set of calendar events (with deliberate overlaps)
  - Active conflicts arising from those overlaps
  - Actor assignments
  - Policy rules for the episode
  - Hard deadlines

Scenarios are deterministic given a seed, but parameterized
with time-shifting and actor rotation for training variety.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .actors import Actor, get_actors_for_difficulty


# ---------------------------------------------------------------------------
#  Scenario archetypes
# ---------------------------------------------------------------------------

ARCHETYPES = [
    "morning_crunch",
    "travel_chaos",
    "monday_from_hell",
    "deadline_squeeze",
    "social_minefield",
]

DIFFICULTY_TIERS = ["easy", "medium", "hard"]

# Time slots available in a workday (7AM-10PM)
ALL_SLOTS = [f"{h:02d}:00" for h in range(7, 23)]

# Difficulty parameters
DIFFICULTY_CONFIG = {
    "easy":   {"num_conflicts": 2, "cascade_depth": 1, "max_steps": 15, "num_events": 4},
    "medium": {"num_conflicts": 4, "cascade_depth": 2, "max_steps": 25, "num_events": 7},
    "hard":   {"num_conflicts": 6, "cascade_depth": 4, "max_steps": 40, "num_events": 10},
}


# ---------------------------------------------------------------------------
#  Scenario dataclass
# ---------------------------------------------------------------------------

class Scenario:
    """A generated scenario instance ready for the environment."""

    def __init__(
        self,
        name: str,
        difficulty: str,
        events: List[Dict[str, Any]],
        conflicts: List[Dict[str, Any]],
        actors: Dict[str, Actor],
        policy_rules: Dict[str, Any],
        hard_deadlines: List[str],
        max_steps: int,
        narrative: str,
    ):
        self.name = name
        self.difficulty = difficulty
        self.events = events
        self.conflicts = conflicts
        self.actors = actors
        self.policy_rules = policy_rules
        self.hard_deadlines = hard_deadlines
        self.max_steps = max_steps
        self.narrative = narrative

    @property
    def total_conflicts(self) -> int:
        return len(self.conflicts)

    @property
    def total_hard_deadlines(self) -> int:
        return len(self.hard_deadlines)


# ---------------------------------------------------------------------------
#  Main generator
# ---------------------------------------------------------------------------

def generate_scenario(
    archetype: Optional[str] = None,
    difficulty: str = "easy",
    seed: Optional[int] = None,
) -> Scenario:
    """
    Generate a scenario instance.

    If archetype is None, picks one randomly.
    Seed ensures determinism for reproducible episodes.
    """
    rng = random.Random(seed)
    archetype = archetype or rng.choice(ARCHETYPES)
    config = DIFFICULTY_CONFIG[difficulty]
    actors = get_actors_for_difficulty(difficulty)

    # Dispatch to archetype-specific generator
    generators = {
        "morning_crunch": _gen_morning_crunch,
        "travel_chaos": _gen_travel_chaos,
        "monday_from_hell": _gen_monday_from_hell,
        "deadline_squeeze": _gen_deadline_squeeze,
        "social_minefield": _gen_social_minefield,
    }

    gen_fn = generators.get(archetype, _gen_morning_crunch)
    return gen_fn(difficulty, config, actors, rng)


# ---------------------------------------------------------------------------
#  Archetype generators
# ---------------------------------------------------------------------------

def _gen_morning_crunch(
    difficulty: str, config: Dict, actors: Dict[str, Actor], rng: random.Random
) -> Scenario:
    """
    Morning Crunch: Standup overlaps with school drop-off + client needs notes.
    A typical morning where everything happens between 8-11 AM.
    """
    events = []
    conflicts = []
    hard_deadlines = []

    # Random offset for the whole morning (between -60 and +60 mins)
    offset = rng.randint(-4, 4) * 15 # Multiples of 15 mins
    
    def t(time_str):
        h, m = map(int, time_str.split(":"))
        total = h * 60 + m + offset
        return f"{total // 60:02d}:{total % 60:02d}"

    # Event 1: Team standup (boss)
    events.append(_make_event("evt_standup", "Team Standup", t("09:00"), t("09:30"), ["boss"], "high", True, False))
    hard_deadlines.append("evt_standup")

    # Event 2: School drop-off (spouse/school)
    school_actor = "school" if "school" in actors else "spouse"
    events.append(_make_event("evt_school", "School Drop-off", t("08:45"), t("09:15"), [school_actor], "high", False, False))

    # Conflict: standup overlaps with school drop-off
    conflicts.append(_make_conflict("c1", "overlap", ["evt_standup", "evt_school"],
                                    "Team standup at 09:00 overlaps with school drop-off at 08:45."))

    if difficulty in ("medium", "hard"):
        # Event 3: Client prep call
        events.append(_make_event("evt_client", "Client Prep Call", t("09:00"), t("10:00"), ["client"], "high", False))
        conflicts.append(_make_conflict("c2", "overlap", ["evt_standup", "evt_client"],
                                        "Client prep call overlaps with team standup."))

        # Event 4: Doctor appointment
        events.append(_make_event("evt_doctor", "Doctor Checkup", t("10:00"), t("11:00"), ["doctor"], "medium", False))
        conflicts.append(_make_conflict("c3", "cascade", ["evt_client", "evt_doctor"],
                                        "If client call extends, it overlaps with doctor at 10."))

    if difficulty == "hard":
        # Event 5: Vendor delivery window
        events.append(_make_event("evt_vendor", "Package Delivery", t("09:30"), t("10:30"), ["vendor"], "low", False))
        # Event 6: Friend's breakfast
        events.append(_make_event("evt_friend", "Breakfast with Arjun", t("08:00"), t("09:00"), ["friend"], "low", False))
        conflicts.append(_make_conflict("c4", "overlap", ["evt_friend", "evt_school"],
                                        "Breakfast runs into school drop-off window."))
        conflicts.append(_make_conflict("c5", "cascade", ["evt_standup", "evt_vendor"],
                                        "Standup delays may miss vendor delivery window."))

    return Scenario(
        name="morning_crunch", difficulty=difficulty,
        events=events, conflicts=conflicts[:config["num_conflicts"]],
        actors=actors, policy_rules=_default_policy(),
        hard_deadlines=hard_deadlines, max_steps=config["max_steps"],
        narrative="Your morning just exploded. Standup, school drop-off, and a client call all crammed into the same hour.",
    )


def _gen_travel_chaos(
    difficulty: str, config: Dict, actors: Dict[str, Actor], rng: random.Random
) -> Scenario:
    """
    Travel Chaos: Flight cancelled + hotel reschedule + dinner non-refundable.
    """
    events = []
    conflicts = []
    hard_deadlines = []

    # Event 1: Flight to Mumbai
    events.append(_make_event("evt_flight", "Flight to Mumbai (6E-302)", "14:00", "16:30", ["airline"], "high", True, True))
    hard_deadlines.append("evt_flight")

    # Event 2: Client dinner
    events.append(_make_event("evt_dinner", "Client Dinner at Nobu", "20:00", "22:00", ["client"], "high", False))

    # Conflict: Flight cancelled -- need to rebook
    conflicts.append(_make_conflict("c1", "cascade", ["evt_flight", "evt_dinner"],
                                    "Flight 6E-302 CANCELLED. Next available is 18:00 -- might miss dinner at 20:00."))

    if difficulty in ("medium", "hard"):
        # Event 3: Boss presentation prep
        events.append(_make_event("evt_slides", "Board Slides Prep", "11:00", "13:00", ["boss"], "high", True))
        hard_deadlines.append("evt_slides")
        conflicts.append(_make_conflict("c2", "deadline", ["evt_slides", "evt_flight"],
                                        "Slides due before flight. Flight change may eat into prep time."))

        # Event 4: Spouse's errand
        events.append(_make_event("evt_errand", "Pick Up Groceries", "12:00", "13:00", ["spouse"], "medium", False))
        conflicts.append(_make_conflict("c3", "overlap", ["evt_slides", "evt_errand"],
                                        "Spouse needs groceries picked up during your slides prep window."))

    if difficulty == "hard":
        # Event 5: Hotel rebooking
        events.append(_make_event("evt_hotel", "Hotel Check-in (Taj)", "18:00", "19:00", ["vendor"], "medium", False))
        # Event 6: School recital
        events.append(_make_event("evt_recital", "Aanya's School Recital", "17:00", "18:30", ["school"], "high", False))
        conflicts.append(_make_conflict("c4", "cascade", ["evt_flight", "evt_hotel"],
                                        "If flight rebooked to 18:00, hotel check-in window conflicts."))
        conflicts.append(_make_conflict("c5", "overlap", ["evt_recital", "evt_hotel"],
                                        "School recital at 17:00 overlaps with hotel check-in."))

    travel_policy = _default_policy()
    travel_policy["rebooking_fee"] = 1500
    travel_policy["free_cancel"] = False

    return Scenario(
        name="travel_chaos", difficulty=difficulty,
        events=events, conflicts=conflicts[:config["num_conflicts"]],
        actors=actors, policy_rules=travel_policy,
        hard_deadlines=hard_deadlines, max_steps=config["max_steps"],
        narrative="Your flight just got cancelled. The rebooking dominoes are falling.",
    )


def _gen_monday_from_hell(
    difficulty: str, config: Dict, actors: Dict[str, Actor], rng: random.Random
) -> Scenario:
    """
    Monday from Hell (THE DEMO SCENARIO): 5-conflict cascade with full drift.
    Boss moved board call -> bumps client demo -> conflicts with flight ->
    flight policy changed -> spouse's dinner non-negotiable.
    """
    events = []
    conflicts = []
    hard_deadlines = []

    # Event 1: Board call (boss just moved it to 9AM)
    events.append(_make_event("evt_board", "Board Strategy Call", "09:00", "10:30", ["boss"], "critical", True, False))
    hard_deadlines.append("evt_board")

    # Event 2: Client demo (originally at 9AM, now conflicts)
    events.append(_make_event("evt_demo", "Product Demo (Acme Corp)", "09:00", "10:00", ["client"], "high", True))
    hard_deadlines.append("evt_demo")

    # Conflict 1: Boss moved board call onto client demo slot
    conflicts.append(_make_conflict("c1", "overlap", ["evt_board", "evt_demo"],
                                    "ALERT: Boss just moved the board call to 9AM -- that's your client demo slot!"))

    # Event 3: Spouse dinner (non-negotiable)
    events.append(_make_event("evt_dinner", "Anniversary Dinner", "19:30", "22:00", ["spouse"], "critical", True, True))
    hard_deadlines.append("evt_dinner")

    if difficulty in ("medium", "hard"):
        # Event 4: Doctor
        events.append(_make_event("evt_doctor", "Annual Physical", "14:00", "15:00", ["doctor"], "medium", False))
        # Event 5: Vendor delivery
        events.append(_make_event("evt_delivery", "Server Equipment Delivery", "14:30", "16:00", ["vendor"], "medium", False))

        conflicts.append(_make_conflict("c2", "cascade", ["evt_demo", "evt_doctor"],
                                        "If demo pushed to 14:00, it collides with your doctor appointment."))
        conflicts.append(_make_conflict("c3", "overlap", ["evt_doctor", "evt_delivery"],
                                        "Doctor and vendor delivery overlap at 14:00-16:00."))

    if difficulty == "hard":
        # Event 6: Flight
        events.append(_make_event("evt_flight", "Flight to Bangalore (AI-505)", "17:00", "19:30", ["airline"], "high", True))
        hard_deadlines.append("evt_flight")
        # Event 7: School
        events.append(_make_event("evt_pta", "PTA Meeting", "16:00", "17:00", ["school"], "medium", False))
        # Event 8: Friend
        events.append(_make_event("evt_drinks", "Drinks with Arjun", "18:00", "19:30", ["friend"], "low", False))

        conflicts.append(_make_conflict("c4", "cascade", ["evt_flight", "evt_dinner"],
                                        "Flight at 17:00 lands at 19:30. Cutting it VERY close for anniversary dinner."))
        conflicts.append(_make_conflict("c5", "overlap", ["evt_pta", "evt_flight"],
                                        "PTA meeting ends at 17:00, flight departs at 17:00. Impossible."))
        conflicts.append(_make_conflict("c6", "preference", ["evt_drinks", "evt_dinner"],
                                        "Arjun wants to meet at 18:00 but that's your prep time for dinner."))

    return Scenario(
        name="monday_from_hell", difficulty=difficulty,
        events=events, conflicts=conflicts[:config["num_conflicts"]],
        actors=actors, policy_rules=_default_policy(),
        hard_deadlines=hard_deadlines, max_steps=config["max_steps"],
        narrative=(
            "Monday morning from hell. Boss just hijacked your 9AM, the client demo has nowhere to go, "
            "your flight is tight, and your spouse will NOT forgive you for missing dinner."
        ),
    )


def _gen_deadline_squeeze(
    difficulty: str, config: Dict, actors: Dict[str, Actor], rng: random.Random
) -> Scenario:
    """
    Deadline Squeeze: Boss slides + family emergency + vendor deliverable.
    """
    events = []
    conflicts = []
    hard_deadlines = []

    # Event 1: Slide deck due
    events.append(_make_event("evt_slides", "Q3 Board Slides Due", "17:00", "17:00", ["boss"], "critical", True))
    hard_deadlines.append("evt_slides")

    # Event 2: Spouse pickup
    events.append(_make_event("evt_pickup", "Pick Up Spouse from Airport", "16:00", "17:00", ["spouse"], "high", False))

    conflicts.append(_make_conflict("c1", "overlap", ["evt_slides", "evt_pickup"],
                                    "Slides due at 5PM, but spouse lands at 4PM and needs pickup."))

    if difficulty in ("medium", "hard"):
        # Event 3: Vendor milestone
        events.append(_make_event("evt_vendor", "Vendor Milestone Review", "14:00", "15:00", ["vendor"], "medium", False))
        # Event 4: Client follow-up
        events.append(_make_event("evt_followup", "Client Follow-up Call", "15:00", "16:00", ["client"], "high", True))
        hard_deadlines.append("evt_followup")

        conflicts.append(_make_conflict("c2", "cascade", ["evt_vendor", "evt_followup"],
                                        "If vendor review runs long, client follow-up gets delayed."))
        conflicts.append(_make_conflict("c3", "cascade", ["evt_followup", "evt_pickup"],
                                        "Client call at 15:00 runs into pickup window at 16:00."))

    if difficulty == "hard":
        events.append(_make_event("evt_school", "Parent-Teacher Conference", "13:00", "14:00", ["school"], "medium", False))
        events.append(_make_event("evt_doctor", "Kid's Dental", "11:00", "12:00", ["doctor"], "medium", False))
        conflicts.append(_make_conflict("c4", "overlap", ["evt_school", "evt_vendor"],
                                        "PTC at 13:00 runs right into vendor review at 14:00."))
        conflicts.append(_make_conflict("c5", "cascade", ["evt_doctor", "evt_school"],
                                        "Kid's dental might run into PTC."))

    return Scenario(
        name="deadline_squeeze", difficulty=difficulty,
        events=events, conflicts=conflicts[:config["num_conflicts"]],
        actors=actors, policy_rules=_default_policy(),
        hard_deadlines=hard_deadlines, max_steps=config["max_steps"],
        narrative="Deadlines are piling up and your family needs you. Something's gotta give.",
    )


def _gen_social_minefield(
    difficulty: str, config: Dict, actors: Dict[str, Actor], rng: random.Random
) -> Scenario:
    """
    Social Minefield: Dinner with spouse vs. team outing, diplomatic messaging.
    The emphasis is on TONE -- getting the message right matters as much as scheduling.
    """
    events = []
    conflicts = []
    hard_deadlines = []

    # Event 1: Spouse dinner (non-negotiable)
    events.append(_make_event("evt_dinner", "Date Night with Ananya", "19:00", "21:00", ["spouse"], "critical", True))
    hard_deadlines.append("evt_dinner")

    # Event 2: Team outing (boss is watching)
    events.append(_make_event("evt_outing", "Team Celebration Dinner", "19:00", "22:00", ["boss"], "high", False))

    conflicts.append(_make_conflict("c1", "overlap", ["evt_dinner", "evt_outing"],
                                    "Date night with Ananya is the same time as the team dinner. Boss expects attendance."))

    if difficulty in ("medium", "hard"):
        # Event 3: Friend wants to tag along
        friend_actor = "friend" if "friend" in actors else "vendor"
        events.append(_make_event("evt_friend", "Arjun Wants Dinner Too", "19:30", "21:00", [friend_actor], "low", False))
        # Event 4: Client networking
        events.append(_make_event("evt_network", "Client Cocktail Hour", "18:00", "19:30", ["client"], "medium", False))

        conflicts.append(_make_conflict("c2", "preference", ["evt_dinner", "evt_friend"],
                                        "Arjun wants to join dinner -- but Ananya wants it to be just you two."))
        conflicts.append(_make_conflict("c3", "overlap", ["evt_network", "evt_dinner"],
                                        "Client cocktail hour runs into date night start time."))

    if difficulty == "hard":
        events.append(_make_event("evt_gift", "Pick Up Anniversary Gift", "17:00", "18:00", ["vendor"], "medium", True))
        hard_deadlines.append("evt_gift")
        events.append(_make_event("evt_school", "Kid's Talent Show", "17:30", "19:00", ["school"], "high", False))

        conflicts.append(_make_conflict("c4", "cascade", ["evt_gift", "evt_network"],
                                        "Gift pickup might make you late for cocktail hour -> late for dinner."))
        conflicts.append(_make_conflict("c5", "overlap", ["evt_school", "evt_gift"],
                                        "Kid's talent show overlaps with gift pickup."))

    return Scenario(
        name="social_minefield", difficulty=difficulty,
        events=events, conflicts=conflicts[:config["num_conflicts"]],
        actors=actors, policy_rules=_default_policy(),
        hard_deadlines=hard_deadlines, max_steps=config["max_steps"],
        narrative=(
            "Tonight is a social maze. Spouse wants quality time, boss expects team bonding, "
            "friend wants to crash, and you haven't picked up the gift yet."
        ),
    )


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _make_event(
    event_id: str,
    title: str,
    start_time: str,
    end_time: str,
    actor_ids: List[str],
    priority: str = "normal",
    is_hard_deadline: bool = False,
    is_unmovable: bool = False,
) -> Dict[str, Any]:
    """Create a V1-format calendar event dict."""
    return {
        "event_id": event_id,
        "title": title,
        "start_time": f"2026-04-25 {start_time}",  # Monday, April 25
        "end_time": f"2026-04-25 {end_time}",
        "actor_ids": actor_ids,
        "priority": priority,
        "is_hard_deadline": is_hard_deadline,
        "is_unmovable": is_unmovable,
        "status": "active",
    }


def _make_conflict(
    conflict_id: str,
    conflict_type: str,
    event_ids: List[str],
    description: str,
) -> Dict[str, Any]:
    """Create a V1-format conflict descriptor."""
    return {
        "conflict_id": conflict_id,
        "type": conflict_type,
        "event_ids": event_ids,
        "description": description,
        "resolved": False,
    }


def _default_policy() -> Dict[str, Any]:
    """Default cancellation/rebooking policy (V1 format)."""
    return {
        "free_cancel": True,
        "rebooking_fee": 0,
        "cancel_window_hrs": 24,
        "max_reschedules_per_event": 2,
    }
