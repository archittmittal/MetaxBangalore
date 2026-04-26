"""
Calendar Bridge -- Real Calendar → ConflictEnv Scenario
========================================================
Converts ICS files or Google Calendar JSON exports into valid
ConflictEnv Scenario objects for real-world conflict resolution.

Usage:
    from calendar_bridge import calendar_to_scenario
    scenario = calendar_to_scenario(uploaded_file)
    env.reset_with_scenario(scenario)
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from conflict_env.scenarios import Scenario, ALL_SLOTS
from conflict_env.actors import Actor


# ---------------------------------------------------------------------------
#  Actor assignment heuristics
# ---------------------------------------------------------------------------

ROLE_KEYWORDS = {
    "boss": ["standup", "sprint", "1:1", "sync", "board", "review", "all-hands", "manager", "lead"],
    "client": ["client", "demo", "pitch", "sales", "presentation", "prospect", "partner"],
    "spouse": ["school", "pickup", "drop-off", "dropoff", "anniversary", "dinner", "family", "kid"],
    "doctor": ["doctor", "dentist", "clinic", "hospital", "therapy", "checkup", "appointment"],
    "friend": ["lunch", "drinks", "coffee", "gym", "yoga", "brunch", "hangout", "catch up"],
}

DEFAULT_ACTORS = {
    "boss": Actor(
        actor_id="boss", name="Rajesh Mehta", role="boss", emoji="[BOSS]",
        priority_weight=0.95, flexibility="low", tone_sensitivity="low",
        preferred_times=["09:00", "10:00", "14:00"], satisfaction=1.0,
    ),
    "client": Actor(
        actor_id="client", name="Client Team", role="client", emoji="[CLIENT]",
        priority_weight=0.85, flexibility="medium", tone_sensitivity="medium",
        preferred_times=["10:00", "14:00", "15:00"], satisfaction=1.0,
    ),
    "spouse": Actor(
        actor_id="spouse", name="Ananya", role="spouse", emoji="[SPOUSE]",
        priority_weight=0.90, flexibility="medium", tone_sensitivity="high",
        preferred_times=["12:00", "18:00", "19:00"], satisfaction=1.0,
    ),
    "doctor": Actor(
        actor_id="doctor", name="Dr. Kapoor", role="doctor", emoji="[DOCTOR]",
        priority_weight=0.80, flexibility="very_low", tone_sensitivity="low",
        preferred_times=["09:30", "14:30"], satisfaction=1.0,
    ),
    "friend": Actor(
        actor_id="friend", name="Arjun", role="friend", emoji="[FRIEND]",
        priority_weight=0.60, flexibility="high", tone_sensitivity="medium",
        preferred_times=["12:00", "18:00", "20:00"], satisfaction=1.0,
    ),
    "colleague": Actor(
        actor_id="colleague", name="Priya", role="colleague", emoji="[COLLEAGUE]",
        priority_weight=0.70, flexibility="medium", tone_sensitivity="medium",
        preferred_times=["09:00", "11:00", "15:00"], satisfaction=1.0,
    ),
}


# ---------------------------------------------------------------------------
#  ICS file parser
# ---------------------------------------------------------------------------

def parse_ics_file(file_path: str) -> List[Dict[str, Any]]:
    """Parse an .ics file into a list of event dicts."""
    try:
        from icalendar import Calendar
    except ImportError:
        raise ImportError("Install icalendar: pip install icalendar")

    with open(file_path, "r", encoding="utf-8") as f:
        cal = Calendar.from_ical(f.read())

    events = []
    for component in cal.walk():
        if component.name != "VEVENT":
            continue

        summary = str(component.get("summary", "Untitled"))
        dtstart = component.get("dtstart")
        dtend = component.get("dtend")

        if dtstart is None:
            continue

        start_dt = dtstart.dt if hasattr(dtstart, "dt") else dtstart
        end_dt = dtend.dt if (dtend and hasattr(dtend, "dt")) else start_dt + timedelta(hours=1)

        # Normalize to datetime if date-only
        if not isinstance(start_dt, datetime):
            start_dt = datetime.combine(start_dt, datetime.min.time().replace(hour=9))
        if not isinstance(end_dt, datetime):
            end_dt = datetime.combine(end_dt, datetime.min.time().replace(hour=10))

        attendees = component.get("attendee", [])
        if isinstance(attendees, str):
            attendees = [attendees]
        attendee_names = [str(a).replace("mailto:", "") for a in attendees]

        events.append({
            "title": summary,
            "start": start_dt,
            "end": end_dt,
            "attendees": attendee_names,
            "location": str(component.get("location", "")),
            "description": str(component.get("description", "")),
        })

    # Sort by start time
    events.sort(key=lambda e: e["start"])
    return events


# ---------------------------------------------------------------------------
#  JSON parser (Google Calendar export format)
# ---------------------------------------------------------------------------

def parse_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Parse a Google Calendar JSON export into event dicts."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_events = data if isinstance(data, list) else data.get("items", data.get("events", []))
    events = []

    for item in raw_events:
        title = item.get("summary", item.get("title", "Untitled"))

        # Handle Google Calendar dateTime format
        start_raw = item.get("start", {})
        end_raw = item.get("end", {})

        if isinstance(start_raw, dict):
            start_str = start_raw.get("dateTime", start_raw.get("date", ""))
            end_str = end_raw.get("dateTime", end_raw.get("date", ""))
        elif isinstance(start_raw, str):
            start_str = start_raw
            end_str = end_raw if isinstance(end_raw, str) else start_raw
        else:
            continue

        try:
            # Try ISO format first, then common formats
            for fmt in ["%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"]:
                try:
                    start_dt = datetime.strptime(start_str[:19], fmt[:len(fmt)-2] if "%z" in fmt else fmt)
                    break
                except ValueError:
                    continue
            else:
                start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00")).replace(tzinfo=None)

            for fmt in ["%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"]:
                try:
                    end_dt = datetime.strptime(end_str[:19], fmt[:len(fmt)-2] if "%z" in fmt else fmt)
                    break
                except ValueError:
                    continue
            else:
                end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            continue

        attendees_raw = item.get("attendees", [])
        attendee_names = [a.get("email", a.get("displayName", "")) for a in attendees_raw] if attendees_raw else []

        events.append({
            "title": title,
            "start": start_dt,
            "end": end_dt,
            "attendees": attendee_names,
            "location": item.get("location", ""),
            "description": item.get("description", ""),
        })

    events.sort(key=lambda e: e["start"])
    return events


# ---------------------------------------------------------------------------
#  Conflict detection
# ---------------------------------------------------------------------------

def detect_conflicts(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Find all pairs of overlapping events."""
    conflicts = []
    for i, a in enumerate(events):
        for j, b in enumerate(events):
            if j <= i:
                continue
            # Two events overlap if one starts before the other ends
            if a["start"] < b["end"] and b["start"] < a["end"]:
                conflicts.append({
                    "conflict_id": f"c{len(conflicts) + 1}",
                    "type": "overlap",
                    "event_ids": [a["event_id"], b["event_id"]],
                    "description": f"'{a['title']}' ({a['start'].strftime('%H:%M')}) overlaps with '{b['title']}' ({b['start'].strftime('%H:%M')})",
                    "resolved": False,
                })
    return conflicts


# ---------------------------------------------------------------------------
#  Actor assignment
# ---------------------------------------------------------------------------

def _guess_role(title: str, description: str = "") -> str:
    """Heuristically assign an actor role based on event title/description."""
    text = (title + " " + description).lower()
    for role, keywords in ROLE_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return role
    return "colleague"


def assign_actors(events: List[Dict[str, Any]]) -> Dict[str, Actor]:
    """Build the actor roster from events, assigning roles heuristically."""
    used_roles = set()
    for ev in events:
        role = _guess_role(ev["title"], ev.get("description", ""))
        ev["_role"] = role
        used_roles.add(role)

    return {role: DEFAULT_ACTORS[role] for role in used_roles if role in DEFAULT_ACTORS}


# ---------------------------------------------------------------------------
#  Scenario builder
# ---------------------------------------------------------------------------

def _is_hard_deadline(title: str) -> bool:
    """Heuristic: events with these keywords are hard deadlines."""
    hard_keywords = ["flight", "board", "demo", "interview", "exam", "surgery", "court", "deadline"]
    return any(kw in title.lower() for kw in hard_keywords)


def build_scenario(
    events: List[Dict[str, Any]],
    scenario_name: str = "my_real_calendar",
) -> Scenario:
    """Convert parsed events into a valid ConflictEnv Scenario."""

    # Assign event IDs and actor roles
    env_events = []
    hard_deadlines = []

    for idx, ev in enumerate(events):
        event_id = f"evt_{idx}"
        ev["event_id"] = event_id  # Needed for conflict detection

        role = _guess_role(ev["title"], ev.get("description", ""))
        is_hard = _is_hard_deadline(ev["title"])

        start_str = ev["start"].strftime("%Y-%m-%d %H:%M")
        end_str = ev["end"].strftime("%Y-%m-%d %H:%M")

        env_events.append({
            "event_id": event_id,
            "title": ev["title"],
            "start_time": start_str,
            "end_time": end_str,
            "actor_ids": [role],
            "priority": "high" if is_hard else "medium",
            "is_hard_deadline": is_hard,
            "is_unmovable": False,
            "status": "active",
            "reschedule_count": 0,
        })

        if is_hard:
            hard_deadlines.append(event_id)

    # Detect conflicts
    conflicts = detect_conflicts(events)

    # Assign actors
    actors = assign_actors(events)

    # Default policy rules
    policy_rules = {
        "free_cancel": True,
        "rebooking_fee": 0,
        "cancel_window_hrs": 24,
        "max_reschedules_per_event": 2,
    }

    # Max steps scales with complexity
    max_steps = max(15, len(conflicts) * 5 + 10)

    narrative = f"Real calendar imported with {len(env_events)} events and {len(conflicts)} conflicts."

    return Scenario(
        name=scenario_name,
        difficulty="custom",
        events=env_events,
        conflicts=conflicts,
        actors=actors,
        policy_rules=policy_rules,
        hard_deadlines=hard_deadlines,
        max_steps=max_steps,
        narrative=narrative,
    )


# ---------------------------------------------------------------------------
#  End-to-end pipeline
# ---------------------------------------------------------------------------

def calendar_to_scenario(file_path: str) -> Scenario:
    """
    Full pipeline: file -> parsed events -> conflicts -> scenario.
    Supports .ics and .json files.
    """
    if file_path.endswith(".ics"):
        events = parse_ics_file(file_path)
    elif file_path.endswith(".json"):
        events = parse_json_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}. Use .ics or .json")

    if not events:
        raise ValueError("No events found in the uploaded file.")

    return build_scenario(events)


def format_events_summary(events: List[Dict[str, Any]], conflicts: List[Dict[str, Any]]) -> str:
    """Format a human-readable summary of detected events and conflicts."""
    lines = [f"Imported {len(events)} events:\n"]
    for ev in events:
        start = ev["start"].strftime("%H:%M") if isinstance(ev["start"], datetime) else str(ev.get("start_time", ""))
        role = ev.get("_role", _guess_role(ev["title"]))
        lines.append(f"  {start} - {ev['title']} ({role})")

    if conflicts:
        lines.append(f"\n{len(conflicts)} Conflicts Detected:")
        for c in conflicts:
            lines.append(f"  {c['conflict_id']}: {c['description']}")
    else:
        lines.append("\nNo conflicts detected -- your schedule is clean!")

    return "\n".join(lines)
