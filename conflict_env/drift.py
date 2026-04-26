"""
ConflictEnv -- Schema Drift Engine
===================================
The differentiating feature of ConflictEnv.

Implements versioned schema mutations that force the agent to generalize
rather than memorize. Targets the Patronus AI bonus prize (Consumer
Workflows with Schema Drift).

Drift versions:
  V1 (episodes 0-49)   -- Baseline, clean schemas
  V2 (episodes 50-99)  -- Mild drift: field renames, format changes
  V3 (episodes 100+)   -- Heavy drift: nested structures, new date formats
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
#  Version constants
# ---------------------------------------------------------------------------

DRIFT_V1 = "v1"
DRIFT_V2 = "v2"
DRIFT_V3 = "v3"


def get_drift_version(episode_number: int, drift_interval: int = 1) -> str:
    """
    Deterministic drift schedule.

    version = episode_number // drift_interval
    0 -> v1, 1 -> v2, 2+ -> v3
    """
    epoch = episode_number // drift_interval
    if epoch == 0:
        return DRIFT_V1
    elif epoch == 1:
        return DRIFT_V2
    else:
        return DRIFT_V3


# ---------------------------------------------------------------------------
#  Calendar schema transformers
# ---------------------------------------------------------------------------

def transform_calendar(events: List[Dict], version: str) -> List[Dict]:
    """
    Transform calendar events according to the schema version.

    V1: {event_id, title, start_time: "2026-04-25 09:00", end_time, actor_ids, priority, is_hard_deadline}
    V2: {event_id, title, startTime: "04/25/2026 9:00AM", endTime, actorIds, priority, isHardDeadline}
    V3: {event_id, title, schedule: {begin: {day,month,year,hour,minute}, end: ...}, participants, meta: {priority, hard_deadline}}
    """
    if version == DRIFT_V1:
        return events  # Already in V1 format

    transformed = []
    for ev in events:
        ev = copy.deepcopy(ev)
        if version == DRIFT_V2:
            transformed.append(_to_v2_event(ev))
        elif version == DRIFT_V3:
            transformed.append(_to_v3_event(ev))
        else:
            transformed.append(ev)

    return transformed


def transform_policy(policy: Dict, version: str) -> Dict:
    """
    Transform policy rules according to the schema version.

    V1: {"free_cancel": true, "rebooking_fee": 0, "cancel_window_hrs": 24}
    V2: {"cancellation_fee": 0, "rebooking_charge": 0, "cancelWindowHours": 24}
    V3: {"policy": {"cancel": {"fee_usd": 0, "window_hrs": 24}, "rebook": {"fee_usd": 0}}}
    """
    if version == DRIFT_V1:
        return policy

    policy = copy.deepcopy(policy)

    if version == DRIFT_V2:
        return _to_v2_policy(policy)
    elif version == DRIFT_V3:
        return _to_v3_policy(policy)

    return policy


def transform_actor_profiles(profiles: Dict[str, Dict], version: str) -> Dict[str, Dict]:
    """
    Transform actor profile keys according to the schema version.

    V1: actor.preferred_times  ->  V2: actor.availability.preferred  ->  V3: actor.scheduling_prefs.ideal_slots
    """
    if version == DRIFT_V1:
        return profiles

    result = {}
    for actor_id, profile in profiles.items():
        profile = copy.deepcopy(profile)
        if version == DRIFT_V2:
            result[actor_id] = _to_v2_actor(profile)
        elif version == DRIFT_V3:
            result[actor_id] = _to_v3_actor(profile)
        else:
            result[actor_id] = profile

    return result


def transform_conflicts(conflicts: List[Dict], version: str) -> List[Dict]:
    """
    Transform conflict descriptors according to the schema version.

    V1: {"conflict_id", "type": "overlap", "event_ids", "description"}
    V2: {"conflict_id", "conflict_kind": "OVERLAP", "eventIds", "details"}
    V3: {"conflict_id", "issue": {"category": "TIME_COLLISION"}, "involved_events", "summary"}
    """
    if version == DRIFT_V1:
        return conflicts

    transformed = []
    for c in conflicts:
        c = copy.deepcopy(c)
        if version == DRIFT_V2:
            transformed.append(_to_v2_conflict(c))
        elif version == DRIFT_V3:
            transformed.append(_to_v3_conflict(c))
        else:
            transformed.append(c)

    return transformed


# ---------------------------------------------------------------------------
#  Apply full drift to an observation dict
# ---------------------------------------------------------------------------

def apply_drift(obs_data: Dict[str, Any], version: str) -> Dict[str, Any]:
    """
    Apply schema drift to all relevant fields of an observation dict.
    This is the main entry point used by env.py.
    """
    obs_data = copy.deepcopy(obs_data)

    # Transform calendar events
    if "calendar" in obs_data and "events" in obs_data["calendar"]:
        obs_data["calendar"]["events"] = transform_calendar(
            obs_data["calendar"]["events"], version
        )

    # Transform policy rules
    if "policy_rules" in obs_data:
        obs_data["policy_rules"] = transform_policy(
            obs_data["policy_rules"], version
        )

    # Transform actor profiles
    if "actor_profiles" in obs_data:
        obs_data["actor_profiles"] = transform_actor_profiles(
            obs_data["actor_profiles"], version
        )

    # Transform active conflicts
    if "active_conflicts" in obs_data:
        obs_data["active_conflicts"] = transform_conflicts(
            obs_data["active_conflicts"], version
        )

    # Update schema version marker
    obs_data["schema_version"] = version

    return obs_data


# ===================================================================
#  V2 transformers (mild drift -- field renames, date format changes)
# ===================================================================

def _to_v2_event(ev: Dict) -> Dict:
    """V1 -> V2: camelCase keys, US date format."""
    return {
        "event_id": ev.get("event_id", ""),
        "title": ev.get("title", ""),
        "startTime": _date_v1_to_v2(ev.get("start_time", "")),
        "endTime": _date_v1_to_v2(ev.get("end_time", "")),
        "actorIds": ev.get("actor_ids", []),
        "priority": ev.get("priority", "normal"),
        "isHardDeadline": ev.get("is_hard_deadline", False),
    }


def _to_v2_policy(pol: Dict) -> Dict:
    """V1 -> V2: renamed keys."""
    return {
        "cancellation_fee": 0 if pol.get("free_cancel", True) else pol.get("rebooking_fee", 50),
        "rebooking_charge": pol.get("rebooking_fee", 0),
        "cancelWindowHours": pol.get("cancel_window_hrs", 24),
    }


def _to_v2_actor(profile: Dict) -> Dict:
    """V1 -> V2: preferred_times -> availability.preferred."""
    prefs = profile.pop("preferred_times", [])
    profile["availability"] = {"preferred": prefs}
    # Rename flexibility
    if "flexibility" in profile:
        profile["flex_level"] = profile.pop("flexibility")
    return profile


def _to_v2_conflict(c: Dict) -> Dict:
    """V1 -> V2: renamed keys, uppercase type."""
    return {
        "conflict_id": c.get("conflict_id", ""),
        "conflict_kind": c.get("type", "overlap").upper(),
        "eventIds": c.get("event_ids", []),
        "details": c.get("description", ""),
    }


# ===================================================================
#  V3 transformers (heavy drift -- nested objects, structural changes)
# ===================================================================

def _to_v3_event(ev: Dict) -> Dict:
    """V1 -> V3: nested schedule object, structured date."""
    return {
        "event_id": ev.get("event_id", ""),
        "title": ev.get("title", ""),
        "schedule": {
            "begin": _date_v1_to_v3(ev.get("start_time", "")),
            "end": _date_v1_to_v3(ev.get("end_time", "")),
        },
        "participants": ev.get("actor_ids", []),
        "meta": {
            "priority": ev.get("priority", "normal"),
            "hard_deadline": ev.get("is_hard_deadline", False),
        },
    }


def _to_v3_policy(pol: Dict) -> Dict:
    """V1 -> V3: deeply nested policy structure."""
    return {
        "policy": {
            "cancel": {
                "fee_usd": 0 if pol.get("free_cancel", True) else pol.get("rebooking_fee", 50),
                "window_hrs": pol.get("cancel_window_hrs", 24),
            },
            "rebook": {
                "fee_usd": pol.get("rebooking_fee", 0),
                "allowed": True,
            },
        }
    }


def _to_v3_actor(profile: Dict) -> Dict:
    """V1 -> V3: scheduling_prefs.ideal_slots."""
    prefs = profile.pop("preferred_times", [])
    profile["scheduling_prefs"] = {"ideal_slots": prefs, "buffer_minutes": 15}
    # Rename flexibility
    if "flexibility" in profile:
        profile["adaptability"] = profile.pop("flexibility")
    return profile


def _to_v3_conflict(c: Dict) -> Dict:
    """V1 -> V3: nested issue category."""
    type_map = {
        "overlap": "TIME_COLLISION",
        "cascade": "DEPENDENCY_CHAIN",
        "deadline": "DEADLINE_BREACH",
        "preference": "PREFERENCE_CONFLICT",
    }
    return {
        "conflict_id": c.get("conflict_id", ""),
        "issue": {
            "category": type_map.get(c.get("type", "overlap"), "UNKNOWN"),
        },
        "involved_events": c.get("event_ids", []),
        "summary": c.get("description", ""),
    }


# ---------------------------------------------------------------------------
#  Date format converters
# ---------------------------------------------------------------------------

def _date_v1_to_v2(date_str: str) -> str:
    """
    Convert V1 date to V2 format.
    "2026-04-25 09:00" -> "04/25/2026 9:00AM"
    """
    if not date_str:
        return ""
    try:
        date_part, time_part = date_str.split(" ")
        year, month, day = date_part.split("-")
        hour, minute = time_part.split(":")
        hour_int = int(hour)
        ampm = "AM" if hour_int < 12 else "PM"
        display_hour = hour_int if hour_int <= 12 else hour_int - 12
        if display_hour == 0:
            display_hour = 12
        return f"{month}/{day}/{year} {display_hour}:{minute}{ampm}"
    except (ValueError, IndexError):
        return date_str


def _date_v1_to_v3(date_str: str) -> Dict:
    """
    Convert V1 date to V3 structured format.
    "2026-04-25 09:00" -> {"day": 25, "month": 4, "year": 2026, "hour": 9, "minute": 0}
    """
    if not date_str:
        return {"day": 0, "month": 0, "year": 0, "hour": 0, "minute": 0}
    try:
        date_part, time_part = date_str.split(" ")
        year, month, day = date_part.split("-")
        hour, minute = time_part.split(":")
        return {
            "day": int(day),
            "month": int(month),
            "year": int(year),
            "hour": int(hour),
            "minute": int(minute),
        }
    except (ValueError, IndexError):
        return {"day": 0, "month": 0, "year": 0, "hour": 0, "minute": 0}
