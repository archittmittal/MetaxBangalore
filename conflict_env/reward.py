"""
ConflictEnv -- Reward Computation
=================================
Multi-signal reward function normalized to (0.05, 0.95) for OpenEnv compliance.

Reward = 0.40 x Conflict Resolution Rate (CRR)
       + 0.30 x Stakeholder Satisfaction Index (SSI)
       + 0.20 x Deadline Adherence
       + 0.10 x Efficiency

Key metrics exported for demo:
  - CRR: % of conflicts fully resolved
  - SSI: Mean satisfaction across all actors
  - DriftAdapt: CRR maintained after schema drift vs. pre-drift CRR
"""

from __future__ import annotations

from typing import Any, Dict, List

from .actors import Actor


# ---------------------------------------------------------------------------
#  Reward weights
# ---------------------------------------------------------------------------

W_CONFLICT_RESOLUTION = 0.40
W_STAKEHOLDER_SATISFACTION = 0.30
W_DEADLINE_ADHERENCE = 0.20
W_EFFICIENCY = 0.10

REWARD_FLOOR = 0.05
REWARD_CEILING = 0.95


# ---------------------------------------------------------------------------
#  Main reward computation
# ---------------------------------------------------------------------------

def compute_reward(
    conflicts: List[Dict[str, Any]],
    actors: Dict[str, Actor],
    hard_deadlines: List[str],
    hard_deadlines_met: List[str],
    steps_taken: int,
    max_steps: int,
    escalated: bool = False,
) -> Dict[str, float]:
    """
    Compute the multi-signal reward.

    Returns a dict with:
      - "reward": The final clamped reward (0.05-0.95)
      - "crr": Conflict Resolution Rate
      - "ssi": Stakeholder Satisfaction Index
      - "deadline_score": Deadline adherence ratio
      - "efficiency": Efficiency score
      - "components": Dict of weighted components
    """
    # --- 1. Conflict Resolution Rate (CRR) ---
    total_conflicts = len(conflicts)
    resolved = sum(1 for c in conflicts if c.get("resolved", False))
    crr = resolved / max(total_conflicts, 1)

    # --- 2. Stakeholder Satisfaction Index (SSI) ---
    if actors:
        satisfactions = [a.satisfaction for a in actors.values()]
        ssi = sum(satisfactions) / len(satisfactions)
    else:
        ssi = 0.5

    # --- 3. Deadline Adherence ---
    total_deadlines = max(len(hard_deadlines), 1)
    deadlines_met = len(hard_deadlines_met)
    deadline_score = deadlines_met / total_deadlines

    # --- 4. Efficiency ---
    efficiency = max(0.0, 1.0 - (steps_taken / max(max_steps, 1)))

    # --- Weighted sum ---
    raw = (
        W_CONFLICT_RESOLUTION * crr
        + W_STAKEHOLDER_SATISFACTION * ssi
        + W_DEADLINE_ADHERENCE * deadline_score
        + W_EFFICIENCY * efficiency
    )

    # --- Escalation penalty ---
    if escalated:
        raw *= 0.3  # Severe penalty -- escalating to human is a failure state

    # --- Clamp to safe OpenEnv range ---
    final_reward = clamp(raw, REWARD_FLOOR, REWARD_CEILING)

    return {
        "reward": round(final_reward, 4),
        "crr": round(crr, 4),
        "ssi": round(ssi, 4),
        "deadline_score": round(deadline_score, 4),
        "efficiency": round(efficiency, 4),
        "components": {
            "conflict_resolution": round(W_CONFLICT_RESOLUTION * crr, 4),
            "stakeholder_satisfaction": round(W_STAKEHOLDER_SATISFACTION * ssi, 4),
            "deadline_adherence": round(W_DEADLINE_ADHERENCE * deadline_score, 4),
            "efficiency": round(W_EFFICIENCY * efficiency, 4),
        },
    }


# ---------------------------------------------------------------------------
#  Step-level reward delta
# ---------------------------------------------------------------------------

def compute_step_reward(
    action_result: str,
    actor_satisfaction_delta: float = 0.0,
    conflict_resolved: bool = False,
    deadline_met: bool = False,
    escalated: bool = False,
) -> float:
    """
    Compute a small immediate reward delta for a single step.
    This provides incremental signal while keeping the main reward sparse.

    Returns a value in [-0.1, +0.15].
    """
    delta = 0.0

    # Positive signals
    if conflict_resolved:
        delta += 0.08
    if deadline_met:
        delta += 0.05
    if actor_satisfaction_delta > 0:
        delta += min(actor_satisfaction_delta * 0.1, 0.05)

    # Negative signals
    if escalated:
        delta -= 0.10
    if actor_satisfaction_delta < 0:
        delta += actor_satisfaction_delta * 0.05  # Already negative

    # Action-specific micro-rewards
    if action_result == "confirmed":
        delta += 0.02
    elif action_result == "invalid_action":
        delta -= 0.03

    return round(clamp(delta, -0.10, 0.15), 4)


# ---------------------------------------------------------------------------
#  Utilities
# ---------------------------------------------------------------------------

def clamp(value: float, floor: float, ceiling: float) -> float:
    """Clamp a value to [floor, ceiling]."""
    return max(floor, min(ceiling, value))


def compute_crr(conflicts: List[Dict]) -> float:
    """Standalone CRR computation for logging."""
    total = len(conflicts)
    if total == 0:
        return 1.0
    resolved = sum(1 for c in conflicts if c.get("resolved", False))
    return round(resolved / total, 4)


def compute_ssi(actors: Dict[str, Actor]) -> float:
    """Standalone SSI computation for logging."""
    if not actors:
        return 0.5
    return round(sum(a.satisfaction for a in actors.values()) / len(actors), 4)
