"""
ConflictEnv -- Shared Constants
================================
Centralized configuration to avoid magic numbers.
"""

# Reward ranges
REWARD_FLOOR = 0.05
REWARD_CEILING = 0.95

# Training rewards
BASE_RESOLUTION_REWARD = 0.40
SATISFACTION_WEIGHT = 0.30
DEADLINE_WEIGHT = 0.20
EFFICIENCY_WEIGHT = 0.10
REASONING_BONUS = 0.10

# Environment limits
DEFAULT_MAX_STEPS = 15
MAX_EVENTS = 8
MAX_CONFLICTS = 6

# Time parameters
MINUTES_IN_DAY = 1440
SLOT_DURATION_MINUTES = 60

# Drift versions
DRIFT_V1 = "v1"
DRIFT_V2 = "v2"
DRIFT_V3 = "v3"
