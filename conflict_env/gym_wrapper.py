"""
ConflictEnv -- Gymnasium Wrapper
=================================
Bridges ConflictEnv to the Gymnasium API for standard RL training
with Stable Baselines3 (PPO, DQN, A2C, etc.).

Key design decisions:
  - Richer observation (32-dim) capturing per-conflict and per-actor signals
  - Multi-target action space using MultiDiscrete (command x target)
  - Curriculum learning support via difficulty scheduling
  - Dense reward shaping for faster convergence
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .env import ConflictEnv
from .models import ConflictAction


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

COMMANDS = ["query_preference", "reschedule", "cancel", "confirm", "resolve"]
RESCHEDULE_SLOTS = ["08:00", "09:00", "10:00", "11:00", "12:00",
                    "13:00", "14:00", "15:00", "16:00", "17:00"]
ACTOR_KEYS = ["boss", "spouse", "client", "friend", "vendor",
              "doctor", "child", "self"]
MAX_EVENTS = 8
MAX_CONFLICTS = 6


class ConflictGymWrapper(gym.Env):
    """
    Gymnasium wrapper for ConflictEnv.

    Observation (32-dim float32 vector):
      [0-7]   Actor satisfaction levels (8 actors)
      [8]     Number of active (unresolved) conflicts, normalized
      [9]     Number of total events, normalized
      [10]    Step progress ratio (step / max_steps)
      [11]    Number of confirmed events, normalized
      [12]    Number of cancelled events, normalized
      [13]    Escalation flag (0 or 1)
      [14-16] Schema version one-hot (v1, v2, v3)
      [17-19] Difficulty one-hot (easy, medium, hard)
      [20-25] Per-conflict resolved flags (up to 6 conflicts)
      [26-31] Per-event status flags (up to 6 events: 1=confirmed, 0.5=active, 0=cancelled)

    Action: Discrete(25)
      action = command_idx * 5 + target_idx
      - command_idx in [0..4]: query_preference, reschedule, cancel, confirm, resolve
      - target_idx in [0..4]: indexes into available events/actors
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, difficulty="easy", curriculum=False):
        super().__init__()

        self.env = ConflictEnv()
        self.base_difficulty = difficulty
        self.curriculum = curriculum
        self._episode_num = 0

        # Spaces
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(32,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(25)  # 5 commands x 5 targets

        # Caches for mapping action targets
        self._event_ids: list = []
        self._actor_ids: list = []
        self._last_obs_data = None

    # ------------------------------------------------------------------
    #  Gym API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Curriculum: escalate difficulty over episodes
        if self.curriculum:
            difficulty = self._curriculum_difficulty()
        elif options and "scenario" in options:
            difficulty = options["scenario"]
        else:
            difficulty = self.base_difficulty

        self._last_obs_data = self.env.reset(difficulty)
        self._cache_ids(self._last_obs_data)
        self._episode_num += 1

        return self._vectorize(self._last_obs_data), {}

    def step(self, action):
        # Decode action
        command_idx = int(action) // 5
        target_idx = int(action) % 5
        conflict_action = self._decode_action(command_idx, target_idx)

        # Execute
        self._last_obs_data = self.env.step(conflict_action)
        self._cache_ids(self._last_obs_data)

        obs_vec = self._vectorize(self._last_obs_data)

        # Dense reward: use step reward + bonus shaping
        reward = self._shape_reward(self._last_obs_data)

        terminated = self._last_obs_data.done
        truncated = False

        info = {
            "feedback": self._last_obs_data.feedback,
            "crr": self._compute_crr(),
            "ssi": self._compute_ssi(),
            "cumulative_reward": self.env.get_reward(),
        }

        return obs_vec, reward, terminated, truncated, info

    def render(self):
        if self._last_obs_data:
            print(f"[Step {self._last_obs_data.step_count}] {self._last_obs_data.feedback}")

    # ------------------------------------------------------------------
    #  Observation vectorization
    # ------------------------------------------------------------------

    def _vectorize(self, obs) -> np.ndarray:
        """Convert ConflictObservation to a 32-dim numpy vector."""
        vec = np.zeros(32, dtype=np.float32)

        # [0-7] Actor satisfaction
        profiles = obs.actor_profiles
        for i, key in enumerate(ACTOR_KEYS):
            if key in profiles:
                vec[i] = np.clip(profiles[key].get("satisfaction", 0.5), 0.0, 1.0)
            else:
                vec[i] = 0.5

        # [8] Active conflict count (normalized)
        active = [c for c in obs.active_conflicts if not c.get("resolved", False)]
        vec[8] = min(len(active) / MAX_CONFLICTS, 1.0)

        # [9] Total event count (normalized)
        events = obs.calendar.get("events", [])
        vec[9] = min(len(events) / MAX_EVENTS, 1.0)

        # [10] Step progress
        vec[10] = obs.step_count / max(obs.max_steps, 1)

        # [11] Confirmed events ratio
        confirmed = sum(1 for e in events if e.get("status") == "confirmed")
        vec[11] = confirmed / max(len(events), 1)

        # [12] Cancelled events ratio
        cancelled = sum(1 for e in events if e.get("status") == "cancelled")
        vec[12] = cancelled / max(len(events), 1)

        # [13] Escalation flag
        vec[13] = 1.0 if "[ESCALATED]" in obs.feedback else 0.0

        # [14-16] Schema version one-hot
        v_map = {"v1": 14, "v2": 15, "v3": 16}
        vec[v_map.get(obs.schema_version, 14)] = 1.0

        # [17-19] Difficulty one-hot
        d_map = {"easy": 17, "medium": 18, "hard": 19}
        vec[d_map.get(obs.difficulty, 17)] = 1.0

        # [20-25] Per-conflict resolved flags
        for i, c in enumerate(obs.active_conflicts[:MAX_CONFLICTS]):
            vec[20 + i] = 1.0 if c.get("resolved", False) else 0.0

        # [26-31] Per-event status
        for i, e in enumerate(events[:MAX_CONFLICTS]):
            status = e.get("status", "active")
            if status == "confirmed":
                vec[26 + i] = 1.0
            elif status == "cancelled":
                vec[26 + i] = 0.0
            else:
                vec[26 + i] = 0.5

        return vec

    # ------------------------------------------------------------------
    #  Action decoding
    # ------------------------------------------------------------------

    def _cache_ids(self, obs):
        """Cache event and actor IDs from the current observation."""
        events = obs.calendar.get("events", [])
        # Use event_id, or fall back to other key names from drift
        self._event_ids = []
        for e in events:
            eid = e.get("event_id", "")
            if eid:
                self._event_ids.append(eid)

        self._actor_ids = list(obs.actor_profiles.keys())

    def _decode_action(self, command_idx, target_idx):
        """Convert (command_idx, target_idx) to a ConflictAction."""
        cmd = COMMANDS[min(command_idx, len(COMMANDS) - 1)]
        params = {}

        if cmd == "query_preference":
            actor = self._get_actor(target_idx)
            params = {"actor_id": actor}

        elif cmd == "reschedule":
            event = self._get_event(target_idx)
            slot = RESCHEDULE_SLOTS[target_idx % len(RESCHEDULE_SLOTS)]
            params = {"event_id": event, "new_slot": slot}

        elif cmd == "cancel":
            event = self._get_event(target_idx)
            params = {"event_id": event}

        elif cmd == "confirm":
            event = self._get_event(target_idx)
            params = {"event_id": event}

        elif cmd == "resolve":
            params = {}

        return ConflictAction(command=cmd, parameters=params)

    def _get_event(self, idx):
        if self._event_ids:
            return self._event_ids[idx % len(self._event_ids)]
        return ""

    def _get_actor(self, idx):
        if self._actor_ids:
            return self._actor_ids[idx % len(self._actor_ids)]
        return "boss"

    # ------------------------------------------------------------------
    #  Reward shaping
    # ------------------------------------------------------------------

    def _shape_reward(self, obs) -> float:
        """
        Dense reward shaping to accelerate learning.
        Combines the environment's step reward with bonus signals.
        """
        base = obs.reward  # Step delta from env

        # Bonus: conflict resolution progress
        total_conflicts = len(obs.active_conflicts) if obs.active_conflicts else 1
        resolved = sum(1 for c in obs.active_conflicts if c.get("resolved", False))
        crr_bonus = 0.05 * (resolved / max(total_conflicts, 1))

        # Bonus: maintaining high satisfaction
        ssi = self._compute_ssi()
        ssi_bonus = 0.02 * max(0, ssi - 0.5)  # Reward for above-baseline satisfaction

        # Penalty: wasting steps (doing nothing useful)
        if "[ERROR]" in obs.feedback:
            step_penalty = -0.05
        else:
            step_penalty = 0.0

        # Terminal bonus: scale by final quality
        terminal_bonus = 0.0
        if obs.done:
            final_reward = self.env.get_reward()
            terminal_bonus = final_reward * 0.5  # Large signal at episode end

        total = base + crr_bonus + ssi_bonus + step_penalty + terminal_bonus
        return float(np.clip(total, -0.5, 1.0))

    def _compute_crr(self) -> float:
        if not self._last_obs_data or not self._last_obs_data.active_conflicts:
            return 1.0
        total = len(self._last_obs_data.active_conflicts)
        resolved = sum(1 for c in self._last_obs_data.active_conflicts if c.get("resolved"))
        return resolved / max(total, 1)

    def _compute_ssi(self) -> float:
        if not self._last_obs_data or not self._last_obs_data.actor_profiles:
            return 0.5
        sats = [p.get("satisfaction", 0.5) for p in self._last_obs_data.actor_profiles.values()]
        return sum(sats) / len(sats) if sats else 0.5

    # ------------------------------------------------------------------
    #  Curriculum
    # ------------------------------------------------------------------

    def _curriculum_difficulty(self) -> str:
        """Progressively increase difficulty over training."""
        if self._episode_num < 50:
            return "easy"
        elif self._episode_num < 150:
            return "medium"
        else:
            return "hard"
