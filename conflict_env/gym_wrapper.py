import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .env import ConflictEnv
from .models import ConflictAction

class ConflictGymWrapper(gym.Env):
    """
    Gymnasium wrapper for ConflictEnv to enable standard RL training.
    
    Observation Space:
    - [0-7]: Normalized satisfaction for key actors (boss, spouse, client, friend, vendor, doctor, child, self)
    - [8]: Number of active conflicts (normalized by 10)
    - [9]: Step ratio (step_count / max_steps)
    - [10-12]: One-hot encoding of schema version (v1, v2, v3)
    - [13-15]: One-hot encoding of difficulty (easy, medium, hard)
    
    Action Space (Discrete):
    - 0: query_preference (primary actor)
    - 1: reschedule (first conflict event)
    - 2: cancel (first conflict event)
    - 3: confirm (first conflict event)
    - 4: resolve
    """
    
    def __init__(self, config=None):
        super().__init__()
        self.env = ConflictEnv()
        
        # Observation space: 16-dimensional vector
        self.observation_space = spaces.Box(low=0, high=1, shape=(16,), dtype=np.float32)
        
        # Action space: 5 discrete commands
        self.action_space = spaces.Discrete(5)
        
        self.actor_list = ["boss", "spouse", "client", "friend", "vendor", "doctor", "child", "self"]

    def _get_obs(self, obs_data):
        """Vectorize the ConflictObservation dictionary."""
        vec = np.zeros(16, dtype=np.float32)
        
        # 1. Satisfaction (0-7)
        profiles = obs_data.actor_profiles
        for i, actor_id in enumerate(self.actor_list):
            if actor_id in profiles:
                # Handle nested drift structures if necessary, but ConflictEnv profiles usually have satisfaction
                # We'll assume a base satisfaction or 0.5
                vec[i] = profiles[actor_id].get("satisfaction", 0.5)
            else:
                vec[i] = 0.5
                
        # 2. Conflict Count (8)
        vec[8] = min(len(obs_data.active_conflicts) / 10.0, 1.0)
        
        # 3. Step Ratio (9)
        vec[9] = obs_data.step_count / max(obs_data.max_steps, 1)
        
        # 4. Schema Version (10-12)
        version_map = {"v1": 10, "v2": 11, "v3": 12}
        idx = version_map.get(obs_data.schema_version, 10)
        vec[idx] = 1.0
        
        # 5. Difficulty (13-15)
        diff_map = {"easy": 13, "medium": 14, "hard": 15}
        idx = diff_map.get(obs_data.difficulty, 13)
        vec[idx] = 1.0
        
        return vec

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Handle options for scenario name
        scenario = options.get("scenario", "easy") if options else "easy"
        obs_data = self.env.reset(scenario)
        return self._get_obs(obs_data), {}

    def step(self, action):
        """Map discrete action to ConflictAction and execute."""
        obs_data = self.env.state.obs # Current state
        
        cmd = "resolve"
        params = {}
        
        # Logic to pick targets for the discrete actions
        target_conflict = obs_data.active_conflicts[0] if obs_data.active_conflicts else None
        target_event = target_conflict.get("event_ids", [""])[0] if target_conflict else ""
        target_actor = list(obs_data.actor_profiles.keys())[0] if obs_data.actor_profiles else "boss"

        if action == 0: # query_preference
            cmd = "query_preference"
            params = {"actor_id": target_actor}
        elif action == 1: # reschedule
            cmd = "reschedule"
            # Simple heuristic: move to a later slot
            params = {"event_id": target_event, "new_slot": "14:00"}
        elif action == 2: # cancel
            cmd = "cancel"
            params = {"event_id": target_event}
        elif action == 3: # confirm
            cmd = "confirm"
            params = {"event_id": target_event}
        elif action == 4: # resolve
            cmd = "resolve"
            params = {}

        action_obj = ConflictAction(command=cmd, parameters=params)
        new_obs_data = self.env.step(action_obj)
        
        obs_vec = self._get_obs(new_obs_data)
        reward = new_obs_data.reward
        terminated = new_obs_data.done
        truncated = False # We handle max steps inside the env
        
        return obs_vec, reward, terminated, truncated, {"feedback": new_obs_data.feedback}

    def render(self):
        print(self.env.state.obs.feedback)
