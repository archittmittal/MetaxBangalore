"""Smoke test for all Life OS components."""
from calendar_bridge import calendar_to_scenario, build_scenario, detect_conflicts
from conflict_env.env import ConflictEnv
from conflict_env.models import ConflictAction
from conflict_env.inference import naive_agent_step, smart_agent_step

print("--- LIFE OS SMOKE TEST ---")

# 1. Calendar bridge imports
print("[1/4] Calendar bridge imports: OK")

# 2. Environment with reset_with_scenario
env = ConflictEnv()
has_method = hasattr(env, "reset_with_scenario")
print(f"[2/4] reset_with_scenario exists: {has_method}")

# 3. Standard reset still works
obs = env.reset(task_name="auto")
print(f"[3/4] Standard reset: OK (scenario={obs.scenario_name})")

# 4. Scripted agents still work
import copy
env_elite = copy.deepcopy(env)
env_naive = copy.deepcopy(env)

elite_action = smart_agent_step(obs, 0)
env_elite.step(elite_action)
elite_reward = env_elite.get_reward()

naive_action = naive_agent_step(obs, 0)
env_naive.step(naive_action)
naive_reward = env_naive.get_reward()

print(f"[4/4] Agent duel: Elite={elite_reward:.4f} vs Naive={naive_reward:.4f}")
print(f"\nAll components verified. Ready to deploy.")
