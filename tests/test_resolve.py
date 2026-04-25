"""Quick test of the resolve_conflict function."""
import copy
import json
from conflict_env.env import ConflictEnv
from conflict_env.models import ConflictAction
from conflict_env.inference import naive_agent_step, smart_agent_step

env = ConflictEnv()

print("--- Testing resolve_conflict flow ---")
try:
    obs = env.reset(task_name="auto")
    print(f"Reset OK: scenario={obs.scenario_name}, conflicts={len(obs.active_conflicts)}")

    env_elite = copy.deepcopy(env)
    print("Deepcopy OK")

    env_naive = copy.deepcopy(env)
    print("Deepcopy 2 OK")

    # Run smart agent
    print("\nRunning smart agent...")
    for step in range(5):
        action = smart_agent_step(obs, step)
        print(f"  Step {step}: {action.command} {action.parameters}")
        obs_e = env_elite.step(action)
        print(f"    Feedback: {obs_e.feedback[:80]}")
        if obs_e.done:
            break
    elite_reward = env_elite.get_reward()
    print(f"Elite reward: {elite_reward}")

    # Run naive agent
    print("\nRunning naive agent...")
    obs2 = env._current_obs  # Reset obs for naive
    for step in range(3):
        action = naive_agent_step(obs2, step)
        print(f"  Step {step}: {action.command} {action.parameters}")
        obs_n = env_naive.step(action)
        print(f"    Feedback: {obs_n.feedback[:80]}")
        if obs_n.done:
            break
    naive_reward = env_naive.get_reward()
    print(f"Naive reward: {naive_reward}")

    print(f"\nSUCCESS: Elite={elite_reward:.4f} vs Naive={naive_reward:.4f}")

except Exception as e:
    import traceback
    print(f"\nFAILED: {e}")
    traceback.print_exc()
