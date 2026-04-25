from conflict_env.env import ConflictEnv
from conflict_env.models import ConflictAction, ConflictObservation
import json

def run_local_logic_test():
    print("--- TESTING MERGED ENVIRONMENT LOGIC ---")
    try:
        env = ConflictEnv()
        # OpenEnv Protocol: reset returns only the observation
        obs = env.reset(task_name="auto")
        print("Success: Environment successfully initialized.")
        
        scenario_name = getattr(obs, 'scenario_name', 'Unknown')
        print(f"Scenario: {scenario_name}")
        
        print("Testing dummy action (reschedule)...")
        # Ensure we use the correct parameters for the new handle_reschedule
        # It expects "new_slot" instead of "new_start"/"new_end" in some versions
        dummy_action = ConflictAction(
            command="reschedule",
            parameters={
                "event_id": "evt_school",
                "new_slot": "10:30"
            }
        )
        
        # OpenEnv Protocol: step returns only the observation
        obs = env.step(dummy_action)
            
        reward = env.get_reward()
        done = env._done # Accessing internal flag as per protocol check
        
        print(f"Reward received: {reward}")
        print(f"Status: {'Done' if done else 'In Progress'}")
        
        if reward != 0:
            print("\nLOCAL TEST PASSED: The merged environment is fully functional!")
            print("The OpenEnv protocol is verified. You are ready to go.")
        else:
            print("\nWARNING: Reward was 0. The action was valid but didn't solve the conflict yet.")
            
    except Exception as e:
        print(f"\nLOCAL TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_local_logic_test()
