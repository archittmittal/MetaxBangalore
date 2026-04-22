import requests
import sys
import json

BASE_URL = "http://127.0.0.1:7860"

def test_health():
    print("Testing /health...")
    try:
        resp = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"Status: {resp.status_code}")
        print(f"Body: {resp.json()}")
        assert resp.status_code == 200
        assert resp.json().get("status") == "healthy"
        print("  [PASS] Health check")
    except Exception as e:
        print(f"  [FAIL] Health check failed: {e}")
        raise

def test_workflow():
    print("\nTesting full workflow (reset -> step -> state)...")
    
    # 1. Reset
    print("  -> POST /reset")
    # Note: openenv-core wraps observation in a top-level field
    reset_data = {"seed": 42} 
    try:
        resp = requests.post(f"{BASE_URL}/reset", json=reset_data, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        obs = data.get("observation", {})
        print(f"  Reset Status: {resp.status_code}")
        print(f"  Initial Feedback: {obs.get('feedback', 'EMPTY')[:100]}...")
        
        assert "calendar" in obs, "Observation missing 'calendar'"
        assert "active_conflicts" in obs, "Observation missing 'active_conflicts'"
        print("  [PASS] Reset")
    except Exception as e:
        print(f"  [FAIL] Reset failed: {e}")
        if 'resp' in locals(): print(f"  Response: {resp.text}")
        raise

    # 2. State
    print("  -> GET /state")
    try:
        resp = requests.get(f"{BASE_URL}/state", timeout=5)
        resp.raise_for_status()
        state_data = resp.json() 
        print(f"  State Status: {resp.status_code}")
        # ConflictState wraps the observation in 'obs'
        assert "obs" in state_data, "State response missing 'obs'"
        assert "calendar" in state_data["obs"], "Observation inside state missing 'calendar'"
        print("  [PASS] State")
    except Exception as e:
        print(f"  [FAIL] State failed: {e}")
        if 'resp' in locals():
            print(f"  Response Body: {json.dumps(resp.json(), indent=2)}")
        raise

    # 3. Step
    print("  -> POST /step")
    # Dynamically pick an actor from the current state
    actors = state_data["obs"].get("actor_profiles", {})
    target_actor = list(actors.keys())[0] if actors else "boss"
    print(f"  Targeting actor: {target_actor}")
    
    step_data = {
        "action": {
            "command": "query_preference",
            "parameters": {"actor_id": target_actor}
        }
    }
    try:
        resp = requests.post(f"{BASE_URL}/step", json=step_data, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        obs_after = data.get("observation", {})
        print(f"  Step Status: {resp.status_code}")
        feedback = obs_after.get('feedback', 'EMPTY')
        print(f"  Feedback: {feedback[:100]}...")
        
        # Check for [ERROR] or lack of info
        if "[ERROR]" in feedback:
            raise AssertionError(f"Step failed with error in feedback: {feedback}")
        if "preference" not in feedback.lower() and "flexibility" not in feedback.lower():
            raise AssertionError(f"Feedback missing expected info for {target_actor}")
            
        print("  [PASS] Step")
    except Exception as e:
        print(f"  [FAIL] Step failed: {e}")
        if 'resp' in locals(): print(f"  Response: {resp.text}")
        raise

    print("\n[SUCCESS] API Protocol Testing Complete")

if __name__ == "__main__":
    print(f"Testing ConflictEnv OpenEnv Protocol Compliance")
    try:
        test_health()
        test_workflow()
    except Exception:
        sys.exit(1)
