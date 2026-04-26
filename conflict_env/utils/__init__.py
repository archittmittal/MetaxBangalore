import json
import os
import datetime

DATA_FILE = "experience_buffer.jsonl"

def log_experience(state, action, reward, feedback, next_state, done):
    """
    Log an experience tuple for later fine-tuning (Continuous Learning).
    This collects data that can be used for GRPO/PPO training.
    """
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "state": state,
        "action": action,
        "reward": reward,
        "feedback": feedback,
        "next_state": next_state,
        "done": done
    }
    
    try:
        with open(DATA_FILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        # Don't crash the env if logging fails
        print(f"Logging failed: {e}")

def get_buffer_size():
    if not os.path.exists(DATA_FILE):
        return 0
    with open(DATA_FILE, "r") as f:
        return sum(1 for _ in f)
