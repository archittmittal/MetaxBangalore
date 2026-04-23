import os
import sys
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Ensure the local package is importable
sys.path.insert(0, os.path.abspath(os.getcwd()))

from conflict_env.gym_wrapper import ConflictGymWrapper
from conflict_env.agents.llm_agent import SimplePromptAgent

def train_rl():
    print("\n--- Phase 1: Training Standard RL (PPO) ---")
    env = ConflictGymWrapper()
    
    # Simple PPO training
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_logs/")
    print("Training for 1000 steps (baseline demo)...")
    model.learn(total_timesteps=1000)
    model.save("conflict_ppo_baseline")
    print("RL Model Saved.")
    return model

def eval_agents(model=None):
    print("\n--- Phase 2: Evaluating Agents ---")
    env = ConflictGymWrapper()
    llm_agent = SimplePromptAgent()
    
    scenarios = ["morning_crunch_easy", "travel_chaos_medium"]
    
    for scenario in scenarios:
        print(f"\n>> SCENARIO: {scenario}")
        
        # RL Evaluation
        if model:
            obs, _ = env.reset(options={"scenario": scenario})
            done = False
            total_reward = 0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated
            print(f"   [RL Agent] Total Reward: {total_reward:.4f}")

        # LLM Agent Evaluation (Mock/Prompt Only)
        obs_vec, _ = env.reset(options={"scenario": scenario})
        # For the prompt agent, we use the internal env state for rich data
        prompt = llm_agent.generate_prompt(env.env.state.obs)
        print(f"   [LLM Agent] Generated prompt (length: {len(prompt)})")
        print(f"   [LLM Agent] (Demo mode: Prompt generation verified)")

def main():
    print("ConflictEnv: Battle of the Agents (RL vs LLM)")
    model = train_rl()
    eval_agents(model)
    print("\nBenchmark Complete.")

if __name__ == "__main__":
    main()
