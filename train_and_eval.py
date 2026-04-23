"""
ConflictEnv -- Training & Evaluation Pipeline
===============================================
Trains a PPO agent on ConflictEnv with curriculum learning,
evaluates it across difficulty tiers and schema drift versions,
and produces a results summary.

Usage:
  python train_and_eval.py                  # Full pipeline (train + eval)
  python train_and_eval.py --eval-only      # Evaluate saved model
  python train_and_eval.py --timesteps 50000 # Custom training duration
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

# Ensure local package is importable
sys.path.insert(0, os.path.abspath(os.getcwd()))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from conflict_env.gym_wrapper import ConflictGymWrapper


# ---------------------------------------------------------------------------
#  Custom callback for logging episode metrics
# ---------------------------------------------------------------------------

class ConflictMetricsCallback(BaseCallback):
    """Logs CRR, SSI, and episode reward to TensorBoard."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._episode_rewards = []
        self._episode_crrs = []
        self._episode_ssis = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "crr" in info:
                self._episode_crrs.append(info["crr"])
                self._episode_ssis.append(info["ssi"])
            if "cumulative_reward" in info and info.get("TimeLimit.truncated", False) or self.locals.get("dones", [False])[0]:
                self._episode_rewards.append(info["cumulative_reward"])

        # Log every 10 episodes
        if len(self._episode_rewards) > 0 and len(self._episode_rewards) % 10 == 0:
            avg_reward = sum(self._episode_rewards[-10:]) / 10
            avg_crr = sum(self._episode_crrs[-10:]) / max(len(self._episode_crrs[-10:]), 1)
            avg_ssi = sum(self._episode_ssis[-10:]) / max(len(self._episode_ssis[-10:]), 1)

            self.logger.record("conflict/avg_reward", avg_reward)
            self.logger.record("conflict/avg_crr", avg_crr)
            self.logger.record("conflict/avg_ssi", avg_ssi)
            self.logger.record("conflict/episodes", len(self._episode_rewards))

            if self.verbose:
                print(f"  [EP {len(self._episode_rewards):>4d}] "
                      f"Reward={avg_reward:.4f}  CRR={avg_crr:.2f}  SSI={avg_ssi:.2f}")

        return True


# ---------------------------------------------------------------------------
#  Training
# ---------------------------------------------------------------------------

def train(timesteps: int = 10000, model_path: str = "conflict_ppo"):
    """Train a PPO agent on ConflictEnv with curriculum learning."""

    print("\n" + "=" * 60)
    print("  PHASE 1: TRAINING (PPO + Curriculum Learning)")
    print("=" * 60)

    # Create environment with curriculum enabled
    env = ConflictGymWrapper(difficulty="easy", curriculum=True)

    # PPO hyperparameters tuned for this environment
    model = PPO(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,       # Encourage exploration
        tensorboard_log="./logs/ppo/",
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128])
        ),
    )

    print(f"  Model: PPO (MLP 128x128)")
    print(f"  Training for {timesteps:,} timesteps")
    print(f"  Curriculum: easy -> medium -> hard")
    print(f"  Logging to: ./logs/ppo/")
    print()

    callback = ConflictMetricsCallback(verbose=1)
    start = time.time()
    model.learn(total_timesteps=timesteps, callback=callback)
    elapsed = time.time() - start

    model.save(model_path)
    print(f"\n  Training complete in {elapsed:.1f}s")
    print(f"  Model saved to: {model_path}.zip")

    return model


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, n_episodes: int = 5):
    """Evaluate the trained model across scenarios and drift versions."""

    print("\n" + "=" * 60)
    print("  PHASE 2: EVALUATION")
    print("=" * 60)

    scenarios = [
        "morning_crunch_easy",
        "travel_chaos_medium",
        "monday_from_hell_hard",
        "deadline_squeeze_medium",
    ]

    results = []

    for scenario in scenarios:
        episode_rewards = []
        episode_crrs = []
        episode_ssis = []
        episode_steps = []

        for ep in range(n_episodes):
            env = ConflictGymWrapper()
            obs, _ = env.reset(options={"scenario": scenario})
            done = False
            total_shaped_reward = 0.0
            steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_shaped_reward += reward
                steps += 1
                done = terminated or truncated

            # Get the environment's true reward (not the shaped one)
            true_reward = env.env.get_reward()
            crr = info.get("crr", 0.0)
            ssi = info.get("ssi", 0.5)

            episode_rewards.append(true_reward)
            episode_crrs.append(crr)
            episode_ssis.append(ssi)
            episode_steps.append(steps)

        avg_reward = sum(episode_rewards) / len(episode_rewards)
        avg_crr = sum(episode_crrs) / len(episode_crrs)
        avg_ssi = sum(episode_ssis) / len(episode_ssis)
        avg_steps = sum(episode_steps) / len(episode_steps)

        results.append({
            "scenario": scenario,
            "avg_reward": round(avg_reward, 4),
            "avg_crr": round(avg_crr, 4),
            "avg_ssi": round(avg_ssi, 4),
            "avg_steps": round(avg_steps, 1),
        })

    # Print results table
    print(f"\n  {'Scenario':<30} {'Reward':>8} {'CRR':>6} {'SSI':>6} {'Steps':>6}")
    print(f"  {'-'*58}")
    for r in results:
        print(f"  {r['scenario']:<30} {r['avg_reward']:>8.4f} {r['avg_crr']:>6.2f} "
              f"{r['avg_ssi']:>6.2f} {r['avg_steps']:>6.1f}")

    # Overall
    overall_reward = sum(r["avg_reward"] for r in results) / len(results)
    overall_crr = sum(r["avg_crr"] for r in results) / len(results)
    overall_ssi = sum(r["avg_ssi"] for r in results) / len(results)
    print(f"  {'-'*58}")
    print(f"  {'OVERALL':<30} {overall_reward:>8.4f} {overall_crr:>6.2f} {overall_ssi:>6.2f}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "per_scenario": results,
        "overall": {
            "avg_reward": round(overall_reward, 4),
            "avg_crr": round(overall_crr, 4),
            "avg_ssi": round(overall_ssi, 4),
        }
    }
    os.makedirs("results", exist_ok=True)
    with open("results/eval_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: results/eval_results.json")

    return results


# ---------------------------------------------------------------------------
#  Schema Drift Stress Test
# ---------------------------------------------------------------------------

def drift_stress_test(model):
    """Test how the RL agent degrades across schema drift versions."""

    print("\n" + "=" * 60)
    print("  PHASE 3: SCHEMA DRIFT STRESS TEST")
    print("=" * 60)

    drift_configs = [
        ("V1 (Baseline)", 0),
        ("V2 (Mild Drift)", 50),
        ("V3 (Heavy Drift)", 100),
    ]

    print(f"\n  {'Drift Version':<25} {'Reward':>8} {'CRR':>6} {'SSI':>6}")
    print(f"  {'-'*47}")

    for label, episode_offset in drift_configs:
        env = ConflictGymWrapper()
        # Force drift version by setting episode count
        env.env._episode_count = episode_offset

        rewards = []
        for _ in range(5):
            obs, _ = env.reset(options={"scenario": "morning_crunch_easy"})
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            rewards.append(env.env.get_reward())

        avg = sum(rewards) / len(rewards)
        crr = info.get("crr", 0)
        ssi = info.get("ssi", 0.5)
        print(f"  {label:<25} {avg:>8.4f} {crr:>6.2f} {ssi:>6.2f}")

    print("\n  If V3 << V1, this proves schema drift degrades RL agents")
    print("  and motivates the need for LLM-based reasoning agents.")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ConflictEnv Training Pipeline")
    parser.add_argument("--timesteps", type=int, default=10000,
                        help="Training timesteps (default: 10000)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip training; evaluate saved model")
    parser.add_argument("--model", type=str, default="conflict_ppo",
                        help="Model path (default: conflict_ppo)")
    args = parser.parse_args()

    print("\n" + "#" * 60)
    print("  ConflictEnv: RL Training & Evaluation Pipeline")
    print("#" * 60)

    if args.eval_only:
        model = PPO.load(args.model)
        print(f"  Loaded model from: {args.model}.zip")
    else:
        model = train(timesteps=args.timesteps, model_path=args.model)

    evaluate(model)
    drift_stress_test(model)

    print("\n" + "#" * 60)
    print("  Pipeline Complete")
    print("#" * 60 + "\n")


if __name__ == "__main__":
    main()
