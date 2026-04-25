"""
ConflictEnv -- Dual-Agent Training & Evaluation Pipeline
==========================================================
Trains an RL agent (PPO) and evaluates it head-to-head against
a Gemini LLM reasoning agent on the same scenarios.

Usage:
  python train_and_eval.py                      # Full pipeline
  python train_and_eval.py --eval-only           # Evaluate saved model + LLM
  python train_and_eval.py --timesteps 50000     # Custom training duration
  python train_and_eval.py --llm-only            # Only run LLM agent
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # Manual .env loading
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

# Ensure local package is importable
sys.path.insert(0, os.path.abspath(os.getcwd()))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from conflict_env.gym_wrapper import ConflictGymWrapper
from conflict_env.agents.llm_agent import GeminiAgent
from conflict_env.env import ConflictEnv
from conflict_env.models import ConflictAction


# ---------------------------------------------------------------------------
#  Custom callback for logging episode metrics
# ---------------------------------------------------------------------------

class ConflictMetricsCallback(BaseCallback):
    """Logs CRR, SSI, and episode reward to TensorBoard."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self._episode_count = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for done, info in zip(dones, infos):
            if done:
                self._episode_count += 1
                crr = info.get("crr", 0)
                ssi = info.get("ssi", 0.5)
                cum_reward = info.get("cumulative_reward", 0)

                self.logger.record("conflict/crr", crr)
                self.logger.record("conflict/ssi", ssi)
                self.logger.record("conflict/cumulative_reward", cum_reward)
                self.logger.record("conflict/episodes", self._episode_count)

                if self.verbose and self._episode_count % 20 == 0:
                    print(f"  [EP {self._episode_count:>4d}] "
                          f"Reward={cum_reward:.4f}  CRR={crr:.2f}  SSI={ssi:.2f}")

        return True


# ---------------------------------------------------------------------------
#  Phase 1: RL Training
# ---------------------------------------------------------------------------

def train_rl(timesteps: int = 10000, model_path: str = "conflict_ppo"):
    """Train a PPO agent on ConflictEnv with curriculum learning."""

    print("\n" + "=" * 60)
    print("  PHASE 1: RL TRAINING (PPO + Curriculum Learning)")
    print("=" * 60)

    env = ConflictGymWrapper(difficulty="easy", curriculum=True)

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
        ent_coef=0.01,
        tensorboard_log="./logs/ppo/",
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
    )

    print(f"  Model: PPO (MLP 128x128)")
    print(f"  Timesteps: {timesteps:,}")
    print(f"  Curriculum: easy -> medium -> hard")
    print()

    callback = ConflictMetricsCallback(verbose=1)
    start = time.time()
    model.learn(total_timesteps=timesteps, callback=callback)
    elapsed = time.time() - start

    model.save(model_path)
    print(f"\n  Training complete in {elapsed:.1f}s")
    print(f"  Model saved: {model_path}.zip")

    return model


# ---------------------------------------------------------------------------
#  Phase 2: RL Evaluation
# ---------------------------------------------------------------------------

def evaluate_rl(model, scenarios, n_episodes=3):
    """Evaluate the RL agent across scenarios."""
    results = []

    for scenario in scenarios:
        rewards, crrs, ssis, steps_list = [], [], [], []

        for _ in range(n_episodes):
            env = ConflictGymWrapper()
            obs, _ = env.reset(options={"scenario": scenario})
            done = False
            steps = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                done = terminated or truncated

            rewards.append(env.env.get_reward())
            crrs.append(info.get("crr", 0))
            ssis.append(info.get("ssi", 0.5))
            steps_list.append(steps)

        results.append({
            "scenario": scenario,
            "avg_reward": round(sum(rewards) / len(rewards), 4),
            "avg_crr": round(sum(crrs) / len(crrs), 4),
            "avg_ssi": round(sum(ssis) / len(ssis), 4),
            "avg_steps": round(sum(steps_list) / len(steps_list), 1),
        })

    return results


# ---------------------------------------------------------------------------
#  Phase 3: LLM Evaluation
# ---------------------------------------------------------------------------

def evaluate_llm(scenarios, n_episodes=1):
    """Evaluate the Gemini LLM agent across scenarios."""

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        print("  [SKIP] GEMINI_API_KEY not set. Skipping LLM evaluation.")
        return []

    agent = GeminiAgent(api_key=api_key, model_name="gemini-2.0-flash")
    results = []

    for scenario in scenarios:
        rewards, crrs, ssis, steps_list = [], [], [], []

        for ep in range(n_episodes):
            env = ConflictEnv()
            obs = env.reset(scenario)
            agent.reset()
            steps = 0

            print(f"    [LLM] {scenario} ep={ep+1} ", end="", flush=True)

            while not obs.done and steps < obs.max_steps:
                action = agent.act(obs)
                obs = env.step(action)
                steps += 1
                print(".", end="", flush=True)

            print(f" done ({steps} steps)")

            rewards.append(env.get_reward())
            # Compute CRR/SSI from final state
            total_c = len(env._conflicts)
            resolved_c = sum(1 for c in env._conflicts if c.get("resolved", False))
            crr = resolved_c / max(total_c, 1)
            ssi_vals = [a.satisfaction for a in env._actors.values()]
            ssi = sum(ssi_vals) / len(ssi_vals) if ssi_vals else 0.5

            crrs.append(crr)
            ssis.append(ssi)
            steps_list.append(steps)

        results.append({
            "scenario": scenario,
            "avg_reward": round(sum(rewards) / len(rewards), 4),
            "avg_crr": round(sum(crrs) / len(crrs), 4),
            "avg_ssi": round(sum(ssis) / len(ssis), 4),
            "avg_steps": round(sum(steps_list) / len(steps_list), 1),
        })

    return results


# ---------------------------------------------------------------------------
#  Phase 4: Head-to-Head Comparison
# ---------------------------------------------------------------------------

def print_comparison(rl_results, llm_results):
    """Print a side-by-side comparison table."""

    print("\n" + "=" * 70)
    print("  HEAD-TO-HEAD: RL (PPO) vs LLM (Gemini Flash)")
    print("=" * 70)

    header = f"  {'Scenario':<25} {'':>5} {'Reward':>8} {'CRR':>6} {'SSI':>6} {'Steps':>6}"
    print(header)
    print(f"  {'-'*62}")

    for rl in rl_results:
        scenario = rl["scenario"]
        llm = next((l for l in llm_results if l["scenario"] == scenario), None)

        print(f"  {scenario:<25} {'[RL]':>5} {rl['avg_reward']:>8.4f} "
              f"{rl['avg_crr']:>6.2f} {rl['avg_ssi']:>6.2f} {rl['avg_steps']:>6.1f}")
        if llm:
            print(f"  {'':25} {'[LLM]':>5} {llm['avg_reward']:>8.4f} "
                  f"{llm['avg_crr']:>6.2f} {llm['avg_ssi']:>6.2f} {llm['avg_steps']:>6.1f}")
        print()

    # Overall comparison
    if rl_results and llm_results:
        rl_avg_r = sum(r["avg_reward"] for r in rl_results) / len(rl_results)
        rl_avg_crr = sum(r["avg_crr"] for r in rl_results) / len(rl_results)
        rl_avg_ssi = sum(r["avg_ssi"] for r in rl_results) / len(rl_results)

        llm_avg_r = sum(r["avg_reward"] for r in llm_results) / len(llm_results)
        llm_avg_crr = sum(r["avg_crr"] for r in llm_results) / len(llm_results)
        llm_avg_ssi = sum(r["avg_ssi"] for r in llm_results) / len(llm_results)

        print(f"  {'OVERALL':<25} {'[RL]':>5} {rl_avg_r:>8.4f} "
              f"{rl_avg_crr:>6.2f} {rl_avg_ssi:>6.2f}")
        print(f"  {'':25} {'[LLM]':>5} {llm_avg_r:>8.4f} "
              f"{llm_avg_crr:>6.2f} {llm_avg_ssi:>6.2f}")

        print(f"\n  {'='*62}")
        if llm_avg_crr > rl_avg_crr:
            print(f"  WINNER (CRR): LLM Agent ({llm_avg_crr:.2f} vs {rl_avg_crr:.2f})")
        else:
            print(f"  WINNER (CRR): RL Agent ({rl_avg_crr:.2f} vs {llm_avg_crr:.2f})")

        if rl_avg_ssi > llm_avg_ssi:
            print(f"  WINNER (SSI): RL Agent ({rl_avg_ssi:.2f} vs {llm_avg_ssi:.2f})")
        else:
            print(f"  WINNER (SSI): LLM Agent ({llm_avg_ssi:.2f} vs {rl_avg_ssi:.2f})")


# ---------------------------------------------------------------------------
#  Phase 5: Schema Drift Stress Test (Both Agents)
# ---------------------------------------------------------------------------

def drift_stress_test(model):
    """Test how both agents degrade across schema drift versions."""

    print("\n" + "=" * 70)
    print("  SCHEMA DRIFT STRESS TEST")
    print("=" * 70)

    api_key = os.getenv("GEMINI_API_KEY", "")
    has_llm = bool(api_key)
    llm_agent = GeminiAgent(api_key=api_key) if has_llm else None

    drift_configs = [
        ("V1 (Baseline)", 0),
        ("V2 (Mild Drift)", 50),
        ("V3 (Heavy Drift)", 100),
    ]

    header = f"  {'Drift Version':<20} {'Agent':>6} {'Reward':>8} {'CRR':>6} {'SSI':>6}"
    print(header)
    print(f"  {'-'*48}")

    for label, episode_offset in drift_configs:
        # RL Agent
        env = ConflictGymWrapper()
        env.env._episode_count = episode_offset
        obs, _ = env.reset(options={"scenario": "morning_crunch_easy"})
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        rl_reward = env.env.get_reward()
        rl_crr = info.get("crr", 0)
        rl_ssi = info.get("ssi", 0.5)
        print(f"  {label:<20} {'[RL]':>6} {rl_reward:>8.4f} {rl_crr:>6.2f} {rl_ssi:>6.2f}")

        # LLM Agent
        if llm_agent:
            raw_env = ConflictEnv()
            raw_env._episode_count = episode_offset
            obs_raw = raw_env.reset("morning_crunch_easy")
            llm_agent.reset()
            steps = 0
            while not obs_raw.done and steps < obs_raw.max_steps:
                action_obj = llm_agent.act(obs_raw)
                obs_raw = raw_env.step(action_obj)
                steps += 1

            llm_reward = raw_env.get_reward()
            total_c = len(raw_env._conflicts)
            resolved_c = sum(1 for c in raw_env._conflicts if c.get("resolved"))
            llm_crr = resolved_c / max(total_c, 1)
            ssi_vals = [a.satisfaction for a in raw_env._actors.values()]
            llm_ssi = sum(ssi_vals) / len(ssi_vals) if ssi_vals else 0.5
            print(f"  {'':20} {'[LLM]':>6} {llm_reward:>8.4f} {llm_crr:>6.2f} {llm_ssi:>6.2f}")

        print()

    print("  If RL degrades on V3 but LLM doesn't, this proves the value")
    print("  of reasoning-based agents under schema drift.")


# ---------------------------------------------------------------------------
#  Save Results
# ---------------------------------------------------------------------------

def save_results(rl_results, llm_results):
    """Save all results to JSON."""
    os.makedirs("results", exist_ok=True)
    output = {
        "timestamp": datetime.now().isoformat(),
        "rl_agent": {"model": "PPO (MLP 128x128)", "results": rl_results},
        "llm_agent": {"model": "Gemini 2.0 Flash", "results": llm_results},
    }
    path = "results/battle_results.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {path}")


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ConflictEnv: Battle of the Agents")
    parser.add_argument("--timesteps", type=int, default=10000,
                        help="RL training timesteps (default: 10000)")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip RL training; evaluate saved model + LLM")
    parser.add_argument("--llm-only", action="store_true",
                        help="Only run LLM agent evaluation")
    parser.add_argument("--model", type=str, default="conflict_ppo",
                        help="RL model path (default: conflict_ppo)")
    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("  ConflictEnv: Battle of the Agents (RL vs LLM)")
    print("#" * 70)

    scenarios = [
        "morning_crunch_easy",
        "travel_chaos_medium",
        "monday_from_hell_hard",
    ]

    # Phase 1: RL Training
    if not args.llm_only:
        if args.eval_only:
            model = PPO.load(args.model)
            print(f"  Loaded RL model: {args.model}.zip")
        else:
            model = train_rl(timesteps=args.timesteps, model_path=args.model)

    # Phase 2: RL Evaluation
    rl_results = []
    if not args.llm_only:
        print("\n" + "=" * 60)
        print("  PHASE 2: RL EVALUATION")
        print("=" * 60)
        rl_results = evaluate_rl(model, scenarios)
        for r in rl_results:
            print(f"  {r['scenario']:<30} R={r['avg_reward']:.4f} "
                  f"CRR={r['avg_crr']:.2f} SSI={r['avg_ssi']:.2f}")

    # Phase 3: LLM Evaluation
    print("\n" + "=" * 60)
    print("  PHASE 3: LLM EVALUATION (Gemini Flash)")
    print("=" * 60)
    llm_results = evaluate_llm(scenarios, n_episodes=1)
    for r in llm_results:
        print(f"  {r['scenario']:<30} R={r['avg_reward']:.4f} "
              f"CRR={r['avg_crr']:.2f} SSI={r['avg_ssi']:.2f}")

    # Phase 4: Comparison
    if rl_results and llm_results:
        print_comparison(rl_results, llm_results)

    # Phase 5: Drift Stress Test
    if not args.llm_only:
        drift_stress_test(model)

    # Save
    save_results(rl_results, llm_results)

    print("\n" + "#" * 70)
    print("  Battle Complete!")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
