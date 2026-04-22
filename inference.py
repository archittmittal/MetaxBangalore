"""
ConflictEnv -- Inference & Demo Runner
=======================================
Runs the "Monday from Hell" demo scenario and generates
before/after comparison output suitable for the hackathon pitch.

Usage:
  python inference.py                    # Run demo with default (no trained model)
  python inference.py --all              # Run all archetypes x difficulties
  python inference.py --drift-demo       # Show schema drift progression
"""

from __future__ import annotations

import argparse
import json
import sys

from env import ConflictEnv
from models import ConflictAction
from reward import compute_crr, compute_ssi


def separator(title: str, char: str = "=") -> None:
    print(f"\n{char*70}")
    print(f"  {title}")
    print(f"{char*70}\n")


# ---------------------------------------------------------------------------
#  Scripted "Before Training" agent (random/naive)
# ---------------------------------------------------------------------------

def naive_agent_step(obs, step: int) -> ConflictAction:
    """
    Simulates an untrained agent making poor decisions.
    Used for the "before" comparison in the demo.
    """
    conflicts = obs.active_conflicts

    if step == 0:
        # Immediately tries to escalate on first step
        return ConflictAction(command="escalate", parameters={"conflict_id": "c1"})

    if step == 1:
        # Sends a blunt message (bad tone for spouse)
        return ConflictAction(command="draft_message", parameters={
            "actor_id": "spouse", "tone": "blunt",
            "content": "Can't make it. Reschedule."
        })

    # Fall back to resolve without doing anything
    return ConflictAction(command="resolve", parameters={})


# ---------------------------------------------------------------------------
#  Scripted "After Training" agent (smart/optimal)
# ---------------------------------------------------------------------------

def smart_agent_step(obs, step: int) -> ConflictAction:
    """
    Simulates a well-trained agent making optimal decisions.
    Used for the "after" comparison in the demo.
    """
    script = [
        # Step 0: Query boss to understand constraints
        ConflictAction(command="query_preference", parameters={"actor_id": "boss"}),
        # Step 1: Reschedule client demo to afternoon (resolves c1)
        ConflictAction(command="reschedule", parameters={"event_id": "evt_demo", "new_slot": "15:00"}),
        # Step 2: Warm message to spouse (protects satisfaction)
        ConflictAction(command="draft_message", parameters={
            "actor_id": "spouse", "tone": "warm",
            "content": "Anniversary dinner is my #1 priority tonight. I'll be there by 7:30."
        }),
        # Step 3: Confirm the board call (meets hard deadline)
        ConflictAction(command="confirm", parameters={"event_id": "evt_board"}),
        # Step 4: Reschedule doctor to morning gap
        ConflictAction(command="reschedule", parameters={"event_id": "evt_doctor", "new_slot": "11:00"}),
        # Step 5: Cancel low-priority drinks (friend is flexible)
        ConflictAction(command="cancel", parameters={"event_id": "evt_drinks"}),
        # Step 6: Friendly message to friend about cancellation
        ConflictAction(command="draft_message", parameters={
            "actor_id": "friend", "tone": "friendly",
            "content": "Hey Arjun, rain check on drinks? Crazy day. Let's do Thursday!"
        }),
        # Step 7: Confirm dinner (lock it in)
        ConflictAction(command="confirm", parameters={"event_id": "evt_dinner"}),
        # Step 8: Resolve
        ConflictAction(command="resolve", parameters={}),
    ]

    if step < len(script):
        return script[step]
    return ConflictAction(command="resolve", parameters={})


# ---------------------------------------------------------------------------
#  Demo: Before vs After
# ---------------------------------------------------------------------------

def run_before_after_demo():
    """
    The main demo for the hackathon pitch.
    Shows naive agent (before training) vs smart agent (after training)
    on the "Monday from Hell" scenario.
    """
    separator("CONFLICTENV DEMO: MONDAY FROM HELL")
    print("  Scenario: Boss hijacked 9AM for board call, client demo displaced,")
    print("  doctor and vendor overlap, flight is tight, and tonight is your")
    print("  anniversary dinner. What does the agent do?")
    print()

    # --- BEFORE training ---
    separator("[X] BEFORE TRAINING (Naive Agent)", "-")
    env_before = ConflictEnv()
    obs = env_before.reset("monday_from_hell_hard")
    print(f"  Conflicts: {len(obs.active_conflicts)}")
    print(f"  Hard deadlines: {obs.max_steps} max steps")
    print()

    for step in range(5):
        action = naive_agent_step(obs, step)
        print(f"  Step {step}: {action.command} {action.parameters}")
        obs = env_before.step(action)
        print(f"    -> {obs.feedback[:100]}")
        if obs.done:
            break

    before_reward = env_before.get_reward()
    print(f"\n  RESULT (Before):")
    print(f"     Reward:  {before_reward:.4f}")
    print(f"     Verdict: {'ESCALATED (failure)' if env_before._escalated else 'Poor resolution'}")

    # --- AFTER training ---
    separator("[OK] AFTER TRAINING (Smart Agent)", "-")
    env_after = ConflictEnv()
    obs = env_after.reset("monday_from_hell_hard")

    for step in range(10):
        action = smart_agent_step(obs, step)
        print(f"  Step {step}: {action.command} {action.parameters}")
        obs = env_after.step(action)
        print(f"    -> {obs.feedback[:100]}")
        if obs.done:
            break

    after_reward = env_after.get_reward()
    print(f"\n  RESULT (After):")
    print(f"     Reward:       {after_reward:.4f}")
    print(f"     Improvement:  {after_reward - before_reward:+.4f}")
    print(f"         ({((after_reward / max(before_reward, 0.01)) - 1) * 100:+.0f}% improvement)")

    # --- Summary ---
    separator("COMPARISON SUMMARY")
    print(f"  {'Metric':<25} {'Before':>10} {'After':>10} {'Delta':>10}")
    print(f"  {'-'*55}")
    print(f"  {'Reward':<25} {before_reward:>10.4f} {after_reward:>10.4f} {after_reward - before_reward:>+10.4f}")
    print()


# ---------------------------------------------------------------------------
#  Schema drift demo
# ---------------------------------------------------------------------------

def run_drift_demo():
    """Show how the same scenario looks under V1, V2, and V3 schemas."""
    separator("SCHEMA DRIFT DEMO")

    env = ConflictEnv()
    versions = [("V1 (Baseline)", 0), ("V2 (Mild Drift)", 50), ("V3 (Heavy Drift)", 100)]

    for label, episode_num in versions:
        env._episode_count = episode_num
        obs = env.reset("morning_crunch_easy")

        print(f"  --- {label} (episode {episode_num}) ---")
        print(f"  Schema: {obs.schema_version}")

        events = obs.calendar.get("events", [])
        if events:
            first = events[0]
            print(f"  First event keys: {list(first.keys())}")
            # Show the date format
            for key in ["start_time", "startTime", "schedule"]:
                if key in first:
                    print(f"  Date ({key}): {first[key]}")
                    break

        print(f"  Policy keys: {list(obs.policy_rules.keys())}")
        if obs.actor_profiles:
            first_actor = list(obs.actor_profiles.values())[0]
            for key in ["preferred_times", "availability", "scheduling_prefs"]:
                if key in first_actor:
                    print(f"  Actor prefs ({key}): {first_actor[key]}")
                    break
        print()


# ---------------------------------------------------------------------------
#  All scenarios runner
# ---------------------------------------------------------------------------

def run_all_scenarios():
    """Run all archetype x difficulty combinations."""
    separator("ALL SCENARIOS")

    env = ConflictEnv()
    archetypes = ["morning_crunch", "travel_chaos", "monday_from_hell", "deadline_squeeze", "social_minefield"]
    difficulties = ["easy", "medium", "hard"]

    print(f"  {'Scenario':<25} {'Diff':<8} {'Events':>7} {'Conflicts':>10} {'Reward':>8}")
    print(f"  {'-'*60}")

    for arch in archetypes:
        for diff in difficulties:
            task = f"{arch}_{diff}"
            obs = env.reset(task)

            # Take a few basic steps
            env.step(ConflictAction(command="query_preference", parameters={"actor_id": "boss"}))
            env.step(ConflictAction(command="resolve", parameters={}))

            reward = env.get_reward()
            n_events = len(obs.calendar.get("events", []))
            n_conflicts = len(obs.active_conflicts)

            print(f"  {arch:<25} {diff:<8} {n_events:>7} {n_conflicts:>10} {reward:>8.4f}")

    print()


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ConflictEnv Demo Runner")
    parser.add_argument("--all", action="store_true", help="Run all scenarios")
    parser.add_argument("--drift-demo", action="store_true", help="Show schema drift progression")
    args = parser.parse_args()

    if args.drift_demo:
        run_drift_demo()
    elif args.all:
        run_all_scenarios()
    else:
        run_before_after_demo()

    if not args.drift_demo:
        run_drift_demo()

    print("\nDemo complete.\n")


if __name__ == "__main__":
    main()
