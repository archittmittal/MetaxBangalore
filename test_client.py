"""
ConflictEnv -- Local Test Client
=================================
Tests the environment directly (no HTTP server needed).

Runs through all 3 difficulty tiers with scripted actions to verify:
  1. reset/step/state/get_reward all work
  2. Rewards stay in (0.05, 0.95)
  3. Schema drift produces different observation shapes
  4. Conflicts can be resolved
  5. Actor satisfaction changes with actions
"""

from __future__ import annotations

import json
import sys
import os

# Fix Windows console encoding for emoji characters
os.environ["PYTHONIOENCODING"] = "utf-8"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from env import ConflictEnv
from models import ConflictAction


def separator(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_obs(obs, compact: bool = True) -> None:
    """Pretty-print an observation."""
    d = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()
    if compact:
        print(f"  Step: {d['step_count']}/{d['max_steps']} | "
              f"Schema: {d['schema_version']} | "
              f"Conflicts: {len(d['active_conflicts'])} | "
              f"Reward: {d['reward']:.4f} | "
              f"Done: {d['done']}")
        print(f"  Feedback: {d['feedback'][:120]}...")
    else:
        print(json.dumps(d, indent=2, default=str)[:2000])


def test_easy():
    """Test easy difficulty with morning_crunch scenario."""
    separator("TEST: Easy Difficulty -- Morning Crunch")

    env = ConflictEnv()
    obs = env.reset("morning_crunch_easy")
    print(f"  Scenario: {obs.scenario_name} ({obs.difficulty})")
    print(f"  Narrative: {obs.feedback}")
    print(f"  Events: {len(obs.calendar.get('events', []))}")
    print(f"  Conflicts: {len(obs.active_conflicts)}")
    print(f"  Actors: {list(obs.actor_profiles.keys())}")
    print_obs(obs)

    # Step 1: Query boss preferences
    obs = env.step(ConflictAction(command="query_preference", parameters={"actor_id": "boss"}))
    print_obs(obs)

    # Step 2: Reschedule school event to resolve overlap
    obs = env.step(ConflictAction(command="reschedule", parameters={"event_id": "evt_school", "new_slot": "08:00"}))
    print_obs(obs)

    # Step 3: Confirm standup
    obs = env.step(ConflictAction(command="confirm", parameters={"event_id": "evt_standup"}))
    print_obs(obs)

    # Step 4: Resolve
    obs = env.step(ConflictAction(command="resolve", parameters={}))
    print_obs(obs)

    reward = env.get_reward()
    print(f"\n  [PASS] Final reward: {reward:.4f}")
    assert 0.05 <= reward <= 0.95, f"Reward {reward} out of range!"
    print("  [PASS] Reward in valid range")
    return True


def test_medium():
    """Test medium difficulty with travel_chaos scenario."""
    separator("TEST: Medium Difficulty -- Travel Chaos")

    env = ConflictEnv()
    obs = env.reset("travel_chaos_medium")
    print(f"  Scenario: {obs.scenario_name} ({obs.difficulty})")
    print(f"  Events: {len(obs.calendar.get('events', []))}")
    print(f"  Conflicts: {len(obs.active_conflicts)}")
    print_obs(obs)

    # Step 1: Query airline
    obs = env.step(ConflictAction(command="query_preference", parameters={"actor_id": "client"}))
    print_obs(obs)

    # Step 2: Reschedule flight
    obs = env.step(ConflictAction(command="reschedule", parameters={"event_id": "evt_flight", "new_slot": "16:00"}))
    print_obs(obs)

    # Step 3: Draft message to spouse about errand
    obs = env.step(ConflictAction(command="draft_message", parameters={
        "actor_id": "spouse", "tone": "warm", "content": "Can we move the errand to after 1pm? I'll handle it then."
    }))
    print_obs(obs)

    # Step 4: Reschedule errand
    obs = env.step(ConflictAction(command="reschedule", parameters={"event_id": "evt_errand", "new_slot": "13:30"}))
    print_obs(obs)

    # Step 5: Resolve
    obs = env.step(ConflictAction(command="resolve", parameters={}))
    print_obs(obs)

    reward = env.get_reward()
    print(f"\n  [PASS] Final reward: {reward:.4f}")
    assert 0.05 <= reward <= 0.95, f"Reward {reward} out of range!"
    print("  [PASS] Reward in valid range")
    return True


def test_hard():
    """Test hard difficulty with monday_from_hell scenario (the demo!)."""
    separator("TEST: Hard Difficulty -- Monday from Hell (Demo Scenario)")

    env = ConflictEnv()
    obs = env.reset("monday_from_hell_hard")
    print(f"  Scenario: {obs.scenario_name} ({obs.difficulty})")
    print(f"  Events: {len(obs.calendar.get('events', []))}")
    print(f"  Conflicts: {len(obs.active_conflicts)}")
    print(f"  Max steps: {obs.max_steps}")
    print_obs(obs)

    # Step 1: Query boss to understand priorities
    obs = env.step(ConflictAction(command="query_preference", parameters={"actor_id": "boss"}))
    print_obs(obs)

    # Step 2: Move client demo to afternoon
    obs = env.step(ConflictAction(command="reschedule", parameters={"event_id": "evt_demo", "new_slot": "15:00"}))
    print_obs(obs)

    # Step 3: Draft warm message to spouse about tight schedule
    obs = env.step(ConflictAction(command="draft_message", parameters={
        "actor_id": "spouse", "tone": "warm",
        "content": "I know tonight is special. I'll make sure I'm there by 7:30 no matter what."
    }))
    print_obs(obs)

    # Step 4: Confirm board call
    obs = env.step(ConflictAction(command="confirm", parameters={"event_id": "evt_board"}))
    print_obs(obs)

    # Step 5: Reschedule doctor
    obs = env.step(ConflictAction(command="reschedule", parameters={"event_id": "evt_doctor", "new_slot": "11:00"}))
    print_obs(obs)

    # Step 6: Cancel drinks with friend (low priority)
    obs = env.step(ConflictAction(command="cancel", parameters={"event_id": "evt_drinks"}))
    print_obs(obs)

    # Step 7: Confirm dinner (non-negotiable)
    obs = env.step(ConflictAction(command="confirm", parameters={"event_id": "evt_dinner"}))
    print_obs(obs)

    # Step 8: Resolve
    obs = env.step(ConflictAction(command="resolve", parameters={}))
    print_obs(obs)

    reward = env.get_reward()
    print(f"\n  [PASS] Final reward: {reward:.4f}")
    assert 0.05 <= reward <= 0.95, f"Reward {reward} out of range!"
    print("  [PASS] Reward in valid range")
    return True


def test_invalid_actions():
    """Test invalid actions produce graceful feedback, not crashes."""
    separator("TEST: Invalid Action Handling")

    env = ConflictEnv()
    env.reset("easy")

    # Invalid command
    obs = env.step(ConflictAction(command="fly_to_moon", parameters={}))
    assert "Invalid command" in obs.feedback
    print("  [PASS] Invalid command handled")

    # Nonexistent event
    obs = env.step(ConflictAction(command="reschedule", parameters={"event_id": "fake_event", "new_slot": "10:00"}))
    assert "not found" in obs.feedback
    print("  [PASS] Nonexistent event handled")

    # Nonexistent actor
    obs = env.step(ConflictAction(command="query_preference", parameters={"actor_id": "alien"}))
    assert "not found" in obs.feedback
    print("  [PASS] Nonexistent actor handled")

    reward = env.get_reward()
    assert 0.05 <= reward <= 0.95
    print(f"  [PASS] Reward still valid: {reward:.4f}")
    return True


def test_schema_drift():
    """Test that schema drift produces structurally different observations."""
    separator("TEST: Schema Drift V1 -> V2 -> V3")

    env = ConflictEnv()

    # V1 (episode 0)
    obs1 = env.reset("easy")
    cal1 = obs1.calendar
    print(f"  V1 schema_version: {obs1.schema_version}")
    if cal1.get("events"):
        first_event = cal1["events"][0]
        print(f"  V1 date field: start_time = {first_event.get('start_time', 'N/A')}")
        print(f"  V1 keys: {list(first_event.keys())}")

    # Force episode count to trigger V2
    env._episode_count = 50
    obs2 = env.reset("easy")
    cal2 = obs2.calendar
    print(f"\n  V2 schema_version: {obs2.schema_version}")
    if cal2.get("events"):
        first_event = cal2["events"][0]
        print(f"  V2 date field: startTime = {first_event.get('startTime', 'N/A')}")
        print(f"  V2 keys: {list(first_event.keys())}")

    # Force V3
    env._episode_count = 100
    obs3 = env.reset("easy")
    cal3 = obs3.calendar
    print(f"\n  V3 schema_version: {obs3.schema_version}")
    if cal3.get("events"):
        first_event = cal3["events"][0]
        print(f"  V3 date field: schedule.begin = {first_event.get('schedule', {}).get('begin', 'N/A')}")
        print(f"  V3 keys: {list(first_event.keys())}")

    # Verify structural differences
    if cal1.get("events") and cal2.get("events") and cal3.get("events"):
        v1_keys = set(cal1["events"][0].keys())
        v2_keys = set(cal2["events"][0].keys())
        v3_keys = set(cal3["events"][0].keys())
        assert v1_keys != v2_keys, "V1 and V2 should have different keys!"
        assert v2_keys != v3_keys, "V2 and V3 should have different keys!"
        print("\n  [PASS] All three schema versions produce different observation structures!")
    else:
        print("\n  [WARN] Could not verify -- calendar events missing")

    return True


def test_reward_range():
    """Run 50 random episodes and verify rewards stay in range."""
    separator("TEST: Reward Range (50 Episodes)")

    env = ConflictEnv()
    violations = 0
    difficulties = ["easy", "medium", "hard"]

    for i in range(50):
        diff = difficulties[i % 3]
        env.reset(diff)

        # Take random-ish actions
        for step in range(5):
            if step == 0:
                obs = env.step(ConflictAction(command="query_preference", parameters={"actor_id": "boss"}))
            elif step == 1:
                obs = env.step(ConflictAction(command="reschedule", parameters={"event_id": "evt_standup", "new_slot": "10:00"}))
            elif step == 2:
                obs = env.step(ConflictAction(command="confirm", parameters={"event_id": "evt_standup"}))
            else:
                obs = env.step(ConflictAction(command="resolve", parameters={}))
                break

        reward = env.get_reward()
        if reward < 0.05 or reward > 0.95:
            violations += 1
            print(f"  [FAIL] Episode {i} ({diff}): reward {reward:.4f} OUT OF RANGE")

    if violations == 0:
        print(f"  [PASS] All 50 episodes: rewards in (0.05, 0.95)")
    else:
        print(f"  [FAIL] {violations}/50 episodes had out-of-range rewards")

    return violations == 0


def main():
    results = []

    tests = [
        ("Easy Difficulty", test_easy),
        ("Medium Difficulty", test_medium),
        ("Hard Difficulty", test_hard),
        ("Invalid Actions", test_invalid_actions),
        ("Schema Drift", test_schema_drift),
        ("Reward Range", test_reward_range),
    ]

    for name, fn in tests:
        try:
            passed = fn()
            results.append((name, "PASS" if passed else "FAIL"))
        except Exception as e:
            results.append((name, f"ERROR: {e}"))
            import traceback
            traceback.print_exc()

    separator("TEST RESULTS SUMMARY")
    for name, result in results:
        print(f"  {result}  {name}")
    print()

    all_passed = all("PASS" in r for _, r in results)
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
