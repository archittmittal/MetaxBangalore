import json
import random
import os
import sys

# Ensure local 'openenv' is found
sys.path.insert(0, os.path.abspath("."))

from conflict_env.env import ConflictEnv

def generate_expert_response(obs):
    """
    Generates a high-quality expert reasoning and action for a given observation.
    FOCUS: RL-centric reasoning, evaluating rewards (CRR/SSI) and policy constraints.
    """
    scenario_name = obs.scenario_name
    conflicts = obs.active_conflicts
    events = obs.calendar.get("events", [])
    actor_profiles = obs.actor_profiles
    
    # Identify critical events
    critical_events = [e for e in events if e.get("priority") in ["critical", "high"]]
    unmovable_events = [e for e in events if e.get("is_unmovable", False)]
    
    if not conflicts:
        return "<thought>\nAll conflicts appear to be resolved based on the current calendar state. Finalizing the schedule will maximize the Conflict Resolution Rate (CRR) to 1.0.\n</thought>\n{\"command\": \"resolve\", \"parameters\": {}}"

    target_conflict = conflicts[0]
    event_ids = target_conflict.get("event_ids", [])
    affected_events = [e for e in events if e.get("event_id") in event_ids]
    
    # RL Logic: Evaluate multiple candidate actions
    movable = [e for e in affected_events if not e.get("is_unmovable", False)]
    
    # Reasoning block starts
    thought = f"### Conflict Analysis: {target_conflict.get('description')}\n"
    thought += f"I am evaluating the best action for the '{scenario_name}' scenario to optimize the Group Relative Reward.\n\n"
    
    if not movable:
        thought += "Critical Constraint: Both events in the current conflict are flagged as 'is_unmovable' in the V1 schema. "
        thought += "Attempting to reschedule would trigger a Policy Error and yield a -0.05 reward penalty. "
        thought += "I must escalate to a human to preserve the Social Satisfaction Index (SSI)."
        return f"<thought>\n{thought}\n</thought>\n{{\"command\": \"escalate\", \"parameters\": {{}}}}"

    # Evaluate options
    thought += "### Policy & Reward Evaluation:\n"
    options = []
    for ev in movable:
        # Check priority
        priority_val = {"low": 0, "normal": 1, "medium": 2, "high": 3, "critical": 4}.get(ev.get("priority"), 1)
        # Check rebooking policy
        resched_count = ev.get("reschedule_count", 0)
        max_resched = obs.policy_rules.get("max_reschedules_per_event", 3)
        
        score = (4 - priority_val) * 10 + (max_resched - resched_count) * 5
        options.append((ev, score))
    
    # Sort by score (higher is better to move)
    options.sort(key=lambda x: x[1], reverse=True)
    to_move, best_score = options[0]
    
    thought += f"1. Candidate: Reschedule '{to_move.get('title')}'. Priority: {to_move.get('priority')}. Status: {to_move.get('status')}.\n"
    thought += f"2. Expected SSI Impact: Moving a {to_move.get('priority')} priority event minimizes the satisfaction delta penalty.\n"
    
    # Find a slot
    all_slots = [f"{h:02d}:00" for h in range(7, 22)]
    occupied_slots = []
    for e in events:
        if e.get("status") not in ["cancelled"]:
            t = e.get("start_time", "").split(" ")[-1]
            occupied_slots.append(t)
            
    available_slots = [s for s in all_slots if s not in occupied_slots]
    
    # Actor preference
    actor_id = to_move.get("actor_ids", [None])[0]
    actor_profile = actor_profiles.get(actor_id, {})
    preferred = actor_profile.get("preferred_times", [])
    
    best_slot = available_slots[0] if available_slots else "12:00"
    slot_reason = "the first available slot"
    for s in preferred:
        if s in available_slots:
            best_slot = s
            slot_reason = f"the actor's preferred time {s}"
            break

    thought += f"### Decision:\nI will reschedule '{to_move.get('title')}' to {best_slot} ({slot_reason}).\n"
    thought += f"This action is optimal because:\n"
    thought += f"- It resolves the '{target_conflict.get('type')}' conflict (increasing CRR).\n"
    thought += f"- It respects the 'max_reschedules' policy ({to_move.get('reschedule_count')}/{obs.policy_rules.get('max_reschedules_per_event', 3)}).\n"
    thought += f"- It uses a preferred time slot, mitigating the negative SSI delta associated with scheduling changes."

    action = {
        "command": "reschedule",
        "parameters": {
            "event_id": to_move.get("event_id"),
            "new_slot": best_slot
        }
    }
    
    return f"<thought>\n{thought}\n</thought>\n{json.dumps(action)}"

def generate_super_dataset(num_samples=1000):
    env = ConflictEnv()
    dataset = []
    
    difficulties = ["easy", "medium", "hard"]
    archetypes = ["morning_crunch", "travel_chaos", "monday_from_hell", "deadline_squeeze", "social_minefield"]
    
    print(f"Generating {num_samples} RL-focused expert examples...")
    
    for i in range(num_samples):
        # Diversity in scenarios
        diff = difficulties[i % 3]
        arch = archetypes[i % 5]
        
        obs = env.reset(task_name=f"{arch}_{diff}")
        
        prompt = (
            "You are an Elite Executive Assistant trained with GRPO. Your goal is to resolve scheduling conflicts "
            "optimizing for both Conflict Resolution Rate (CRR) and Social Satisfaction Index (SSI).\n\n"
            f"### Scenario: {obs.scenario_name}\n"
            f"### Context: {obs.feedback}\n\n"
            "### Observation State:\n"
            f"- Active Conflicts: {len(obs.active_conflicts)}\n"
            f"- Schema Version: {obs.schema_version}\n"
            f"- Policy Rules: {json.dumps(obs.policy_rules)}\n"
            f"- Calendar Events: {json.dumps(obs.calendar.get('events', []))}\n"
            f"- Actor Profiles: {json.dumps(obs.actor_profiles)}\n\n"
            "Respond in the following format:\n"
            "<thought>\n[Policy Analysis & Reward Optimization Reasoning]\n</thought>\n"
            "{\"command\": \"...\", \"parameters\": {...}}"
        )
        
        completion = generate_expert_response(obs)
        
        # Use simple text format for fine-tuning
        dataset.append({
            "text": f"### Instruction: {prompt}\n\n### Response: {completion}"
        })

        if (i+1) % 100 == 0:
            print(f"Generated {i+1}/{num_samples}...")

    with open("train.jsonl", "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")
    
    print(f"Successfully saved {len(dataset)} examples to train.jsonl")

if __name__ == "__main__":
    generate_super_dataset(1000)
