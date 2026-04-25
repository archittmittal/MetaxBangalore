"""
ConflictEnv -- Training Data Generator
======================================
Generates a dataset of scheduling conflict prompts (V1-V3) 
to be used for GRPO reasoning training.
"""

import json
from conflict_env.env import ConflictEnv

def generate_prompts(num_samples=100):
    env = ConflictEnv()
    dataset = []
    
    difficulties = ["easy", "medium", "hard"]
    
    print(f"Generating {num_samples} prompts for training...")
    
    for i in range(num_samples):
        # Rotate through difficulties and archetypes
        diff = difficulties[i % 3]
        obs = env.reset(task_name=diff)
        
        # Construct the prompt for the reasoning model
        prompt = (
            "You are an Elite Executive Assistant. Your goal is to resolve scheduling conflicts "
            "in the following calendar. You must think deeply about social satisfaction and "
            "hard deadlines before taking an action.\n\n"
            f"### Scenario: {obs.scenario_name}\n"
            f"### Context: {obs.feedback}\n\n"
            "### Current State:\n"
            f"- Active Conflicts: {len(obs.active_conflicts)}\n"
            f"- Schema Version: {obs.schema_version}\n\n"
            "Respond in the following format:\n"
            "<thought>\n[Your step-by-step reasoning here]\n</thought>\n"
            "{\"command\": \"...\", \"parameters\": {...}}"
        )
        
        dataset.append({
            "prompt": prompt,
            "difficulty": diff,
            "scenario": obs.scenario_name
        })

    with open("training_prompts.json", "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Successfully saved {len(dataset)} prompts to training_prompts.json")

if __name__ == "__main__":
    generate_prompts(5000)
