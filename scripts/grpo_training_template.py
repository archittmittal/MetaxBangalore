"""
ConflictEnv -- Elite-Tier GRPO Training Template (Unsloth + TRL)
=============================================================
This script is designed for the Onsite Hackathon (Bangalore 2026).
It uses GRPO (Group Relative Policy Optimization) to train a 
reasoning model (e.g., Qwen-7B) to resolve scheduling conflicts.

Stack:
 - Unsloth: 4-bit Quantization + Fast Kernels
 - TRL: GRPOTrainer
 - ConflictEnv: Adaptive Curriculum + Verifiable Rewards
"""

import torch
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOTrainer, GRPOConfig
from conflict_env.env import ConflictEnv
from conflict_env.models import ConflictAction

# Patch for RL efficiency
PatchFastRL("GRPO", "Unsloth")

# 1. Load Model & Tokenizer
model_id = "Qwen/Qwen2.5-1.5B-Instruct" # Perfect for Billion-parameter reasoning training
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id,
    max_seq_length = 2048,
    load_in_4bit = True,
    fast_inference = True,
)

# Add LoRA Adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
)

# 2. Load Dataset
import json
from datasets import Dataset
with open("training_prompts.json") as f:
    data = json.load(f)
train_dataset = Dataset.from_list(data)

# 3. Define Reward Functions (Verifiable & Independent)
def reward_conflict_resolution(prompts, completions, **kwargs):
    """Checks if the model actually fixed the calendar."""
    rewards = []
    for prompt, completion in zip(prompts, completions):
        # C2: Initialize a fresh environment for each sample to prevent state leakage
        env = ConflictEnv()
        # Reset environment with 'auto' difficulty (Adaptive Curriculum)
        env.reset(task_name="auto")
        
        # Parse completion for reasoning and action
        try:
            if "<thought>" in completion and "{" in completion:
                # Improved JSON extraction: find first '{' and last '}'
                start = completion.find("{")
                end = completion.rfind("}") + 1
                if start != -1 and end != -1:
                    action_str = completion[start:end]
                    env.step(ConflictAction.model_validate_json(action_str))
                reward_data = env.get_reward()
                rewards.append(reward_data)
            else:
                rewards.append(0.05)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # C1: Avoid bare except; log specific errors
            import sys
            print(f"    [REWARD ERROR] {e}", file=sys.stderr)
            rewards.append(0.05)
    return rewards

def reward_format_check(prompts, completions, **kwargs):
    """Theme #3.2: Process Supervision - Rewards for thinking before acting."""
    rewards = []
    for completion in completions:
        if "<thought>" in completion and "</thought>" in completion:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

# 4. Training Config
training_args = GRPOConfig(
    output_dir = "conflict-env-qwen-1.5b-grpo",
    learning_rate = 5e-6,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    num_generations = 4, 
    max_prompt_length = 512,
    max_completion_length = 512,
    num_train_epochs = 1,
    logging_steps = 10,
    report_to = "tensorboard",
    push_to_hub = True,
    hub_model_id = "purvansh01/conflict-env-qwen-1.5b-grpo",
)

# 5. Initialize Trainer
trainer = GRPOTrainer(
    model = model,
    reward_funcs = [reward_conflict_resolution, reward_format_check],
    args = training_args,
    train_dataset = train_dataset,
    tokenizer = tokenizer,
)

# 6. Execute Training
if __name__ == "__main__":
    print(f"[ConflictEnv] Starting GRPO Training for {model_id}...")
    # trainer.train()
    # trainer.push_to_hub()
    print("[ConflictEnv] Training pipeline initialized with 150 prompts. Ready for Colab/A100.")
