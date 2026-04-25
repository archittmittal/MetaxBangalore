import os
import json
import re
import subprocess
import torch
import requests
from datasets import Dataset

# 1. Install Dependencies (Kaggle Specific)
def install_deps():
    print("[Kaggle] Installing Elite RL Stack...")
    commands = [
        "pip install --upgrade pip",
        "pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"",
        "pip install --no-deps \"trl<0.13.0\" peft accelerate bitsandbytes",
        "pip install git+https://github.com/archittmittal/MetaxBangalore.git",
        "pip install python-dotenv gymnasium stable-baselines3"
    ]
    for cmd in commands:
        try:
            subprocess.run(cmd.split(), check=True)
        except Exception as e:
            print(f"Error installing {cmd}: {e}")

try:
    from unsloth import FastLanguageModel, PatchFastRL
    from trl import GRPOTrainer, GRPOConfig
except ImportError:
    install_deps()
    from unsloth import FastLanguageModel, PatchFastRL
    from trl import GRPOTrainer, GRPOConfig

# Patch for RL efficiency
PatchFastRL("GRPO", "Unsloth")

# 2. Setup & Auth
hf_token = os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = hf_token

# 3. Model & Tokenizer Configuration
model_id = "Qwen/Qwen2.5-1.5B-Instruct"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id,
    max_seq_length = 2048,
    load_in_4bit = True,
)
model = FastLanguageModel.get_peft_model(
    model, r = 16, lora_alpha = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# 4. Data Preparation (ChatML Optimized)
SYSTEM_PROMPT = "You are a scheduling AI. You MUST: 1. Start with <thought> 2. End with JSON. Commands: reschedule, draft_message, cancel, query_preference, escalate, confirm, resolve."

def map_dataset(example):
    prompt_text = example.get("scenario", example.get("prompt", "Default Scenario"))
    return {
        "prompt": tokenizer.apply_chat_template([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text}
        ], tokenize=False, add_generation_prompt=True)
    }

dataset_url = "https://raw.githubusercontent.com/archittmittal/MetaxBangalore/main/training_prompts.json"
data = requests.get(dataset_url).json()
train_dataset = Dataset.from_list(data).map(map_dataset).shuffle(seed=42).select(range(min(len(data), 800)))

# 5. Reward Functions (The V3.1 Jackpot System)
def reward_format_check(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        reward = 0.0
        if "<thought>" in completion: reward += 0.4
        if "{" in completion: reward += 0.3 
        if "</thought>" in completion and "}" in completion: reward += 0.3
        rewards.append(reward)
    return rewards

def reward_conflict_resolution(prompts, completions, **kwargs):
    from conflict_env.env import ConflictEnv
    from conflict_env.models import ConflictAction
    rewards = []
    env = ConflictEnv()
    for completion in completions:
        env.reset(task_name="auto")
        try:
            json_match = re.search(r'\{.*\}', completion, re.DOTALL)
            if json_match:
                # Validation and environment step
                action_data = json.loads(json_match.group(0))
                env.step(ConflictAction(**action_data))
                rewards.append(env.get_reward())
            else:
                rewards.append(0.05) # Small reward for trying
        except:
            rewards.append(0.0)
    return rewards

# 6. Training Execution (Milestone Final)
training_args = GRPOConfig(
    output_dir = "/kaggle/working/conflict-env-final-grpo",
    learning_rate = 2e-5,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    num_generations = 4, 
    max_completion_length = 350,
    num_train_epochs = 1,
    max_steps = 150,
    logging_steps = 1,
    temperature = 0.9,
    push_to_hub = True,
    hub_model_id = "purvansh01/conflict-env-final", # PRODUCTION MODEL
    hub_token = hf_token,
)

trainer = GRPOTrainer(
    model = model,
    reward_funcs = [reward_format_check, reward_conflict_resolution],
    args = training_args,
    train_dataset = train_dataset,
)

if __name__ == "__main__":
    print("🚀 Starting Final ConflictEnv Training Sprint...")
    trainer.train()
