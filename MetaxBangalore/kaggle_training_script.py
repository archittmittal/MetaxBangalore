import os
import json
import re
import subprocess
import torch
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
        subprocess.run(cmd.split(), check=True)

try:
    from unsloth import FastLanguageModel, PatchFastRL
    from trl import GRPOTrainer, GRPOConfig
except ImportError:
    install_deps()
    from unsloth import FastLanguageModel, PatchFastRL
    from trl import GRPOTrainer, GRPOConfig

# Patch for RL efficiency
PatchFastRL("GRPO", "Unsloth")

# 2. Model & Tokenizer Configuration
model_id = "Qwen/Qwen2.5-1.5B-Instruct"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_id,
    max_seq_length = 2048,
    load_in_4bit = True,
    fast_inference = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
)

# 3. Data Preparation (ChatML Optimized)
SYSTEM_PROMPT = """You are an Elite Executive Assistant. Resolve scheduling conflicts using deep reasoning.
Follow this format EXACTLY:
<thought>
Reasoning about social satisfaction and hard deadlines here.
</thought>
{JSON command}"""

def apply_chatml_formatting(example):
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Scenario: {example['scenario']}\nDifficulty: {example['difficulty']}"}
    ]
    # We leave the assistant part open for the model to generate <thought>
    return {"prompt": tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) + "<thought>\n"}

import requests
dataset_url = "https://raw.githubusercontent.com/archittmittal/MetaxBangalore/main/training_prompts.json"
data = requests.get(dataset_url).json()
raw_dataset = Dataset.from_list(data)
train_dataset = raw_dataset.map(apply_chatml_formatting)

# 4. Reward Functions
def reward_format_check(completions, **kwargs):
    responses = [c[0]["content"] for c in completions]
    rewards = []
    for response in responses:
        score = 0.0
        if "</thought>" in response: score += 15.0
        try:
            thought_end = response.rfind("</thought>")
            json_part = response[thought_end + 10:].strip()
            if json_part.startswith("{") and json_part.endswith("}"):
                json.loads(json_part)
                score += 15.0
        except: pass
        rewards.append(score)
    return rewards

def reward_on_topic(completions, **kwargs):
    responses = [c[0]["content"] for c in completions]
    keywords = ["prioritize", "conflict", "meeting", "reschedule", "flight", "spouse"]
    return [sum(1.0 for k in keywords if k in r.lower()) for r in responses]

# 5. Training Execution
training_args = GRPOConfig(
    output_dir = "output",
    learning_rate = 5e-6,
    lr_scheduler_type = "cosine",
    logging_steps = 1,
    max_steps = 200, # Milestone 1
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    num_generations = 4,
    max_prompt_length = 512,
    max_completion_length = 512,
    push_to_hub = True,
    hub_model_id = "purvansh01/conflict-env-qwen-grpo-v2",
    save_steps = 100,
)

trainer = GRPOTrainer(
    model = model,
    reward_funcs = [reward_format_check, reward_on_topic],
    args = training_args,
    train_dataset = train_dataset,
)

print("🚀 Starting Production Training Loop...")
trainer.train()
