"""
ConflictEnv -- Kaggle Training Script (Dual T4 / P100)
======================================================
Optimized for Kaggle Kernels. 
Requires: Internet Enabled, HF_TOKEN in Kaggle Secrets.
"""

import os
import subprocess

# 1. Install Dependencies (Kaggle Specific)
def install_deps():
    print("[Kaggle] Installing Elite RL Stack (Unsloth, TRL, OpenEnv)...")
    commands = [
        "pip install --upgrade pip",
        "pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\"",
        "pip install --no-deps \"trl<0.13.0\" peft accelerate bitsandbytes",
        "pip install git+https://github.com/archittmittal/MetaxBangalore.git", # Install the environment package
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

from conflict_env.env import ConflictEnv
from conflict_env.models import ConflictAction
from datasets import Dataset

# Patch for RL efficiency
PatchFastRL("GRPO", "Unsloth")

# 2. Authenticate (using Kaggle Secrets)
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
os.environ["HF_TOKEN"] = hf_token

# 3. Load Model & Tokenizer
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

# 4. Load Dataset (From GitHub)
import requests
dataset_url = "https://raw.githubusercontent.com/archittmittal/MetaxBangalore/main/training_prompts.json"
data = requests.get(dataset_url).json()
train_dataset = Dataset.from_list(data)

# 5. Reward Functions
def reward_conflict_resolution(prompts, completions, **kwargs):
    rewards = []
    env = ConflictEnv()
    for prompt, completion in zip(prompts, completions):
        env.reset(task_name="auto")
        try:
            if "<thought>" in completion and "{" in completion:
                action_str = completion.split("}")[0].split("{")[1] + "}"
                env.step(ConflictAction.model_validate_json(action_str))
                rewards.append(env.get_reward())
            else:
                rewards.append(0.05)
        except:
            rewards.append(0.05)
    return rewards

def reward_format_check(prompts, completions, **kwargs):
    rewards = []
    for completion in completions:
        if "<thought>" in completion and "</thought>" in completion:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

# 6. Training Config (Kaggle Optimized)
training_args = GRPOConfig(
    output_dir = "/kaggle/working/conflict-env-qwen-1.5b-grpo",
    learning_rate = 5e-6,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 4,
    num_generations = 4, 
    max_prompt_length = 512,
    max_completion_length = 512,
    num_train_epochs = 1,
    logging_steps = 5,
    report_to = "none", # Set to "wandb" if you have a key
    push_to_hub = True,
    hub_model_id = "purvansh01/conflict-env-qwen-1.5b-grpo",
    hub_token = hf_token,
)

# 7. Initialize & Train
trainer = GRPOTrainer(
    model = model,
    reward_funcs = [reward_conflict_resolution, reward_format_check],
    args = training_args,
    train_dataset = train_dataset,
    tokenizer = tokenizer,
)

if __name__ == "__main__":
    print(f"[Kaggle] Starting GRPO Training Loop for {model_id}...")
    trainer.train()
    print("[Kaggle] Training Complete. Model pushed to Hugging Face Hub.")
