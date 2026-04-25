"""
ConflictEnv GRPO V3 — Overnight Training Script
=================================================
Train for 500 steps with 5 reward functions:
  1. Format (structure compliance)
  2. Logic (constraint awareness)
  3. Social (empathy & tone)
  4. Anti-Hack (penalize repetition & lazy outputs)
  5. Creative Alternatives (reward 3rd-party solutions)

Run on Kaggle with Dual T4 GPUs.
Estimated time: ~1.5 hours for 500 steps.
"""

import os
import json
import re
import subprocess
import torch
from datasets import Dataset

# ============================================================
# 1. Install Dependencies (Kaggle Specific)
# ============================================================
def install_deps():
    print("[Kaggle] Installing Elite RL Stack...")
    commands = [
        "pip install --upgrade pip",
        'pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"',
        'pip install --no-deps "trl<0.13.0" peft accelerate bitsandbytes',
        "pip install git+https://github.com/archittmittal/MetaxBangalore.git",
        "pip install python-dotenv gymnasium stable-baselines3"
    ]
    for cmd in commands:
        subprocess.run(cmd, shell=True, check=True)

try:
    from unsloth import FastLanguageModel, PatchFastRL
    from trl import GRPOTrainer, GRPOConfig
except ImportError:
    install_deps()
    from unsloth import FastLanguageModel, PatchFastRL
    from trl import GRPOTrainer, GRPOConfig

PatchFastRL("GRPO", "Unsloth")

# ============================================================
# 2. Model & Tokenizer (Qwen 2.5 1.5B)
# ============================================================
model_id = "Qwen/Qwen2.5-1.5B-Instruct"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_id,
    max_seq_length=2048,
    load_in_4bit=True,
    fast_inference=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=32,             # Doubled from 16 -> more capacity to learn nuance
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,    # Match r for stable training
    lora_dropout=0,
)

# ============================================================
# 3. Data — ChatML Formatting
# ============================================================
SYSTEM_PROMPT = """You are an Elite Executive Assistant AI. Your job is to resolve complex, overlapping scheduling conflicts.

RULES:
1. Think step-by-step inside <thought>...</thought> tags.
2. Consider ALL stakeholders: Boss, Spouse, Client, Doctor, etc.
3. Never move Hard Deadlines (flights, medical appointments).
4. ALWAYS suggest creative 3rd-party alternatives before canceling anything.
5. Output a valid JSON command after </thought>.

ALTERNATIVE PLAYBOOK (Use these instead of canceling):
- Wife/Spouse pickup conflict → Book an Uber/Ola for them
- Can't attend meeting in person → Join via Zoom/Google Meet video call
- Team standup clashes with client → Delegate to Technical Lead or send async Slack update
- Doctor appointment vs work → Send a family member or reschedule to next available slot
- Server crash during dinner → Delegate to on-call engineer, set up PagerDuty alert
- Child school pickup → Ask a trusted neighbor/family member, or book a cab
- Gym session clashes → Move gym to early morning or skip (lowest priority)

FORMAT:
<thought>
[Your step-by-step reasoning here, including which alternatives you considered]
</thought>
{"command": "reschedule|cancel|delegate|book_service", "event_id": "...", "parameters": {...}}"""

def apply_chatml_formatting(example):
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Scenario: {example['scenario']}\nDifficulty: {example['difficulty']}"}
    ]
    return {"prompt": tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) + "<thought>\n"}

import requests
dataset_url = "https://raw.githubusercontent.com/archittmittal/MetaxBangalore/main/training_prompts.json"
data = requests.get(dataset_url).json()
raw_dataset = Dataset.from_list(data)
train_dataset = raw_dataset.map(apply_chatml_formatting)

print(f"✅ Dataset loaded: {len(train_dataset)} scenarios")

# ============================================================
# 4. REWARD FUNCTIONS (The Brain — 4 Dimensions)
# ============================================================

def reward_format(completions, **kwargs):
    """
    R_format: Structural compliance reward.
    +10 for <thought>...</thought> tags
    +10 for valid JSON after </thought>
    +5  bonus for JSON containing "command" key
    """
    responses = [c[0]["content"] for c in completions]
    rewards = []
    for r in responses:
        score = 0.0
        # Check thought tags
        if "<thought>" in r and "</thought>" in r:
            score += 10.0
        # Check JSON
        try:
            thought_end = r.rfind("</thought>")
            if thought_end != -1:
                json_part = r[thought_end + 10:].strip()
                # Remove trailing text after the JSON block
                brace_count = 0
                json_end = -1
                for i, ch in enumerate(json_part):
                    if ch == '{': brace_count += 1
                    elif ch == '}': brace_count -= 1
                    if brace_count == 0 and i > 0:
                        json_end = i + 1
                        break
                if json_end > 0:
                    clean_json = json_part[:json_end]
                    parsed = json.loads(clean_json)
                    score += 10.0
                    # Bonus for having "command" key
                    if "command" in parsed:
                        score += 5.0
        except:
            pass
        rewards.append(score)
    return rewards


def reward_logic(completions, **kwargs):
    """
    R_logic: Constraint awareness reward.
    Rewards the model for showing awareness of hard vs soft constraints.
    +3 for mentioning "hard deadline" or "unmovable"
    +3 for mentioning "reschedule" or "delegate" (action awareness)
    +2 for mentioning "conflict" or "overlap"
    +2 for mentioning time-specific reasoning (AM/PM, specific hours)
    -5 penalty for suggesting to cancel a flight or medical appointment
    """
    responses = [c[0]["content"] for c in completions]
    rewards = []
    for r in responses:
        r_lower = r.lower()
        score = 0.0

        # Hard constraint awareness
        if any(kw in r_lower for kw in ["hard deadline", "unmovable", "cannot move", "fixed", "non-negotiable"]):
            score += 3.0

        # Action awareness
        if any(kw in r_lower for kw in ["reschedule", "delegate", "move to", "shift to", "uber", "video call"]):
            score += 3.0

        # Conflict identification
        if any(kw in r_lower for kw in ["conflict", "overlap", "clash", "collide", "simultaneous"]):
            score += 2.0

        # Time reasoning
        if re.search(r'\d{1,2}:\d{2}', r):
            score += 2.0

        # PENALTY: Never cancel flights or medical
        if any(bad in r_lower for bad in ["cancel flight", "cancel the flight", "cancel doctor", "cancel medical"]):
            score -= 5.0

        rewards.append(max(score, 0.0))
    return rewards


def reward_social(completions, **kwargs):
    """
    R_social: Social intelligence & empathy reward.
    +2 for mentioning stakeholder by role (boss, spouse, client)
    +2 for showing empathy (apologize, understand, accommodate)
    +2 for considering multiple perspectives
    +2 for mentioning 3rd party solutions
    -3 penalty for dismissing personal/family events
    """
    responses = [c[0]["content"] for c in completions]
    rewards = []
    for r in responses:
        r_lower = r.lower()
        score = 0.0

        # Stakeholder awareness
        stakeholders_found = sum(1 for s in ["boss", "spouse", "wife", "client", "doctor", "team"] if s in r_lower)
        score += min(stakeholders_found, 3) * 0.7  # Up to 2.1

        # Empathy signals
        if any(kw in r_lower for kw in ["apologize", "sorry", "understand", "accommodate", "sensitive", "important to"]):
            score += 2.0

        # Multiple perspective analysis
        if any(kw in r_lower for kw in ["on one hand", "however", "balance", "trade-off", "priority", "weigh"]):
            score += 2.0

        # 3rd party solutions
        if any(kw in r_lower for kw in ["uber", "taxi", "delegate", "send someone", "video call", "remote"]):
            score += 2.0

        # PENALTY: Dismissing family/personal
        if any(bad in r_lower for bad in ["not important", "skip wife", "ignore spouse", "cancel anniversary", "not a priority"]):
            score -= 3.0

        rewards.append(max(score, 0.0))
    return rewards


def reward_anti_hack(completions, **kwargs):
    """
    R_antihack: Penalizes lazy, repetitive, or gaming outputs.
    -3 for repetitive sentences (copy-paste loops)
    -2 for extremely short thoughts (< 20 words)
    -2 for extremely long outputs (> 500 words, likely hallucination)
    +1 bonus for concise, focused output (50-200 words)
    """
    responses = [c[0]["content"] for c in completions]
    rewards = []
    for r in responses:
        score = 0.0
        words = r.split()
        word_count = len(words)

        # Penalize too short (lazy)
        if word_count < 20:
            score -= 2.0

        # Penalize too long (hallucination)
        if word_count > 500:
            score -= 2.0

        # Reward sweet spot
        if 50 <= word_count <= 200:
            score += 1.0

        # Detect repetition (same sentence appearing 3+ times)
        sentences = r.split('.')
        seen = {}
        for s in sentences:
            s_clean = s.strip().lower()
            if len(s_clean) > 10:
                seen[s_clean] = seen.get(s_clean, 0) + 1
        if any(v >= 3 for v in seen.values()):
            score -= 3.0

        rewards.append(score)
    return rewards

def reward_creative_alternatives(completions, **kwargs):
    """
    R_alternatives: Rewards creative 3rd-party problem solving.
    The model should suggest alternatives INSTEAD of canceling.
    
    +3 for each unique alternative mentioned (Uber, delegate, video call, etc.)
    +2 bonus for pairing an alternative with a specific event
    +3 bonus for suggesting 2+ different alternatives in one response
    -4 penalty for canceling without suggesting any alternative first
    """
    responses = [c[0]["content"] for c in completions]
    rewards = []
    
    ALTERNATIVES = {
        "transport": ["uber", "ola", "taxi", "cab", "lyft", "book a ride", "send a car"],
        "delegation": ["delegate", "assign to", "hand off", "ask someone", "send someone", 
                       "technical lead", "on-call", "team member can", "colleague"],
        "remote": ["video call", "zoom", "google meet", "teams call", "join remotely", 
                   "dial in", "virtual", "async update", "slack update"],
        "reschedule_smart": ["next available", "early morning", "move to tomorrow", 
                             "swap with", "combine with"],
        "family_help": ["family member", "neighbor", "relative", "parent can", 
                        "sibling", "friend can help"],
    }
    
    for r in responses:
        r_lower = r.lower()
        score = 0.0
        categories_hit = 0
        
        # Check each category of alternatives
        for category, keywords in ALTERNATIVES.items():
            if any(kw in r_lower for kw in keywords):
                score += 3.0
                categories_hit += 1
        
        # Bonus for multi-category thinking (2+ different types of alternatives)
        if categories_hit >= 2:
            score += 3.0
        
        # Bonus for pairing alternative with specific event context
        event_words = ["meeting", "pickup", "appointment", "dinner", "flight", "standup", "demo"]
        if categories_hit > 0 and any(ew in r_lower for ew in event_words):
            score += 2.0
        
        # PENALTY: Canceling without suggesting any alternative
        if "cancel" in r_lower and categories_hit == 0:
            score -= 4.0
        
        rewards.append(max(score, 0.0))
    return rewards



# ============================================================
# 5. Training Configuration (Overnight — 500 Steps)
# ============================================================
training_args = GRPOConfig(
    output_dir="output_v3",
    learning_rate=3e-6,          # Slightly lower for stability over longer runs
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,           # Warm up for first 5% of steps
    logging_steps=1,
    max_steps=500,               # OVERNIGHT: 500 steps (~1.5 hrs on Dual T4)
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=6,           # More candidates per step = better GRPO signal
    max_prompt_length=512,
    max_completion_length=768,   # Longer outputs for richer reasoning
    push_to_hub=True,
    hub_model_id="purvansh01/conflict-env-v3-overnight",
    save_steps=100,
    bf16=True,                   # Mixed precision for speed
)

# ============================================================
# 6. Launch Training
# ============================================================
trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_format, reward_logic, reward_social, reward_anti_hack, reward_creative_alternatives],
    args=training_args,
    train_dataset=train_dataset,
)

print("=" * 60)
print("🌙 OVERNIGHT V3 TRAINING — STARTING")
print(f"   Steps: {training_args.max_steps}")
print(f"   Rewards: Format + Logic + Social + AntiHack + Alternatives")
print(f"   LoRA Rank: 32 (doubled)")
print(f"   Push to: {training_args.hub_model_id}")
print("=" * 60)

trainer.train()

print("🎉 V3 Training Complete! Model pushed to HuggingFace.")
