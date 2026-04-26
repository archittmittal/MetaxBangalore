# ConflictEnv: Teaching a 1.5B Model to Navigate Human Conflict

**By Archit Mittal & Purvansh Joshi** | OpenEnv Hackathon India 2026

---

## TL;DR

We built **ConflictEnv**, a reinforcement learning environment where an AI agent must resolve scheduling conflicts involving real human social dynamics — not just move calendar blocks around. Using **Group Relative Policy Optimization (GRPO)** on a tiny **Qwen-2.5-1.5B** model, we trained an agent that went from outputting garbage to producing structured, socially-aware resolutions in under 150 training steps.

This blog walks through what we built, why it matters, how we trained it, and what we learned.

---

## 1. The Problem We Wanted to Solve

Every calendar app can tell you that two meetings overlap. None of them can tell you *which one to move*.

Consider this: Your boss's investor pitch and your spouse's anniversary dinner are both at 7 PM. A naive AI moves whichever is "easier." But the *right* answer depends on social context — how flexible is your spouse? How important is this investor? Has your boss already rescheduled twice this week?

**This is not a scheduling problem. It's a social negotiation problem.**

We wanted to build an environment that forces AI agents to reason about these human dynamics, not just optimize time slots.

---

## 2. What is ConflictEnv?

ConflictEnv is an **OpenEnv-compliant** reinforcement learning environment that simulates high-stakes calendar conflict resolution.

### The Setup
- **7 Stakeholder Types**: Boss, Spouse, Client, Doctor, School, Vendor, Airline — each with different flexibility levels, preferred times, and social weights.
- **3 Difficulty Tiers**: Easy (single overlap), Medium (cascading conflicts), Hard (multi-day chaos with hard deadlines).
- **7 Agent Commands**: `reschedule`, `cancel`, `draft_message`, `query_preference`, `escalate`, `confirm`, `resolve`.

### What Makes It Hard

**1. Cascading Conflicts**  
When you reschedule Event A, it might now overlap with Event B, which pushes against Event C's hard deadline. The agent must think multiple steps ahead.

**2. Dynamic Counter-Proposals**  
Stakeholders aren't passive. If the agent reschedules a meeting to a bad time, the environment generates a counter-proposal: *"I can't do 3 PM, but 4 PM works."* The agent must adapt.

**3. Social Burnout**  
Every stakeholder has a satisfaction score. Push someone too hard (reschedule their events three times), and they "burn out" — refusing all further negotiations. The agent learns that being aggressive is counterproductive.

**4. Anti-Gaming Measures**  
- The agent *must* output a `<thought>` reasoning block before its JSON action, or it gets zero reasoning bonus.
- Loop detection penalizes agents that oscillate between states to farm rewards.

---

## 3. Prior Art & Context

ConflictEnv sits at the intersection of **Game Theory**, **Social Simulation**, and **Constraint Satisfaction**. While benchmarks like *Diplomacy* (CICERO) focus on strategic deception, and social simulations like *The Sims* focus on life management, ConflictEnv is unique because it treats **Social Capital** as a hard constraint in an agentic workflow. We draw inspiration from:
- **Negotiation Environments**: Where agents must find a Pareto-optimal solution among competing stakeholder needs.
- **Hierarchical Planning**: Where high-level social goals must be translated into low-level API calls.
- **Stanford Town**: We extend the idea of "Generative Agents" by moving from simple roleplay to rigorous, rewarded optimization.

---

## 4. The Reward Signal

Getting the reward function right was the hardest part. We needed a signal that:
1. Rewards resolution (obviously)
2. Penalizes social damage (moving your spouse's dinner for the third time)
3. Rewards efficiency (fewer steps = better)
4. Forces structured output (valid JSON or nothing)

Our reward function (`conflict_env/reward.py`) uses a base weighted sum that defines the "Ideal Executive Behavior," with an additional additive bonus for reasoning:

| Component | Weight | Target Metric |
|---|---|---|
| **Conflict Resolution Rate (CRR)** | 0.40 | Binary success: Was the calendar conflict fixed? |
| **Stakeholder Satisfaction (SSI)** | 0.30 | Social debt: Did you maintain actor happiness? |
| **Deadline Adherence** | 0.20 | Physical constraints: Did you respect the flight/demo times? |
| **Efficiency** | 0.10 | Temporal cost: Did you solve it in the fewest steps? |

### The "Reasoning Booster" (Process Supervision)
To ensure the model doesn't just "guess" the right JSON, we added a **+0.10 Reasoning Bonus** if a valid `<thought>` block is present. Conversely, we apply a **-0.20 Loop Penalty** to prevent "reward hacking" (where agents oscillate between two states to farm step-wise formatting rewards). This brings the theoretical maximum reward to 1.10, which is then clamped to the OpenEnv-safe range of `[0.05, 0.95]`.

---

## 5. Training: GRPO on a 1.5B Model

### Why GRPO?
Group Relative Policy Optimization compares a *group* of model outputs rather than using a single baseline. For each prompt, the model generates 4 completions, and the one with the highest reward gets reinforced while the worst gets suppressed. This is perfect for our problem because there are many "acceptable" resolutions for a conflict, and GRPO lets the model explore them naturally.

### Why 1.5B?
We chose **Qwen-2.5-1.5B-Instruct** because:
- It runs on free Kaggle T4 GPUs (no A100 needed)
- It proves the environment works even with a small model
- It keeps inference fast for the live demo

### Training Configuration
```python
# Key hyperparameters
model: Qwen/Qwen2.5-1.5B-Instruct
quantization: 4-bit (via Unsloth)
lora_r: 16
learning_rate: 3e-5
batch_size: 1 (gradient accumulation: 4)
num_generations: 4  # GRPO group size
max_completion_length: 600
max_steps: 150
temperature: 0.9
```

### The V3.1 "Guidance Update"
Our first training runs hit a **zero gradient problem** — the model couldn't figure out what format to output, so all 4 generations scored equally, producing zero advantage signal.

The fix was surprisingly simple:
1. **Inject valid commands into the system prompt**: Instead of hoping the model discovers `reschedule` on its own, we listed all 7 commands explicitly.
2. **Increase completion length**: 400 tokens was causing JSON truncation. Bumping to 600 fixed it.
3. **Add tie-breaking**: A tiny length penalty (`1/(len+1)`) ensures completions rarely tie.

After these changes, training exploded into life.

---

## 6. Results

### The Learning Curve (Real Evidence)
The agent's reward improved from ~0.45 (random format guessing) to ~0.94 (near-perfect resolution) over 150 steps. 

<img width="800" alt="GRPO Training Progress" src="./plots/reward_curve.png" />
*Figure 1: Mean reward across GRPO generations. Note the explosive growth after the V3.1 guidance update.*

<img width="800" alt="Reward Component Breakdown" src="./plots/reward_components.png" />
*Figure 2: Decomposed reward signals showing the emergence of different capabilities over time.*

---

### Observation: Reward Decomposition Guides Curriculum Emergence

One of our most interesting observations was how the model spontaneously developed a hierarchical learning path based on the "gradient density" of our reward signals, without any manual curriculum scheduling.

### Phase 1: Format & Structure (Steps 1-25)
The model first mastered the "Easy Rewards"—the formatting bonus. It learned that `<thought>` tags and `{}` brackets were the most consistent way to avoid the `0.05` reward floor.

### Phase 2: Action Alignment (Steps 25-75)
Once the format was stable, the model began exploring the tool-use space. It learned that the `reschedule` command was the most effective way to trigger the `Conflict Resolution Rate` (40% weight) signal.

### Phase 3: Social Prioritization (Steps 75-150+)
The "Elite" behavior emerged last. Once it knew how to resolve conflicts, it began optimizing for the `Stakeholder Satisfaction Index` (30% weight). It learned the nuanced difference between moving a Vendor meeting (low SSI cost) vs. moving a Spouse meeting (high SSI cost). 

**While the weights are human-defined, the emergence of this sequential mastery suggests that reward decomposition can implicitly guide curriculum emergence in complex reasoning tasks.**

### Comparison: Before vs. After Training

| Metric | Base Qwen-2.5-1.5B | GRPO-Trained Agent |
|---|---|---|
| JSON Output Adherence | 0% | 100% |
| Deadline Compliance | 33% | 100% |
| Creative Solutions Used | 0% | 84% |
| Average Reward* | 0.06 | 0.94 |

*\*Methodology: Evaluated over 50 randomized episodes across all difficulty tiers. Both models utilized the identical system prompt and context. The 0.94 score represents performance at the 0.95 reward ceiling defined by the OpenEnv protocol.*

The untrained model literally could not produce valid JSON in our format. After training, it does so perfectly and uses creative strategies like `draft_message` to smooth over rescheduling.

### What the Model Actually Outputs

**Before training** (raw Qwen-2.5-1.5B):
```
I would suggest rescheduling the meeting. Let me know if you need help.
```

**After training** (GRPO-tuned):
```
<thought>
The investor pitch has a hard deadline and the Boss's satisfaction is critical 
(weight: 0.9). The spouse dinner is flexible but SSI is already at 0.6 — 
one more reschedule will trigger burnout. Best approach: keep the pitch, 
reschedule the vendor delivery (most flexible), and draft an empathetic 
message to the spouse about the dinner delay.
</thought>
{"command": "reschedule", "parameters": {"event_id": "vendor_delivery_001", 
"new_time": "14:00", "reason": "Conflict with high-priority investor pitch"}}
```

---

## 8. Lessons Learned & Baselines

### The PPO Baseline Comparison
To anchor our results, we compared the GRPO-trained reasoning agent against a traditional RL baseline (**PPO via Stable-Baselines3**). While PPO could solve "Easy" scenarios by brute-forcing calendar slots, it hit a complete reasoning ceiling on "Medium" and "Hard" tasks:
- **PPO Reward (Hard)**: 0.12 (Failed to maintain social capital)
- **GRPO Reward (Hard)**: 0.94 (Successfully navigated cascading constraints)

The traditional RL agent struggled to understand *why* to prioritize one stakeholder over another, often sacrificing high-weight relationships for immediate temporal slots.

### What Worked
- **GRPO over PPO**: PPO requires a value function, which adds complexity. GRPO's group comparison is simpler and works better for text generation tasks.
- **Dense reward signals**: The 4-component reward gave the model clear gradients to follow. A sparse "did you solve it?" reward would have failed with a 1.5B model.
- **Process supervision**: Requiring `<thought>` blocks forced the model to actually reason, not just pattern-match.

### What Didn't Work
- **Pure RL baseline (PPO/SB3)**: A traditional RL agent using Stable-Baselines3 could solve Easy scenarios but hit a complete reasoning ceiling on Medium/Hard. It couldn't understand *why* to prioritize one stakeholder over another.
- **Low completion length**: 400 tokens caused JSON truncation, which meant the model was punished for correct reasoning simply because it ran out of space.
- **No command guidance**: Without listing valid commands, the model invented its own (e.g., `"command": "fix_everything"`) which of course didn't parse.

### What Surprised Us
- **1.5B is enough**: We expected to need at least 7B for social reasoning. The 1.5B model handles it well when the environment provides rich enough feedback.

---

## 9. Try It Yourself

### Live Demo & Reproducibility
👉 **[ConflictEnv on Hugging Face Spaces](https://huggingface.co/spaces/purvansh01/conflict-env)**

The environment is OpenEnv-compliant and runs as a **FastAPI server** on port 7860 within the Space, exposing `/reset`, `/step`, and `/state` endpoints for automated evaluation.

### Reproduce Training
The full training notebook is available in this Space:
- `notebooks/conflictenv_training.ipynb`
- Or run the script: `python train_and_eval.py`

### Key Files in This Space
| File | Description |
|---|---|
| `app.py` | FastAPI server powering the environment |
| `conflict_env/env.py` | Core environment logic & state management |
| `conflict_env/reward.py` | Multi-signal reward function (0.05-0.95) |
| `openenv.yaml` | OpenEnv manifest & agent parameters |
| `plots/` | Training evidence (reward curves, comparisons) |
| `notebooks/` | Training & Demo notebooks |

---

## 10. Future Work

- **Larger Models**: Fine-tuning Qwen-7B or Llama-3-8B would likely unlock deeper social reasoning chains.
- **Multi-Turn Dialogue**: Currently the agent takes one action per step. A future version could have back-and-forth negotiation with stakeholders.
- **Real Calendar Integration**: Connecting to Google Calendar or Outlook to handle actual scheduling conflicts.
- **Schema Drift**: We designed the environment to support "schema drift" — where the JSON format changes mid-episode — but didn't fully train against it in this version.

---

## 11. Acknowledgments

Built for the **OpenEnv Hackathon (India 2026)**.

- **OpenEnv Team** for the framework that made this possible
- **Unsloth** for making 4-bit GRPO training accessible on free GPUs
- **HuggingFace TRL** for the GRPOTrainer implementation
- **Kaggle** for the free T4 GPU compute

---

*Built by Archit Mittal and Purvansh Joshi.*
