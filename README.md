---
title: Conflict Env
emoji: 📈
colorFrom: green
colorTo: indigo
sdk: docker
pinned: false
---

# 🗓️ ConflictEnv: The Personal Assistant Stress Test

**A Benchmark for Reasoning-Based Scheduling & Social Nuance under Schema Drift.**

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/purvansh01/conflict-env)
[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-green)](https://github.com/OpenEnv/OpenEnv)

---

## 🚀 The Mission: Beyond Simple Scheduling
Most AI assistants can book a calendar invite. But what happens when **everything goes wrong at once?**

**ConflictEnv** is a high-fidelity simulation designed for **Theme #3.2 (Personalized Tasks)** of the OpenEnv Hackathon. It challenges agents to act as executive assistants managing a "Monday from Hell": cascading conflicts, stakeholder egos, and dynamic API changes (Schema Drift).

### Why it matters:
*   **Complexity**: Resolving one conflict often creates three more.
*   **Social Nuance**: It’s not just about the time; it’s about *who* you offend.
*   **Resilience**: The environment evolves. If your agent relies on fixed field names (V1), it will break when the "API" updates (V3).

---

## 📊 The "Battle of the Agents" (Leaderboard)
We benchmarked a standard **Reinforcement Learning (PPO)** agent against a **72B Parameter Reasoning Agent (Qwen-2.5)**.

| Scenario | RL Agent (PPO) | LLM Agent (Qwen-72B) | The Verdict |
| :--- | :--- | :--- | :--- |
| **Morning Crunch (Easy)** | 100% Resolved | 100% Resolved | RL is efficient for simple tasks. |
| **Travel Chaos (Medium)** | **0% Resolved** | **100% Resolved** | **The Reasoning Gap**: RL hits a ceiling. |
| **Monday from Hell (Hard)** | 0% Resolved | 0% (In Progress) | Complex multi-step reasoning is the frontier. |

![The Reasoning Gap](https://raw.githubusercontent.com/archittmittal/MetaxBangalore/main/docs/reasoning_gap.png)

### 📈 The Reasoning Gap
Our data shows that while RL agents can be optimized for static rewards, they lack the **Theory of Mind** required to negotiate between a "Grumpy Boss" and a "Strict Doctor." LLMs use zero-shot reasoning to bridge this gap.

---

## 🛠️ Environment Innovation
1.  **Dual-Metric Rewards**: 
    *   **CRR (Conflict Resolution Rate)**: Did you fix the schedule?
    *   **SSI (Social Satisfaction Index)**: Are the humans still happy?
2.  **Schema Drift (V1-V3)**: We simulate real-world software evolution. Fields like `time` might change to `start_period` across episodes, testing the agent's semantic understanding.
3.  **OpenEnv Protocol**: Fully compliant with the latest OpenEnv spec (`/reset`, `/step`, `/state`).

---

## 🧪 Training & Evaluation
### Minimal Training Script
We provide a unified pipeline for training RL baselines and evaluating LLMs.
```bash
# Train the RL Baseline
python train_and_eval.py --timesteps 50000

# Evaluate a Reasoning Agent (via HF Inference)
python train_and_eval.py --llm-only --llm-model qwen-72b
```

### [🔗 Colab Training Notebook (Unsloth/TRL)](https://github.com/archittmittal/MetaxBangalore/blob/main/notebooks/colab_training.py)
*Check the notebook directory for the training script.*

---

## 🎬 Presentation Materials
*   **Mini-Blog**: [ConflictEnv: Why RL isn't enough for your Calendar](https://huggingface.co/blog/purvansh01/conflict-env)
*   **Demo Video**: [The Monday From Hell Simulation](https://youtube.com/...)
*   **Pitch Deck**: [ConflictEnv Strategy PDF](https://github.com/...)

---

## 🧑‍💻 The Team
Built with ❤️ for the **OpenEnv Hackathon (India 2026)**.

*"Ambition is the reward for solving the first conflict."*
