---
title: Conflict Env
emoji: 📈
colorFrom: green
colorTo: indigo
sdk: docker
pinned: false
---

# 🗓️ ConflictEnv: The Personal Assistant Stress Test

**A Unified Full-Spectrum Benchmark for Reasoning-Based Scheduling & Social Nuance.**

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/purvansh01/conflict-env)
[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-green)](https://github.com/OpenEnv/OpenEnv)

---

## 🌌 Open Innovation: Covering All 5 Themes
**ConflictEnv** is the first environment to naturally unify all five hackathon themes into a single, high-stakes human problem.

| Theme | Feature in ConflictEnv | Core Technical Driver |
| :--- | :--- | :--- |
| **#1 Multi-Agent** | 7 Distinct Actor Archetypes | **Theory-of-Mind Negotiation**: Actors counter-propose & reject. |
| **#2 Long-Horizon** | Cascading 5-Day Conflicts | **Multi-Step Reasoning**: Monday's fix creates Friday's crisis. |
| **#3 World Modeling** | Real API Ecosystems | **Causal Logic**: Interacting with Calendar, Travel, & Venue APIs. |
| **#3.2 Personalized** | Executive Assistant Logic | **Social Nuance**: Managing emails, errands, and egos. |
| **#4 Self-Improvement** | V1 → V3 Schema Drift | **Forced Generalization**: Adapting to evolving API contracts. |
| **#5 Wild Card** | Unified "Life-Manager" Narrative | **Innovation**: One coherent env for the entire hackathon scope. |

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
Our data proves that while RL agents can be optimized for static rewards, they lack the **Theory of Mind** required to negotiate between a "Grumpy Boss" and a "Strict Doctor." LLMs use zero-shot reasoning to bridge this gap.

---

## 🛠️ Environment Innovation
1.  **Dual-Metric Rewards**: 
    *   **CRR (Conflict Resolution Rate)**: Did you fix the schedule?
    *   **SSI (Social Satisfaction Index)**: Are the humans still happy?
2.  **Schema Drift (V1-V3)**: Simulating real-world software evolution. Field names mutate across versions to test agent resilience.
3.  **OpenEnv Protocol**: 100% compliant with the latest OpenEnv spec (`/reset`, `/step`, `/state`).

---

## 📄 [Technical Deep Dive: The Project Report](https://github.com/archittmittal/MetaxBangalore/blob/main/docs/ConflictEnv_Project_Report.html)
For a comprehensive breakdown of our architecture, SWOT analysis, and competitive positioning, refer to our full **Project Report**.

---

## 🧪 Training & Evaluation
### Minimal Training Script
```bash
# Train the RL Baseline
python train_and_eval.py --timesteps 50000

# Evaluate a Reasoning Agent (via HF Inference)
python train_and_eval.py --llm-only --llm-model qwen-72b
```

### [🔗 Colab Training Notebook (Unsloth/TRL/GRPO)](https://github.com/archittmittal/MetaxBangalore/blob/main/notebooks/colab_training.py)
*Optimized for onsite fine-tuning of 7B-parameter reasoning models.*

---

## 🧑‍💻 The Team
Built with ❤️ for the **OpenEnv Hackathon (India 2026)**.

*"Ambition is the reward for solving the first conflict."*
