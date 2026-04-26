---
title: ConflictEnv
emoji: 📈
colorFrom: green
colorTo: indigo
sdk: docker
pinned: false
---

# 🤖 ConflictEnv: The Elite Reasoning Executive Assistant
### *Deep Reinforcement Learning for High-Stakes Scheduling*

**"Because scheduling is easy, but human life is complex."**

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/purvansh01/conflict-env)
[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-green)](https://github.com/OpenEnv/OpenEnv)

## 🚨 The Problem Statement
Existing AI agents can parse calendars, but they fail at **social judgment** and **dynamic adaptation**. 
- **The Efficiency Gap**: Every knowledge worker loses 2–4 hours per week to scheduling conflicts—totaling ~150 billion hours annually.
- **The Social Gap**: Moving an *investor pitch* is catastrophic, while moving a *1:1 with an intern* is acceptable but requires empathy. Current models lack a "Social IQ" for these trade-offs.
- **The Stability Gap**: Real-world APIs (Google Calendar, Travel APIs, Booking Systems) evolve. Static benchmarks fail when field names rename or date formats shift—a phenomenon known as **Schema Drift**.

## 🚀 Our Approach: ConflictEnv (OpenEnv Native)
We built **ConflictEnv**, a high-fidelity RL environment strictly following the **OpenEnv protocol**, to train agents that don't just "solve" calendars, but **negotiate life**.

### 1. Cascading Conflict Engine
In ConflictEnv, actions have consequences. Moving a dentist appointment might be the only way to attend a board meeting, but that dentist only has availability during your child's recital. Conflicts cascade **3–5 levels deep**, requiring the agent to reason through long-term dependencies.

### 2. Multi-Agent Negotiation (Social IQ)
We simulate **7 distinct stakeholders** (Boss, Spouse, Client, Doctor, School, Vendor, Airline), each with:
- **Power Weights**: Rescheduling the Boss carries higher risk than rescheduling a Vendor.
- **Flexibility Scores**: Some events are "Hard Deadlines" (Flights), others are negotiable.
- **Tone Sensitivity**: Actors respond to the agent's message tone. Empathy preserves the **Stakeholder Satisfaction Index (SSI)**.

### 3. Dynamic Schema Drift (Patronus AI Bonus)
To ensure the agent genuinely understands the *world* rather than just memorizing a prompt, we implemented a **Schema Drift Engine**. Every 50 episodes, the underlying API contracts mutate (V1 → V2 → V3):
- **V1 (Baseline)**: Standard JSON structures.
- **V2 (Renames)**: `start_time` becomes `startTime`.
- **V3 (Structural)**: Flat structures become nested objects.

### 4. Training with GRPO
We trained a **Qwen-2.5-7B-Instruct** model using **Group Relative Policy Optimization (GRPO)**. Our reward function provides a rich, multi-dimensional signal:
- **40% CRR** (Conflict Resolution Rate)
- **30% SSI** (Stakeholder Satisfaction Index)
- **20% Deadline Adherence**
- **10% Efficiency** (Step count optimization)

## 🎭 Themes Covered (The "Wild Card" Play)
ConflictEnv is the first benchmark to **naturally unify all five hackathon themes** into a single, coherent human scenario:

1.  **Multi-Agent Interactions**: Managing 7 actors with competing incentives and counter-proposals.
2.  **Long-Horizon Planning**: Resolving 5-day cascades with sparse end-of-episode rewards.
3.  **World Modeling (Prof/Pers)**: Interacting with drifting tool APIs while managing personal life trade-offs.
4.  **Self-Improvement**: An **Adaptive Curriculum** that increases difficulty (more actors, deeper cascades) as the agent's resolution rate improves.
5.  **Wild Card**: Using "Calendar Chaos" as a meta-narrative to demonstrate that general reasoning is the ultimate solution to personal productivity.

## 📊 Performance Evidence
*The model genuinely learned to reason through the noise of schema drift.*

### GRPO Reward Curve
<img width="800" alt="Reward Curve" src="./plots/reward_curve.png" />
*Figure 1: Reward improves from ~5.0 (formatting) to ~29.7 (reasoning) over 200 GRPO steps.*

### Baseline vs. Trained Agent
<img width="800" alt="Baseline vs Trained" src="./plots/baseline_vs_trained.png" />
*Figure 2: The GRPO-trained agent achieves 100% JSON adherence and 84% creative resolution success.*

## 🔗 Quickstart
```bash
pip install openenv
git clone https://github.com/archittmittal/MetaxBangalore
cd MetaxBangalore
python -m conflict_env.server  # Starts the OpenEnv-compliant server
```

---
*Built with ❤️ for the OpenEnv Hackathon (India 2026) by Archit Mittal and Purvansh Joshi.*
