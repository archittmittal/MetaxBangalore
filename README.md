#  ConflictEnv: The Elite Reasoning Executive Assistant
### *Deep Reinforcement Learning for High-Stakes Scheduling*

**"Because scheduling is easy, but human life is complex."**

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/purvansh01/conflict-env)
[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Compliant-green)](https://github.com/OpenEnv/OpenEnv)
[![Theme #5](https://img.shields.io/badge/Hackathon_Theme-5_Wild_Card-purple)](#themes-covered-the-wild-card-play)

## 📺 [ConflictEnv Official Demo Video](https://youtu.be/TaVhJlouib4?si=ycyEp7aVV9fNk7gM)

> ** Official Hackathon Submission for Theme #5 (Wild Card)**  
> *Why choose one theme when you can tackle them all? ConflictEnv is a Wild Card submission engineered to naturally unify all hackathon themes (Multi-Agent Interactions, Long-Horizon Planning, World Modeling, and Self-Improvement) into a single, cohesive real-world challenge: cascading human scheduling conflicts.*


##  The Problem Statement
Existing AI agents can parse calendars, but they fail at **social judgment** and **dynamic adaptation**. 
- **The Efficiency Gap**: Every knowledge worker loses 2–4 hours per week to scheduling conflicts—totaling ~150 billion hours annually.
- **The Social Gap**: Moving an *investor pitch* is catastrophic, while moving a *1:1 with an intern* is acceptable but requires empathy. Current models lack a "Social IQ" for these trade-offs.
- **The Stability Gap**: Real-world APIs (Google Calendar, Travel APIs, Booking Systems) evolve. Static benchmarks fail when field names rename or date formats shift—a phenomenon known as **Schema Drift**.

**ConflictEnv** is an OpenEnv-compliant benchmark built to teach LLMs **constraint satisfaction under social pressure**. We move beyond standard text fine-tuning by using Group Relative Policy Optimization (GRPO) to train an agent that explores thousands of resolutions and learns what constitutes a "good" executive decision.

### System Architecture
ConflictEnv follows a strict **Reasoning-then-Action** protocol to ensure every decision is grounded in logic.

```mermaid
graph TD
    A[User Conflict Scenario] --> B{ConflictEnv Agent}
    B --> C[<thought> Deep Reasoning Block]
    C --> D[Social Intelligence Check]
    C --> E[Constraint Validation]
    D --> F[Final Action Decision]
    E --> F
    F --> G[Structured JSON Command]
    G --> H[Environment Execution]
    H --> I[Updated Calendar State]
```

##  Our Approach: ConflictEnv (OpenEnv Native)
We built **ConflictEnv**, a high-fidelity RL environment strictly following the **OpenEnv protocol**, to train agents that don't just "solve" calendars, but **negotiate life**.

### High-Level Workflow
```mermaid
graph TD
    subgraph Initialization ["1. Episode Setup"]
        S[Scenario Generator] -->|Tier Selection| D{Difficulty?}
        D -->|Easy/Med/Hard| P[Procedural Calendar Setup]
        V[Schema Drift Engine] -->|episode // 50| M[API Version V1/V2/V3]
    end

    subgraph AgentLoop ["2. Reasoning & Action Loop"]
        O[Observation State] -->|JSON Schema| T[thought Reasoning Block]
        T -->|Constraint Check| A[Action Selection]
        A -->|Tool Call| API[Simulated Tool API]
    end

    subgraph EnvCore ["3. Conflict Cascade Engine"]
        API -->|Reschedule/Cancel| C{Conflict?}
        C -->|Yes| CE[Cascade Logic: 3-5 Levels Deep]
        CE -->|Impact| AN[Actor Negotiation System]
        AN -->|Counter-Proposal| O
        C -->|No| O
    end

    subgraph Evaluation ["4. Reward & Terminal"]
        O -->|Terminal State| R[Multi-Signal Reward Computer]
        R -->|40% CRR| R1[Conflict Resolution Rate]
        R -->|30% SSI| R2[Stakeholder Satisfaction]
        R -->|20% Adh| R3[Deadline Adherence]
        R -->|10% Eff| R4[Step Optimization]
        R1 & R2 & R3 & R4 --> Final[Normalized Reward 0.05 - 0.95]
    end

    Initialization --> AgentLoop
    AgentLoop --> EnvCore
    EnvCore -->|Max Steps / All Resolved| Evaluation
```

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

##  Themes Covered (The "Wild Card" Play)
ConflictEnv is the first benchmark to **naturally unify all five hackathon themes** into a single, coherent human scenario:

1.  **Multi-Agent Interactions**: Managing 7 actors with competing incentives and **dynamic LLM-powered personalities**.
2.  **Long-Horizon Planning**: Resolving 5-day cascades with sparse end-of-episode rewards.
3.  **World Modeling (Prof/Pers)**: Interacting with drifting tool APIs while managing personal life trade-offs.
4.  **Self-Improvement**: An **Adaptive Curriculum** that increases difficulty (more actors, deeper cascades) as the agent's resolution rate improves.
5.  **Wild Card**: A **Continuous Learning Data Flywheel** that harvests every interaction into a structured RL dataset for offline fine-tuning (GRPO/PPO).

##  Environment Innovation (What Makes It Hard)
We deliberately pushed the boundaries of the OpenEnv framework to create a dynamic, game-theoretic environment that cannot be solved by simple regex or prompt engineering.

### 1. Dynamic LLM Stakeholders (Llama-3.2-1B)
Actors in our environment aren't passive scripts. They are powered by **Llama-3.2-1B-Instruct**, enabling:
*   **Contextual Negotiation**: Stakeholders generate dynamic, in-character rejections or counter-proposals based on their satisfaction level.
*   **Personality-Driven Feedback**: Move a board call to 3 AM, and the Boss will be "annoyed" or "passive-aggressive" in their feedback.
*   **Positive Reinforcement**: Good moves trigger appreciative acceptance messages, rewarding the agent with socially intelligent feedback.

### 2. Continuous Learning "Data Flywheel"
ConflictEnv is the first environment designed to **train itself**. 
*   **Experience Harvesting**: Every interaction (state, action, reward, LLM feedback) is saved into a high-performance **JSONL Experience Buffer**.
*   **RL-Ready Dataset**: This buffer builds a massive, structured dataset in real-time that can be pulled to fine-tune agents using GRPO or PPO, creating a closed-loop self-improvement system.

### 3. Anti-Reward Hacking
*   **Process Supervision:** The agent *must* output a `<thought>` block analyzing the social dynamic before its JSON action, or forfeit the reasoning bonus.
*   **Loop Detection:** Penalties are applied if the agent oscillates between states to farm formatting rewards.

##  Training Pipeline & Results (Proof It Works)
We trained a **Qwen-2.5-1.5B** model using **GRPO** (HuggingFace TRL + Unsloth) for 200 steps on Kaggle Dual-T4 GPUs. The training pipeline directly connected the RL loop to the `ConflictEnv` reward signals.

### Reward Engineering
Our reward function (`conflict_env/reward.py`) provides a rich, multi-dimensional signal normalized to `[0.05, 0.95]`:
*   **40% CRR** (Conflict Resolution Rate)
*   **30% SSI** (Stakeholder Satisfaction Index)
*   **20% Deadline Adherence**
*   **10% Efficiency** (Fewer steps)

```mermaid
graph LR
    A[Model Output] --> B(Reward Calculator)
    B --> C{Advantage Check}
    C -- High Score --> D[Reinforce Policy]
    C -- Low Score --> E[Update Weights]
```

### Observable Improvement
Reviewers, please note: *The model genuinely learned to reason.*

#### 1. GRPO Reward Curve
<img width="800" alt="Reward Curve" src="./plots/reward_curve.png" />
*Figure 1: Agent reward improves from ~5.0 (random format guessing) to ~29.7 (near-perfect) over 200 GRPO steps.*

#### 2. Policy Loss Convergence
<img width="800" alt="Loss Curve" src="./plots/loss_curve.png" />
*Figure 2: Policy loss drops from ~2.5 to ~0.28, indicating stable convergence.*

#### 3. Baseline vs. Trained Agent
<img width="800" alt="Baseline vs Trained" src="./plots/baseline_vs_trained.png" />
*Figure 3: After 200 GRPO steps, the trained agent achieves 100% JSON adherence, zero deadline violations, and 84% creative solution usage.*

#### 4. Reward Component Breakdown
<img width="800" alt="Reward Components" src="./plots/reward_components.png" />
*Figure 4: Decomposed reward shows a natural curriculum: the agent learns formatting first, then JSON structure, then domain reasoning.*

#### 5. Head-to-Head Battle: RL vs LLM
<img width="800" alt="Battle Heatmap" src="./plots/battle_heatmap.png" />
*Figure 5: The GRPO-trained reasoning agent dominates across all scenarios.*

##  Quickstart & Reproducibility

### Minimum Submission Requirements Checklist:
- [x] **OpenEnv Framework Used**: Built strictly on top of the framework.
- [x] **Working Training Script**: [Colab Training Notebook (Judges: Run Here)](https://colab.research.google.com/github/archittmittal/MetaxBangalore/blob/main/notebooks/conflictenv_training.ipynb)
- [x] **Real Training Evidence**: Loss and reward plots embedded above.
- [x] **HF Space Environment**: [Live on Hugging Face Spaces](https://huggingface.co/spaces/purvansh01/conflict-env)
- [x] **Pitch/Writeup**: See our [Technical Blog Post](https://huggingface.co/spaces/purvansh01/conflict-env/discussions/1) and [Project Report](docs/ConflictEnv_Project_Report.html)


### 1. Run the Environment Locally
```bash
pip install openenv
# Clone repository
git clone https://github.com/archittmittal/MetaxBangalore
cd MetaxBangalore
python -m conflict_env.server  # starts MCP server on localhost:8000
```

### 2. Run the Training Script
We recommend using our Unsloth-optimized Kaggle script.
```bash
# To run the end-to-end evaluation battle
python scripts/train_and_eval.py
```

## 🔗 Additional Resources
*   **[Technical Blog Post (Hackathon Writeup)](https://huggingface.co/spaces/purvansh01/conflict-env/discussions/1)**
*   **[HuggingFace Space (Live Environment)](https://huggingface.co/spaces/purvansh01/conflict-env)**
*   **[Colab Training Notebook (Judges: Run Here)](https://colab.research.google.com/github/archittmittal/MetaxBangalore/blob/main/notebooks/conflictenv_training.ipynb)**
*   **[Project Report / Technical Walkthrough](docs/ConflictEnv_Project_Report.html)**
*   **[Main Training Notebook (Local/Kaggle)](notebooks/conflictenv_training.ipynb)**
*   **[Kaggle Training Script](scripts/kaggle_training_script.py)**
*   **[GRPO Template (TRL)](scripts/grpo_training_template.py)**

---
*Built with ❤️ for the OpenEnv Hackathon (India 2026) by Archit Mittal and Purvansh Joshi.*
