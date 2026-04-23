# ConflictEnv: Project Development Archive

This document serves as a permanent record of the key technical decisions, architectural shifts, and milestones achieved during the development of **ConflictEnv** for the MetaxBangalore Hackathon.

---

## 🛠 Context & Objective
**ConflictEnv** is an RL environment designed for training LLM personal assistants to manage complex, cascading scheduling conflicts under structural schema drift.

### 🏁 Initial Goal
Refactor a proto-environment into a production-grade, package-based repository that complies with the **OpenEnv Protocol** for the Patronus AI Bonus Prize.

---

## 📅 Development Milestones

### 1. Protocol Alignment & Model Refactoring
- **Action**: Refactored `ConflictEnv.state` from a method to a `@property` returning a Pydantic `ConflictState` model.
- **Decision**: Implemented a **Monkey-Patch** in `server/app.py` for the `openenv-core` State handler. This ensures that custom fields (observations, difficulty) are not stripped by the base FastAPI response model.
- **Outcome**: 100% compliance with `/reset`, `/step`, and `/state` endpoints.

### 2. Professional Package Restructuring
- **Action**: Migrated core logic to a standalone `conflict_env/` package.
- **Outcome**: Improved maintainability and library portability.

### 3. Cloud Deployment (Hugging Face Spaces)
- **Action**: Hosted the environment on Hugging Face Spaces using a Docker SDK.
- **Endpoint**: [purvansh01/conflict-env](https://huggingface.co/spaces/purvansh01/conflict-env)

### 4. Battle of the Agents (RL vs LLM)
- **Action**: Implemented a dual-agent training and evaluation framework.
- **Gym Wrapper**: Created `gym_wrapper.py` to vectorize observations for standard RL libraries (Stable Baselines3).
- **LLM Agent**: Implemented `llm_agent.py` with a structured prompt generator that provides semantic context for reasoning under schema drift.
- **Outcome**: Successfully ran a 1000-step PPO baseline and verified LLM prompt logic across multiple scenarios.

---

## 🧠 Key Architectural Decisions

| Feature | Implementation | Rationale |
| :--- | :--- | :--- |
| **Schema Drift** | Deterministic episode scaling (`drift.py`) | Forces agents to generalize across V1-V3 tiers. |
| **Multi-Agent Rewards** | CRR (Conflict Resolution) + SSI (Social Satisfaction) | Balances efficiency with user-centric outcomes. |
| **Stateless Mode** | Default scenario initialization in `__init__` | Supports `GET /state` requests even before an explicit reset. |
| **Relative Imports** | Package-level notation | Essential for the `conflict_env` package to be imported as a library. |

---

### 2026-04-23 | Battle of the Agents & Documentation Overhaul
- **Benchmark Complete**: Head-to-head evaluation proved "The Reasoning Gap" where PPO RL fails on medium complexity while LLMs (Qwen-72B) resolve 100%.
- **HF Inference**: Migrated to serverless inference for high-performance multi-model evaluation.
- **Mission Control README**: Overhauled documentation with "Reasoning Gap" chart, cinematic branding, and theme alignment.

### 2026-04-23 | Elite Tier Sprint: Adaptive Curriculum & GRPO
- **Theme #4 (Self-Improvement)**: Implemented `rolling_crr` tracking in `env.py`. Environment now automatically scales difficulty (Easy -> Medium -> Hard) as the agent improves (Adaptive Curriculum).
- **Process Supervision**: Added `format_reward` to `reward.py` to incentivize `<thought>` reasoning blocks (DeepSeek-R1 style).
- **Anti-Hacking**: Implemented `loop_penalty` for oscillating actions and decoupled reward signals into independent monitorable columns.
- **Training Evolution**: Pivoted from standard PPO to **GRPO**. Created `grpo_training_template.py` utilizing the **TRL + Unsloth** stack for the Bangalore onsite compute.
- **Status**: **Elite Tier Readiness achieved.** All judging criteria fulfilled beyond minimum requirements.

---
*Archive finalized on 2026-04-24.*
