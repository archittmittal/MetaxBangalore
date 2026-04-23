# OpenEnv Hackathon: Ultimate FAQ & Resource Guide

## 📑 Table of Contents
1. [Core Concepts: RL, Rewards & Environments](#core-concepts)
2. [Technical Stack: OpenEnv, TRL & Unsloth](#technical-stack)
3. [Advanced Training: PPO vs GRPO](#ppo-vs-grpo)
4. [Environment & Reward Design Pitfalls](#pitfalls)
5. [Hackathon Strategy](#strategy)
6. [Unsloth RL Recipes](#unsloth-recipes)
7. [Official Resources](#official-resources)

---

<a name="core-concepts"></a>
## 1. Core Concepts: RL, Rewards & Environments

### What is reinforcement learning in the context of LLMs?
Reinforcement learning (RL) is a loop where the model generates an answer, plan, or action; that output is evaluated by a verifier or environment; and the resulting reward is used to update the model. It turns trial-and-error into weight updates instead of stuffing more examples into the prompt.
*   **SFT**: "Copy this good target."
*   **RL**: "Try many possibilities and move probability toward the ones that score better."

### Why do rewards matter so much?
Rewards are the **task specification**. RL gives you what you asked for, not necessarily what you meant. If your reward is gamed, the model will optimize for the exploit rather than the goal (specification gaming).

### What is rewards engineering?
The work of designing, combining, and monitoring signals. A practical reward often combines:
*   Execution success (Passes test)
*   Correctness (Matches schema)
*   Format compliance (Regex/JSON)
*   Latency/Memory/Safety checks

### What is RLVR vs RLVE?
*   **RLVR (Verifiable Rewards)**: Using a programmatic verifier (math, code, etc.) instead of a learned reward model.
*   **RLVE (Verifiable Environments)**: Environments that procedurally generate tasks and adjust difficulty, preventing the model from saturating on a static dataset.

---

<a name="technical-stack"></a>
## 2. Technical Stack: OpenEnv, TRL & Unsloth

### What is OpenEnv?
A standardized interface (`reset`, `step`, `state`) for LLM-agent environments. It treats environments as portable, versioned software artifacts (FastAPI/Docker).
*   **Hackathon Value**: Reduces plumbing; lets you focus on task design and rewards.

### Where do TRL and Unsloth fit in?
*   **TRL (Hugging Face)**: The training library providing RL algorithms like GRPO and PPO.
*   **Unsloth**: The acceleration layer. Makes training and inference (rollout generation) significantly faster and more memory-efficient.

---

<a name="ppo-vs-grpo"></a>
## 3. Advanced Training: PPO vs GRPO

### PPO (Proximal Policy Optimization)
A classic, stable algorithm that constrains how much a policy changes per update. Requires a "Value Model" alongside the policy.

### GRPO (Group Relative Policy Optimization)
Popularized by DeepSeekMath. It compares sampled outputs within a group to estimate advantage.
*   **Key Advantage**: Removes the need for a Value Model, making it significantly more memory-efficient for large LLMs.

---

<a name="pitfalls"></a>
## 4. Environment & Reward Design Pitfalls

### Common Environment Pitfalls
*   **Brittleness**: Rule-based verifiers that reject correct but oddly-formatted answers.
*   **Judge Exploitation**: Using an LLM-as-a-judge that the policy learns to "fool" with surface-level patterns.
*   **Static Difficulty**: If tasks are too hard, success probability is zero and learning stalls.

### Common Reward Pitfalls
*   **Reward Hacking**: Model finds a shortcut (e.g., editing variables or bypassing timers).
*   **Conflicting Signals**: Rewarding brevity and verbosity at the same time causes instability.
*   **Sparse Rewards**: Only giving a reward at the very end of a long task makes learning extremely slow.

---

<a name="strategy"></a>
## 5. Hackathon Strategy

### The Debugging Order
1.  Debug environment manually.
2.  Debug the verifier.
3.  Run scripted baseline policies.
4.  Run a frozen model.
5.  Run a tiny RL experiment.
6.  Scale up.

### Winning Team Structure
*   **Person A (Env)**: Logic, reset/step, safety.
*   **Person B (Rewards)**: Verifiers, anti-hacking.
*   **Person C (Training)**: TRL + Unsloth, metrics.
*   **Person D (Demo)**: UI, recording, benchmarking.

---

<a name="unsloth-recipes"></a>
## 6. Unsloth RL Recipes

*   **Simple Starter**: Qwen2.5 (3B) GRPO Notebook.
*   **Capability Focus**: Llama 3.1 (8B) GRPO Notebook.
*   **Interactive Agent Focus**: GPT-OSS 20B + 2048 Game RL Notebook.
*   **Advanced Rewards**: Qwen3 (4B) GRPO with proximity scoring.

---

<a name="official-resources"></a>
## 7. Official Resources

### GitHub Repos
*   [OpenEnv Core](https://github.com/meta-pytorch/OpenEnv)
*   [OpenEnv Tutorials](https://github.com/meta-pytorch/OpenEnv/tree/main/tutorial)
*   [OpenEnv Hub (HF)](https://huggingface.co/openenv)

### Videos
*   [Mega Lecture: RL & OpenEnv (Recommended)](https://www.youtube.com/watch?v=Jew4lhAiqnw)
*   [Workshop: Building RL Environments](https://www.youtube.com/watch?v=0airz7BhBiA)

---
*Reference: Official Hackathon FAQ and Resource Collection.*
