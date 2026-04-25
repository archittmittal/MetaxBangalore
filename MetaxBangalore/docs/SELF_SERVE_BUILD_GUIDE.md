# Hackathon Self-Serve Guide: Build an RL Environment, Train an LLM, Ship a Demo

## 0) What You Are Building
The core idea is not just to fine-tune a text model, but to build a specialized LLM system that can act inside an environment, get feedback, and improve through reinforcement learning.
**Stack**: Environment → verifier/reward functions → TRL trainer → Unsloth for efficiency → deployment on OpenEnv / Spaces.

## 1) Start with the Right Project Idea
Pick a task with these properties:
*   **Step-by-step actions**: The model can act incrementally.
*   **Programmatic verification**: Success can be verified by code.
*   **Goldilocks difficulty**: Hard enough to be interesting, but success probability must be > 0 for RL to work.
*   **Crisp verification**: Prefer objective rewards over subjective "human-like" looks.

## 2) The RL Loop
1.  **Prompt**: Give the model a prompt.
2.  **Act**: Model generates an action/strategy.
3.  **Execute**: Run the action in the environment/verifier.
4.  **Reward**: Convert result into a numerical reward.
5.  **Update**: Shift probability mass toward higher-reward behavior via backpropagation.

## 3) SFT vs RL
*   **Use SFT**: If you have a lot of good data.
*   **Use RL**: If you have no data but can verify outputs.
*   **Hybrid**: Do light SFT first for formatting/scaffolding, then RL for improvement.

## 4) Environment Design (First-Class Artifact)
*   `reset()`: Fresh episode.
*   `step(action)`: Apply action, return result.
*   `state()` / observation: What the agent sees.
*   `reward`: Definition of success/progress.

## 5) Building with OpenEnv
*   Clean separation: Env handles dynamics/scoring; Trainer handles optimization; Model learns to act.
*   Use OpenEnv CLI for scaffolding.

## 6) Curriculum Learning
*   Start simple: Easy tasks with short horizons.
*   Escalate: Only move to harder tasks once the model gets non-zero rewards.

## 7) Reward Design
Your reward function is your task specification. Use **multiple independent reward functions** to prevent hacking:
*   Execution success
*   Correctness
*   Format compliance (Regex/JSON)
*   Timeouts / Resource usage
*   Safety constraints

## 8) Protect Against Reward Hacking
Failure mode: Model finds shortcuts to max reward without solving the task.
*   **Fixes**: Multiple independent signals, time limits, restricted execution, avoid unrestricted global state, periodic human inspection of generations.

## 9) Process-Aware Feedback
Use richer supervision for intermediate steps (process supervision) where possible:
*   Line-by-line checks
*   Step-level verifiers
*   LLM-as-a-judge for reasoning (but watch for gaming).

## 10) Training Stack
*   **TRL**: For RL algorithms (like GRPO).
*   **Unsloth**: For efficient training and inference.
*   **OpenEnv**: For standardized environment interaction.

## 11) GRPO / RLVR Style Training
*   Highly recommended for verifiable tasks.
*   Simplifies PPO by removing the value model.
*   Build the verifier first, then plug into RL.

## 12) Efficiency
*   Inference (rollout generation) is often the bottleneck.
*   Prioritize fast sampling and tight environment loops.

## 13) Early Deployment
*   Deploy to Hugging Face Spaces early to catch API and packaging issues.

## 14) Scale Later
*   Confirm `reset`, `step`, rewards, and timeouts work before increasing batch size or task diversity.

## 15) Monitoring
Monitor individual reward components (e.g., "function works" vs "total reward") and inspect actual generations.

## 16) Saving Models
*   **Warning**: Do not upcast 4-bit models to 16-bit and merge LoRA weights naively. Use the proper merged-save path or adapters directly.

## 17) Team Structure
*   **Person A**: Environment (reset/step/state, safety).
*   **Person B**: Verifier / Rewards (multiple functions, anti-hacking).
*   **Person C**: Training (TRL + Unsloth, metrics).
*   **Person D**: Demo / Product (HF Space, UI, benchmarks).

## 18) 1-Day Execution Plan
1.  **Phase 1-2**: Narrow task & Build Env.
2.  **Phase 3-4**: Build Rewards & Deploy.
3.  **Phase 5-6**: Train small & Inspect for hacking.
4.  **Phase 7-9**: Add curriculum, Train bigger, Save & Demo.

---
*Reference: Hackathon Self-Serve Build Guide.*
