# OpenEnv Hackathon India 2026: Themes & Judging Criteria

## 🏆 Hackathon Themes

### Theme #1 - Multi-Agent Interactions
Environments for this theme involve cooperation, competition, negotiation, and coalition formation. Learning from these environments will enable agents to model the beliefs and incentives of others in partially observable settings. This drives theory-of-mind reasoning and emergent strategic behavior.
*   **Expected Outcome**: An environment that can be used to train multi-agent task handling in a LLM.
*   **Example environments**: Market simulations, compute-allocation negotiations, collaborative puzzle worlds, mixed cooperative/competitive strategy games.

### Theme #2 - (Super) Long-Horizon Planning & Instruction Following
You will build environments that require deep, multi-step reasoning with sparse or delayed rewards. After using these environments, the goal is to enable agents to decompose goals, track state over extended trajectories, and recover from early mistakes. The aim is to push beyond shallow next-token reasoning toward structured planning and durable internal representations.
*   **Expected Outcome**: An environment that can capture and improve LLM behaviour on challenging long horizon tasks that need long running sessions beyond context memory limits.
*   **Example environments**: Research-planning simulators, large-scale codebase refactoring tasks, strategic resource management worlds, long-horizon logistics optimization, extremely complicated long-horizon instruction following.

### Theme #3 - World Modeling

#### #3.1 Professional Tasks
Develop environments that require real interaction with tools, APIs, or dynamic systems where the model is expected to do real hard work instead of exploiting short-cuts to arrive at the desired outcome. Learning from these environments will enable agents to maintain consistent internal state, update beliefs based on outcomes, and orchestrate multi-step workflows.
*   **Expected Outcome**: An environment capturing nuances of a defined partially observable world and improve LLM interaction with it.

#### #3.2 Personalized Tasks
Develop an environment that offers real personalized task handling, imagine replying to personal messages or handling dinner conflicts due to work conflicts, replying to tough emails. Think any personal assistant tasks.
*   **Expected Outcome**: An environment that gives the model a realistic simulation of handling personal tasks, conflicts and managing them as delegations.
*   **Example environments**: Executive Assistant Meeting Planner, Dinner and drive planning, email and message replying, shopping, etc.

### Theme #4 - Self-Improvement
The focus here is to create environments where agents can learn to generate new challenges, escalate difficulty, and improve through self-play or adaptive curricula. Rather than optimizing fixed tasks, the goal is for agents to learn to drive their own capability growth.
*   **Expected Outcome**: An environment for improving self-play of a LLM over a defined set of tasks.
*   **Example environments**: Self-play negotiation arenas, auto-generated math/proof tasks, evolving coding competitions, adaptive RL curricula.

### Theme #5: Wild Card - Impress Us!
We do not want to limit your focus if your idea doesn’t fit the boxes above. We want and WILL reward out of box tasks, please be creative but remember to add submissions that meaningfully add value to LLM training on a certain task.

---

## ⚖️ Judging Criteria

### Minimum Submission Requirements (Non-Negotiable)
1.  **Usage of OpenEnv**: Build on top of the framework (latest release).
2.  **Training Script**: A working training script using Unsloth or HF TRL in Colab.
3.  **Hugging Face Space**: Environment must be hosted on HF Spaces.
4.  **Writeup/Video**: A mini-blog on HF or a <2 minute video on YouTube.
5.  **README**: Motivates the problem, explains the env, and shows results with a link to the HF Space.

### Judging Overview
| Criterion | Weight | What Judges Look For |
| :--- | :---: | :--- |
| **Environment Innovation** | 40% | Novelty, creativity, and genuine challenge. Does it test behavior in a new way? |
| **Storytelling & Presentation** | 30% | Clear explanation of the problem/env. Engaging demo for non-technical audience. |
| **Showing Improvement** | 20% | Evidence of training progress (reward curves, before/after behavior). |
| **Reward & Training Pipeline** | 10% | Coherent reward logic and meaningful improvement in agent behavior. |

---

## 🎯 What Makes a Submission Stand Out
*   **Ambitious Problem**: Teach an LLM something it currently can't do well.
*   **Informative Reward Signal**: Use OpenEnv's Rubric system thoughtfully. Ensure rewards are hard to game.
*   **End-to-End Evidence**: Show real training curves (not just a script).
*   **Readable Plots**: Label axes, include units, and embed them in the README.
*   **Story-Driven README**: Focus on the *Why* and *Results* rather than just an API doc.
*   **Clean Engineering**: Respect client/server separation and Gym-style API.

---
*Reference: OpenEnv Hackathon India 2026 Guidelines.*
