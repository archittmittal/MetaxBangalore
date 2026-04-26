# ConflictEnv: Teaching a 1.5B Model to Navigate Human Conflict

**By Archit Mittal & Purvansh Joshi** | OpenEnv Hackathon India 2026

---

## TL;DR

We built **ConflictEnv**, a reinforcement learning environment where an AI agent must resolve scheduling conflicts involving real human social dynamics. Using **Group Relative Policy Optimization (GRPO)** on a tiny **Qwen-2.5-1.5B** model, we achieved near-optimal convergence ($R \approx 0.94$) in under 150 steps.

ConflictEnv treats **Social Capital** as a differentiable resource, forcing agents to optimize for both temporal constraints and human sentiment.

---

## 1. The Core Thesis: Scheduling as Social Negotiation

Traditional calendar optimization is a **Constraint Satisfaction Problem (CSP)**. However, real-world scheduling is a **Game Theoretic Negotiation**. Moving a meeting isn't just about finding an empty slot; it's about spending "social tokens."

**The Efficiency Gap**: Every knowledge worker loses 2–4 hours per week to scheduling conflicts.
**The Social Gap**: Moving an investor pitch is high-risk; moving a 1:1 with an intern is low-risk but requires empathy.

---

## 2. Mathematical Framework of ConflictEnv

ConflictEnv is defined as a **Partially Observable Markov Decision Process (POMDP)** where the reward signal is decomposed into four orthogonal vectors.

### 2.1 The Multi-Signal Reward Function
The reward $R$ for an action $a$ in state $s$ is computed as:

$$R(s, a) = \sum_{i \in \{CRR, SSI, DL, EFF\}} \omega_i \cdot \phi_i(s, a) + \Gamma_{reasoning} - \Lambda_{loop}$$

Where:
*   **$\phi_{CRR}$ (Conflict Resolution Rate)**: A binary indicator $\in \{0, 1\}$ representing if the primary overlap is cleared.
*   **$\phi_{SSI}$ (Stakeholder Satisfaction Index)**: Computed as $\frac{1}{N} \sum_{j=1}^N S_j$, where $S_j$ is the latent satisfaction of actor $j$.
*   **$\phi_{DL}$ (Deadline Adherence)**: Measures proximity to "Hard Deadlines" (e.g., flight departures).
*   **$\phi_{EFF}$ (Efficiency)**: Defined as $1 - \frac{t}{t_{max}}$, rewarding fewer steps.
*   **$\Gamma_{reasoning}$**: A +0.10 process supervision bonus for valid `<thought>` blocks.
*   **$\Lambda_{loop}$**: A -0.20 penalty for state oscillation.

### 2.2 Stakeholder Dynamics (Social Physics)
Each stakeholder $j$ has a social weight $w_j \in [0.1, 1.0]$. The satisfaction $S_j$ decays non-linearly with each reschedule:

$$S_j(t+1) = \max(0, S_j(t) - \eta \cdot w_j \cdot k)$$

Where $k$ is the number of times event $j$ has been moved. When $S_j \to 0$, the actor "burns out" and rejects all future commands, creating a terminal penalty.

---

## 3. The Optimization: Group Relative Policy Optimization (GRPO)

We utilized **GRPO**, a variant of PPO that removes the need for a separate Value Function (Critic) by calculating advantages relative to a group of completions.

### 3.1 The GRPO Objective
For a prompt $q$, we generate a group of $G$ outputs $\{o_1, o_2, ..., o_G\}$. The objective function $J(\theta)$ is:

$$J(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \left( \min \left( \rho_i(\theta) \hat{A}_i, \text{clip}(\rho_i(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta D_{KL}(\pi_\theta || \pi_{ref}) \right) \right]$$

Where:
*   **$\rho_i(\theta)$**: The importance sampling ratio $\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}$.
*   **$\hat{A}_i$ (Relative Advantage)**: Computed by normalizing rewards within the group:
    $$\hat{A}_i = \frac{R_i - \text{mean}(\{R_j\}_{j=1}^G)}{\text{std}(\{R_j\}_{j=1}^G) + \epsilon}$$

This approach allows the 1.5B model to "see" better alternatives within its own generations, accelerating convergence without a massive critic model.

---

## 4. Training Evidence & Results

### 4.1 Emergent Curriculum
We observed that the model follows a natural hierarchical learning path:
1.  **Format Mastery** ($t < 25$): Learning the `<thought>` and JSON syntax.
2.  **Constraint Alignment** ($25 < t < 75$): Optimizing for $\phi_{CRR}$.
3.  **Social IQ Emergence** ($t > 75$): Learning to prioritize high-$w_j$ actors.

### 4.2 Quantitative Comparison

| Metric | Base Model (Qwen-1.5B) | GRPO-Trained Agent |
|---|---|---|
| Avg. Reward ($\bar{R}$) | 0.06 | **0.94** |
| JSON Validity | 0.0% | **100%** |
| Conflict Resolution (Hard) | 12% | **91%** |
| Social Burnout Rate | 88% | **4%** |

<img width="800" alt="Training Progress" src="https://huggingface.co/spaces/purvansh01/conflict-env/resolve/main/plots/reward_curve.png" />
*Figure 1: Exponential reward growth. The "kink" at step 50 corresponds to the model successfully bridging the gap between formatting and reasoning.*

---

## 5. Implementation Details

### V3.1 Guidance Update (Gradient Injection)
Initially, the model suffered from **Sparse Reward Collapse**. To fix this, we refined the System Prompt to include a **State-Action Schema**:

```text
Current State: Conflict detected between [Event A] and [Event B].
Stakeholders: Spouse (S=0.4, w=0.9), Boss (S=0.8, w=1.0).
Available Actions: [reschedule, cancel, query_preference, ...]
Requirement: You MUST think before acting.
```

By providing the latent variables ($S_j, w_j$) explicitly in the observation, we allowed the model to map tokens directly to the reward function's components.

---

## 6. Conclusion & Future Work

ConflictEnv proves that **Reasoning is the ultimate productivity tool**. By training on a 1.5B model, we demonstrate that specialized, high-density environments can induce "Social Intelligence" in LLMs far more efficiently than general-purpose pre-training.

### Future Research:
- **Differentiable Social Graphs**: Modeling multi-actor dependency as a graph neural network.
- **Latent Drift Adaptation**: Agents that infer API changes via few-shot error feedback.

---
**[Try the Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/purvansh01/conflict-env)**
**[View the OpenEnv Manifest](openenv.yaml)**

*Built with ❤️ for the OpenEnv Hackathon India 2026.*
