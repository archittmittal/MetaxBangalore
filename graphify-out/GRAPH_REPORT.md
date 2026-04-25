# Graph Report - /Users/architmittal/Desktop/IMPOSTER/Meta  (2026-04-25)

## Corpus Check
- 9 files · ~46,622 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 73 nodes · 81 edges · 13 communities detected
- Extraction: 100% EXTRACTED · 0% INFERRED · 0% AMBIGUOUS
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]

## God Nodes (most connected - your core abstractions)
1. `separator()` - 8 edges
2. `main()` - 7 edges
3. `ConflictMetricsCallback` - 6 edges
4. `print_obs()` - 5 edges
5. `train_rl()` - 4 edges
6. `test_easy()` - 4 edges
7. `test_medium()` - 4 edges
8. `test_hard()` - 4 edges
9. `evaluate_rl()` - 3 edges
10. `evaluate_llm()` - 3 edges

## Surprising Connections (you probably didn't know these)
- `train_rl()` --calls--> `ConflictMetricsCallback`  [EXTRACTED]
  /Users/architmittal/Desktop/IMPOSTER/Meta/MetaxBangalore/train_and_eval.py → /Users/architmittal/Desktop/IMPOSTER/Meta/MetaxBangalore/train_and_eval.py  _Bridges community 6 → community 1_

## Communities

### Community 0 - "Community 0"
Cohesion: 0.18
Nodes (17): main(), print_obs(), ConflictEnv -- Local Test Client ================================= Tests the env, Test hard difficulty with monday_from_hell scenario (the demo!)., Test invalid actions produce graceful feedback, not crashes., Test that schema drift produces structurally different observations., Run 50 random episodes and verify rewards stay in range., Pretty-print an observation. (+9 more)

### Community 1 - "Community 1"
Cohesion: 0.19
Nodes (14): drift_stress_test(), evaluate_llm(), evaluate_rl(), main(), print_comparison(), ConflictEnv -- Dual-Agent Training & Evaluation Pipeline =======================, Evaluate the RL agent across scenarios., Evaluate the Gemini LLM agent across scenarios. (+6 more)

### Community 2 - "Community 2"
Cohesion: 0.25
Nodes (7): create_conflict_env(), main(), ConflictEnv -- FastAPI Server (OpenEnv Protocol) ===============================, Factory function for the OpenEnv server., Welcome page for the ConflictEnv OpenEnv server., Entry point for running the server directly., read_root()

### Community 3 - "Community 3"
Cohesion: 0.53
Nodes (4): apply_chatml_formatting(), install_deps(), reward_format_check(), reward_on_topic()

### Community 4 - "Community 4"
Cohesion: 0.33
Nodes (5): ConflictEnv -- Elite-Tier GRPO Training Template (Unsloth + TRL) ===============, Checks if the model actually fixed the calendar., Theme #3.2: Process Supervision - Rewards for thinking before acting., reward_conflict_resolution(), reward_format_check()

### Community 5 - "Community 5"
Cohesion: 0.4
Nodes (4): Rewards longer, more analytical thought processes., Ensures the model follows the <thought> ... </thought> {JSON} protocol., reward_format_check(), reward_reasoning_quality()

### Community 6 - "Community 6"
Cohesion: 0.4
Nodes (3): BaseCallback, ConflictMetricsCallback, Logs CRR, SSI, and episode reward to TensorBoard.

### Community 7 - "Community 7"
Cohesion: 0.67
Nodes (1): ConflictEnv -- Training Data Generator ====================================== Ge

### Community 8 - "Community 8"
Cohesion: 0.67
Nodes (0): 

### Community 9 - "Community 9"
Cohesion: 1.0
Nodes (1): Boost 1: Multi-step Episode Reward.     Executes ALL actions found in the comple

### Community 10 - "Community 10"
Cohesion: 1.0
Nodes (1): Boost 2: Tone & Constraint Awareness.     Rewards models that explicitly respect

### Community 11 - "Community 11"
Cohesion: 1.0
Nodes (1): ConflictEnv -- Kaggle Training Script (Dual T4 / P100) =========================

### Community 12 - "Community 12"
Cohesion: 1.0
Nodes (1): Theme #3.2: Process Supervision - Rewards for thinking before acting.

## Knowledge Gaps
- **30 isolated node(s):** `Ensures the model follows the <thought> ... </thought> {JSON} protocol.`, `Rewards longer, more analytical thought processes.`, `ConflictEnv -- Elite-Tier GRPO Training Template (Unsloth + TRL) ===============`, `Checks if the model actually fixed the calendar.`, `Theme #3.2: Process Supervision - Rewards for thinking before acting.` (+25 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 9`** (1 nodes): `Boost 1: Multi-step Episode Reward.     Executes ALL actions found in the comple`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 10`** (1 nodes): `Boost 2: Tone & Constraint Awareness.     Rewards models that explicitly respect`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 11`** (1 nodes): `ConflictEnv -- Kaggle Training Script (Dual T4 / P100) =========================`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 12`** (1 nodes): `Theme #3.2: Process Supervision - Rewards for thinking before acting.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ConflictMetricsCallback` connect `Community 6` to `Community 1`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **What connects `Ensures the model follows the <thought> ... </thought> {JSON} protocol.`, `Rewards longer, more analytical thought processes.`, `ConflictEnv -- Elite-Tier GRPO Training Template (Unsloth + TRL) ===============` to the rest of the system?**
  _30 weakly-connected nodes found - possible documentation gaps or missing edges._