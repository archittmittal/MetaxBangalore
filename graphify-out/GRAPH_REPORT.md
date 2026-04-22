# Graph Report - /Users/architmittal/Desktop/IMPOSTER/Meta/MetaxBangalore  (2026-04-22)

## Corpus Check
- 10 files · ~22,264 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 189 nodes · 487 edges · 12 communities detected
- Extraction: 53% EXTRACTED · 47% INFERRED · 0% AMBIGUOUS · INFERRED: 230 edges (avg confidence: 0.59)
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

## God Nodes (most connected - your core abstractions)
1. `ConflictEnv` - 53 edges
2. `ConflictAction` - 49 edges
3. `Actor` - 43 edges
4. `Scenario` - 30 edges
5. `ConflictObservation` - 29 edges
6. `ConflictState` - 28 edges
7. `run_before_after_demo()` - 10 edges
8. `compute_step_reward()` - 10 edges
9. `run_all_scenarios()` - 9 edges
10. `test_easy()` - 9 edges

## Surprising Connections (you probably didn't know these)
- `ConflictEnv -- Scenario Generator ================================== 5 scenario` --uses--> `Actor`  [INFERRED]
  /Users/architmittal/Desktop/IMPOSTER/Meta/MetaxBangalore/scenarios.py → /Users/architmittal/Desktop/IMPOSTER/Meta/MetaxBangalore/actors.py
- `Generate a scenario instance.      If archetype is None, picks one randomly.` --uses--> `Actor`  [INFERRED]
  /Users/architmittal/Desktop/IMPOSTER/Meta/MetaxBangalore/scenarios.py → /Users/architmittal/Desktop/IMPOSTER/Meta/MetaxBangalore/actors.py
- `Morning Crunch: Standup overlaps with school drop-off + client needs notes.` --uses--> `Actor`  [INFERRED]
  /Users/architmittal/Desktop/IMPOSTER/Meta/MetaxBangalore/scenarios.py → /Users/architmittal/Desktop/IMPOSTER/Meta/MetaxBangalore/actors.py
- `Travel Chaos: Flight cancelled + hotel reschedule + dinner non-refundable.` --uses--> `Actor`  [INFERRED]
  /Users/architmittal/Desktop/IMPOSTER/Meta/MetaxBangalore/scenarios.py → /Users/architmittal/Desktop/IMPOSTER/Meta/MetaxBangalore/actors.py
- `Monday from Hell (THE DEMO SCENARIO): 5-conflict cascade with full drift.     Bo` --uses--> `Actor`  [INFERRED]
  /Users/architmittal/Desktop/IMPOSTER/Meta/MetaxBangalore/scenarios.py → /Users/architmittal/Desktop/IMPOSTER/Meta/MetaxBangalore/actors.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.15
Nodes (35): Action, ConflictEnv, Environment, main(), naive_agent_step(), ConflictEnv -- Inference & Demo Runner ======================================= R, The main demo for the hackathon pitch.     Shows naive agent (before training) v, Show how the same scenario looks under V1, V2, and V3 schemas. (+27 more)

### Community 1 - "Community 1"
Cohesion: 0.08
Nodes (33): apply_drift(), _date_v1_to_v2(), _date_v1_to_v3(), get_drift_version(), ConflictEnv -- Schema Drift Engine =================================== The diffe, Transform conflict descriptors according to the schema version.      V1: {"confl, Apply schema drift to all relevant fields of an observation dict.     This is th, V1 -> V2: camelCase keys, US date format. (+25 more)

### Community 2 - "Community 2"
Cohesion: 0.11
Nodes (22): apply_satisfaction_delta(), compute_satisfaction_delta(), _default_actors(), Flexibility, generate_counter_proposal(), get_actors_for_difficulty(), ConflictEnv -- Actor System =========================== 8 actor archetypes with, Return a fresh copy of actors appropriate for the difficulty tier.      Easy: (+14 more)

### Community 3 - "Community 3"
Cohesion: 0.18
Nodes (17): _default_policy(), _gen_deadline_squeeze(), _gen_monday_from_hell(), _gen_morning_crunch(), _gen_social_minefield(), _gen_travel_chaos(), _make_conflict(), _make_event() (+9 more)

### Community 4 - "Community 4"
Cohesion: 0.18
Nodes (12): Declare the episode complete., clamp(), compute_crr(), compute_reward(), compute_ssi(), compute_step_reward(), ConflictEnv -- Reward Computation ================================= Multi-signal, Compute a small immediate reward delta for a single step.     This provides incr (+4 more)

### Community 5 - "Community 5"
Cohesion: 0.27
Nodes (10): Actor, A person or service the agent must negotiate with., Return the current environment state., OpenEnv-compatible RL environment for training LLMs to resolve     cascading per, Parse task_name into (difficulty, archetype).          Supports formats:, Check if a time slot is already occupied by an active event., Convert 'YYYY-MM-DD HH:MM' to total minutes., Reset the environment with a new scenario.          task_name can be: (+2 more)

### Community 6 - "Community 6"
Cohesion: 0.24
Nodes (9): create_conflict_env(), main(), ConflictEnv -- FastAPI Server (OpenEnv Protocol) ===============================, Factory function for the OpenEnv server., Entry point for running the server directly., Process an agent action and return the new observation., ConflictState, The internal state of the environment, used for protocol introspection. (+1 more)

### Community 7 - "Community 7"
Cohesion: 0.22
Nodes (7): Serialize for the observation (schema V1 format)., Return the cumulative reward for the current episode., Build the observation dict, applying schema drift., ConflictObservation, ConflictEnv -- Pydantic Action / Observation Schemas ===========================, The observation returned to the agent after each step., Observation

### Community 8 - "Community 8"
Cohesion: 0.33
Nodes (3): Cancel an event entirely. Significant satisfaction penalty., Confirm/lock in a resolved event., Check if any unresolved conflict involving the changed event is now resolved.

### Community 9 - "Community 9"
Cohesion: 0.33
Nodes (4): ConflictEnv -- Core OpenEnv Environment ========================================, Check if any pair of events overlap in time., state(), _time_to_minutes()

### Community 10 - "Community 10"
Cohesion: 0.33
Nodes (3): Query an actor's scheduling preferences., Escalate to a human. This is a failure state with severe penalty., Accumulate reward delta with safety clamping.

### Community 11 - "Community 11"
Cohesion: 0.67
Nodes (0): 

## Knowledge Gaps
- **31 isolated node(s):** `ConflictEnv -- Pydantic Action / Observation Schemas ===========================`, `An action taken by the LLM agent to resolve scheduling conflicts.`, `The observation returned to the agent after each step.`, `The internal state of the environment, used for protocol introspection.`, `ConflictEnv -- Schema Drift Engine =================================== The diffe` (+26 more)
  These have ≤1 connection - possible missing edges or undocumented components.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `ConflictEnv` connect `Community 0` to `Community 2`, `Community 4`, `Community 5`, `Community 6`, `Community 7`, `Community 8`, `Community 9`, `Community 10`?**
  _High betweenness centrality (0.379) - this node is a cross-community bridge._
- **Why does `Actor` connect `Community 5` to `Community 0`, `Community 2`, `Community 3`, `Community 4`, `Community 6`, `Community 7`, `Community 8`, `Community 9`, `Community 10`?**
  _High betweenness centrality (0.274) - this node is a cross-community bridge._
- **Why does `apply_drift()` connect `Community 1` to `Community 7`?**
  _High betweenness centrality (0.183) - this node is a cross-community bridge._
- **Are the 32 inferred relationships involving `ConflictEnv` (e.g. with `ConflictAction` and `ConflictObservation`) actually correct?**
  _`ConflictEnv` has 32 INFERRED edges - model-reasoned connections that need verification._
- **Are the 46 inferred relationships involving `ConflictAction` (e.g. with `ConflictEnv` and `ConflictEnv -- Core OpenEnv Environment ========================================`) actually correct?**
  _`ConflictAction` has 46 INFERRED edges - model-reasoned connections that need verification._
- **Are the 39 inferred relationships involving `Actor` (e.g. with `Scenario` and `ConflictEnv -- Scenario Generator ================================== 5 scenario`) actually correct?**
  _`Actor` has 39 INFERRED edges - model-reasoned connections that need verification._
- **Are the 22 inferred relationships involving `Scenario` (e.g. with `Actor` and `ConflictEnv`) actually correct?**
  _`Scenario` has 22 INFERRED edges - model-reasoned connections that need verification._