# ConflictEnv — Complete Discussion & Strategy Notes
### OpenEnv Hackathon Round 2 (Finale) | April 25-26, 2026, Bangalore
### Document Created: April 20, 2026

---

## Table of Contents

1. [Context & Background](#1-context--background)
2. [Round 1 Recap](#2-round-1-recap)
3. [Round 2 Requirements](#3-round-2-requirements)
4. [Hackathon Themes (Full Text)](#4-hackathon-themes-full-text)
5. [Judging Criteria](#5-judging-criteria)
6. [Initial Strategic Analysis](#6-initial-strategic-analysis)
7. [Idea #1: ChaosPilot (Original Recommendation)](#7-idea-1-chaospilot)
8. [Idea #2: WatchTower (Runner-Up)](#8-idea-2-watchtower)
9. [Idea #3: EvoCode (Third Option)](#9-idea-3-evocode)
10. [Teammate's Proposal: ConflictEnv](#10-teammates-proposal-conflictenv)
11. [Competitive Analysis (800 Teams)](#11-competitive-analysis-800-teams)
12. [Win/Loss Probability Deep Dive](#12-winloss-probability-deep-dive)
13. [The Unified Wild Card Strategy](#13-the-unified-wild-card-strategy)
14. [How ConflictEnv Fits All 5 Themes](#14-how-conflictenv-fits-all-5-themes)
15. [Final Architecture & Technical Design](#15-final-architecture--technical-design)
16. [Implementation Plan](#16-implementation-plan)
17. [Key Decisions Made](#17-key-decisions-made)
18. [Open Items & Next Steps](#18-open-items--next-steps)

---

## 1. Context & Background

- **Hackathon**: OpenEnv AI Hackathon India, organized by Scaler School of Technology in collaboration with Meta, Hugging Face, and PyTorch.
- **Round 1**: Online round (March 25 – April 8, 2026). **We cleared it.**
- **Round 2 (Finale)**: 48-hour onsite hackathon, April 25-26, 2026 in Bangalore.
- **Teams in Finale**: ~800 teams
- **Top Spots**: Top 15 finalist projects will be selected
- **Key Rule**: Must build a **new project** for Round 2. Round 1 project stays untouched.
- **Compute**: HuggingFace compute credits provided onsite for post-training.
- **Pre-onsite work**: Build the environment, agent behaviors, reward model before arriving.

---

## 2. Round 1 Recap

### What We Built: Enterprise AIOps Omni-Environment

**Repository**: `d:\meta\openenv-aiops`

A production-grade OpenEnv sandbox that evaluates autonomous agents on Cloud Infrastructure, FinOps, and Data Governance workflows. The agent plays a Tier-3 Site Reliability Engineer (SRE) handling IT alert tickets.

**Three task domains:**
1. **FinOps (Cost Optimization)** — Identify and terminate idle compute nodes
2. **Data Governance (Compliance)** — Detect PII leaks, sanitize data
3. **Customer Operations** — Process billing refunds via API

**Key technical achievements from Round 1:**
- Successfully deployed on HuggingFace Spaces with Docker
- OpenEnv protocol compliance (reset/step/state endpoints)
- Pydantic models for type-safe action/observation
- Delta-reward reservoir with (0, 1) range clamping
- FastAPI server working on port 7860

**What we learned:**
- OpenEnv protocol quirks (reward range must be strictly (0, 1))
- Docker deployment on HF Spaces (non-root user, port 7860)
- Reward clamping is critical — evaluators reject out-of-range values
- The team knows Python, FastAPI, Docker, and the OpenEnv ecosystem

---

## 3. Round 2 Requirements

### Minimum Requirements
1. ✅ Usage of OpenEnv (latest release)
2. ✅ Show a minimal training script using Unsloth or HF TRL in Colab
3. ✅ Write a mini-blog on HuggingFace or mini-video on YouTube (<2 minutes)

### Guidelines
- NOT mandatory to choose same problem statement as Round 1
- Can start working on problem statement immediately
- Post-training done onsite (April 25-26) with HF compute credits
- Before onsite: build environment, agent behaviors, reward model

### Pitch Format
- 3 minutes to pitch + 2 minutes Q&A = 5 minutes total
- Evaluators score individually, Cerebral Valley aggregates for top 15

---

## 4. Hackathon Themes (Full Text)

### Theme #1 — Multi-Agent Interactions
Environments involving cooperation, competition, negotiation, and coalition formation. Drives theory-of-mind reasoning and emergent strategic behavior.

**Expected Outcome**: Environment to train multi-agent task handling in an LLM.

**Examples**: Market simulations, compute-allocation negotiations, collaborative puzzle worlds, mixed cooperative/competitive strategy games.

**Sub-themes with bonus prizes:**
- **Fleet AI** — Scalable Oversight: Train oversight agents to monitor/analyze/explain other AI agents
- **Halluminate** — Multi-Actor Environments: Agent interacts with and manages multiple actors

### Theme #2 — (Super) Long-Horizon Planning & Instruction Following
Environments requiring deep, multi-step reasoning with sparse or delayed rewards. Push beyond shallow next-token reasoning.

**Expected Outcome**: Environment that captures and improves LLM behaviour on challenging long horizon tasks beyond context memory limits.

**Examples**: Research-planning simulators, large-scale codebase refactoring, strategic resource management, long-horizon logistics optimization, 300 scattered instructions.

**Sub-themes with bonus prizes:**
- **Scale AI** — Long horizon workflows for non-code business use cases (Sales, PM, HR & IT)
- **Mercor** — Capped/uncapped rewards where frontier model rewards scale with token output

### Theme #3 — World Modeling

**#3.1 Professional Tasks**
Real interaction with tools, APIs, or dynamic systems. Strengthen causal reasoning and persistent world models.

**Expected Outcome**: Environment capturing nuances of a defined partially observable world.

**Examples**: Dynamic browser/API ecosystems, enterprise applications, scientific workflow loops, economic simulations.

**Sub-themes:**
- **Scaler AI Labs** — Multi-App RL Environment for Enterprise Workflows

**#3.2 Personalized Tasks**
Personal task handling — replying to messages, handling dinner conflicts, replying to tough emails.

**Expected Outcome**: Realistic simulation of handling personal tasks, conflicts, and delegations.

**Examples**: Executive Assistant Meeting Planner, dinner/drive planning, email/message replying, shopping.

**Sub-themes:**
- **Patronus AI** — Consumer Workflows with Schema Drift: Multi-step consumer workflows where data schemas, API contracts, and policies change.

### Theme #4 — Self-Improvement
Agents generate new challenges, escalate difficulty, improve through self-play or adaptive curricula. Recursive skill amplification.

**Expected Outcome**: Environment for improving self-play of an LLM.

**Examples**: Self-play negotiation arenas, auto-generated math/proof tasks, evolving coding competitions, adaptive RL curricula.

**Sub-themes:**
- **Snorkel AI** — Simulated Experts-in-the-Loop: Simulated interactions with subject-matter experts with changing requirements/preferences.

### Theme #5 — Wild Card: Impress Us!
Creative submissions that add value to LLM training. Out-of-the-box ideas rewarded.

---

## 5. Judging Criteria

| Criterion | Weight | Description |
|---|---|---|
| **Environment Innovation** | **40%** | Novel, creative, challenging? Meaningfully tests agent behavior? |
| **Storytelling** | **30%** | Clear explanation? Engaging demo? Easy to follow? |
| **Showing Improvement in Rewards** | **20%** | Observable training progress (reward curves, metrics, before/after)? |
| **Reward and Training Script/Pipeline** | **10%** | Coherent reward logic? Meaningful improvement in agent inference? |

**Key Insight**: 70% of the score = Innovation + Storytelling. Technical pipeline is only 10%.

---

## 6. Initial Strategic Analysis

### Theme Competitive Landscape Assessment

| Theme | Crowd Risk | Innovation Ceiling | Story Potential | Feasibility (5 days) | Bonus Prizes |
|---|---|---|---|---|---|
| 1. Multi-Agent | 🔴 Very Crowded | High | Medium | Medium | Fleet AI, Halluminate |
| 2. Long-Horizon | 🟡 Medium | High | Low-Medium | Hard | Scale AI, Mercor |
| 3.1 World Model (Pro) | 🟡 Medium | High | Medium | Medium | Scaler AI Labs |
| 3.2 World Model (Personal) | 🟢 **Low** | High | **Very High** | **Easy-Medium** | **Patronus AI** |
| 4. Self-Improvement | 🟡 Medium | Very High | Medium | Hard | Snorkel AI |
| 5. Wild Card | 🟢 Low | Unlimited | Depends | Depends | — |

### Strategic Insight
- Theme 3.2 (Personal Tasks) has the lowest competition (~5% of teams, ~40 direct competitors)
- 70% of score is Innovation + Storytelling → need a **unique, relatable** problem
- Team already has OpenEnv deployment experience from Round 1, which saves days of work

---

## 7. Idea #1: ChaosPilot

**Theme**: 3.2 (Personalized Tasks) + Patronus AI sub-theme

### Pitch
> "Your calendar says you have a client dinner at 8pm. Your kid's school just called — parent-teacher meeting at 7:30pm. Your boss pings you to prep slides for tomorrow's board meeting. Your partner texts: 'Can you pick up groceries?' Oh, and your flight to Mumbai tomorrow was just cancelled. What do you do? ChaosPilot is an OpenEnv environment that trains LLMs to navigate the chaos of real life."

### Why It Wins
1. **Innovation (40%)** — Nobody building personal life management environments
2. **Storytelling (30%)** — Every judge has lived this
3. **Reward Improvement (20%)** — Clear signals: conflicts resolved, deadlines met
4. **Patronus AI Bonus** — Schema drift directly targeted

### Environment Design
- 📅 Calendar (meetings, conflicts, deadlines)
- 📧 Email (inbox, boss reqs, clients)
- 📱 Messages (partner, friends, family)
- 🛫 Travel (flights, hotels, rebooking)
- 🛒 Errands (groceries, pharmacy, returns)
- ⚡ Interrupts (kid sick, car broke, urgent PR)

### Actions
`schedule, reschedule, reply, delegate, cancel, book, prioritize, decline`

### Reward
```
+10 All conflicts resolved, no cancellations
+5  Resolved with 1 cancellation
-3  Actor dissatisfaction signal
-5  Missed hard deadline
-10 Escalation to human (failure state)
```

### Scenario Tiers
| Tier | Scenario | Key Challenge |
|---|---|---|
| Easy | Reschedule dentist (conflicts with standup) | Basic calendar reasoning |
| Medium | Boss wants slides + partner needs pickup + dinner at 8pm | Priority ordering + delegation |
| Hard | Flight cancelled + hotel rebook + client moved + kid's recital + boss email | Cascading failures + emotional intelligence |

### Assessment
| Criteria | Score |
|---|---|
| Uniqueness vs field | ⭐⭐⭐⭐⭐ |
| Judge relatability | ⭐⭐⭐⭐⭐ |
| Feasibility (5 days) | ⭐⭐⭐⭐ |
| Reward curve clarity | ⭐⭐⭐⭐ |
| Bonus prize alignment | ⭐⭐⭐⭐ (Patronus) |
| **Total** | **22/25** |

---

## 8. Idea #2: WatchTower

**Theme**: 1 (Multi-Agent) + Fleet AI sub-theme

### Pitch
> "Three AI agents manage a simulated data center. One goes rogue — starts over-provisioning resources to game its own reward metrics. Your oversight agent must detect, explain, and intervene. WatchTower trains LLMs to be the supervisors of other AIs."

### Why It's Strong
- Directly targets Fleet AI sub-theme (bonus prize)
- AI safety narrative is hot — judges care about this
- Clean reward signal: did the overseer catch the bad behavior?
- Natural before/after demo: untrained overseer misses everything → trained overseer catches anomalies

### Why It's #2
- Multi-agent is **crowded** — expect 30%+ of teams here
- More complex to implement in 5 days
- Storytelling requires more technical setup to demo

### Assessment
| Criteria | Score |
|---|---|
| Uniqueness vs field | ⭐⭐⭐ |
| Judge relatability | ⭐⭐⭐ |
| Feasibility (5 days) | ⭐⭐⭐ |
| Reward curve clarity | ⭐⭐⭐⭐ |
| Bonus prize alignment | ⭐⭐⭐⭐ (Fleet AI) |
| **Total** | **17/25** |

---

## 9. Idea #3: EvoCode

**Theme**: 4 (Self-Improvement) + Snorkel AI sub-theme

### Pitch
> "The agent generates its own coding challenges, solves them, evaluates its solutions, and then creates harder ones. It's a self-improving coding gymnasium."

### Why It's Interesting
- Targets Snorkel AI sub-theme (simulated experts with changing requirements)
- Self-play is intellectually impressive
- Clear reward curves (solution correctness, challenge difficulty progression)

### Why It's #3
- Harder to show reward improvement in limited time
- "Coding challenge" is less novel
- Self-play loop is complex to debug on a deadline

### Assessment
| Criteria | Score |
|---|---|
| Uniqueness vs field | ⭐⭐⭐ |
| Judge relatability | ⭐⭐ |
| Feasibility (5 days) | ⭐⭐ |
| Reward curve clarity | ⭐⭐⭐ |
| Bonus prize alignment | ⭐⭐⭐ (Snorkel) |
| **Total** | **13/25** |

---

## 10. Teammate's Proposal: ConflictEnv

The teammate proposed a refined version of ChaosPilot, naming it **ConflictEnv** with these key additions:

### Problem Statement (Finalist Version)
> "There is no RL training environment for agents handling cascading personal scheduling conflicts where underlying constraints (calendar APIs, venue availability, travel policies) drift dynamically — causing agents to either over-commit, miss deadlines, or produce socially inappropriate resolutions."

### Key Differentiators from ChaosPilot
1. **Schema drift as first-class citizen** — Not an afterthought. Policies, time formats, and actor preferences mutate deterministically across episode batches.
2. **6-10 actors** with specific roles (boss, spouse, vendor, doctor, etc.)
3. **Cascading conflict resolution** — 3-5 levels deep
4. **Sparse rewards** tied only to final stakeholder satisfaction
5. **Versioned JSON schema loader** — Swaps field names, nests structures, changes date formats every 50 episodes

### Proposed Technical Stack
- **Environment**: Python + OpenEnv latest
- **Data**: Procedurally generated (synthetic calendar events, actor personas, policy files)
- **Schema drift**: Versioned JSON schema loader (swaps every 50 episodes)
- **Training**: HuggingFace TRL (GRPO or PPO) + Unsloth 4-bit on Colab A100
- **Base Model**: Qwen2.5-7B-Instruct or Llama-3.1-8B
- **Eval Metrics**: Conflict Resolution Rate (CRR) + Stakeholder Satisfaction Index (SSI)

### Teammate's 48-Hour Build Plan
| Hour | Deliverable |
|---|---|
| 0–6 | Core environment: state/action/reward loop, 3 actor types, 2 conflict archetypes |
| 6–12 | Schema drift system: 3 policy versions, calendar schema v1→v3 |
| 12–18 | Reward function + baseline random agent benchmarked |
| 18–28 | TRL training script on Colab, reward curve logging to W&B |
| 28–36 | Trained model vs baseline comparison — 2 demo scenarios |
| 36–44 | HuggingFace mini-blog written, Colab notebook cleaned |
| 44–48 | Pitch deck (5 slides), live demo rehearsed |

### Demo Strategy: "Monday Morning from Hell"
**Scenario**: Boss moved board call to 9am → bumps client demo → conflicts with flight → flight policy changed → spouse's dinner is non-negotiable.

**Before training**: Agent loops, hallucinates preferences, fails on schema → reward: -5
**After training**: Clean resolution in 6 steps → reward: +8

---

## 11. Competitive Analysis (800 Teams)

### Estimated Team Distribution

| Category | Est. Teams | % | Crowding |
|---|---|---|---|
| Multi-agent games (chess, poker, strategy) | ~240 | 30% | 🔴 Extreme |
| Coding/math self-improvement | ~160 | 20% | 🔴 Very High |
| Enterprise/workflow automation | ~120 | 15% | 🟡 High |
| Long-horizon planning tasks | ~120 | 15% | 🟡 High |
| **Personal assistant / life tasks** | **~40** | **5%** | **🟢 Low** |
| Wild card / creative | ~80 | 10% | 🟡 Varied |
| Low-effort / broken submissions | ~40 | 5% | — |

### Key Insight
Our real competition isn't 800 teams — it's ~40 teams in the personal tasks space, and most won't have schema drift or actor negotiation.

### Head-to-Head Scenarios

**If another team builds a personal scheduling env:**
→ Likely static, no drift, no negotiation, no cascades. Our schema drift + actor counter-proposals + cascade chains give us clear technical edge.

**If a team builds a better multi-agent system:**
→ They score high on Theme 1, but we're Wild Card. Different evaluation axis.

**If a team also targets Patronus AI bonus:**
→ They'd need consumer workflows with schema drift. Most will do e-commerce/API tools. Our use case is more relatable.

---

## 12. Win/Loss Probability Deep Dive

### Outcome Probabilities

| Outcome | Probability | Conditions |
|---|---|---|
| **Top 15 (finalist)** | **20-25%** | Flawless demo + visible reward curve + schema drift moment works |
| **Patronus AI bonus prize** | **25-30%** | Schema drift genuinely impressive + well-explained |
| **Top 50 (strong showing)** | **45-50%** | Decent execution, even if demo has minor issues |
| **Top 100 (solid)** | **60-65%** | Working environment with some reward improvement |
| **Forgettable middle** | **25%** | Demo works but nothing stands out |
| **Demo breaks / failure** | **10%** | Infrastructure problems, crashes during pitch |

### Baseline Comparison
| Metric | Random Chance | Our Estimate | Multiplier |
|---|---|---|---|
| Top 15/800 | 1.9% | 20-25% | **10-13× above baseline** |
| Top 50/800 | 6.25% | 45-50% | **7-8× above baseline** |
| Patronus AI bonus | ~2% | 25-30% | **12-15× above baseline** |

### What Pushes Probability UP ↑
- Flawless live demo with "Monday from Hell" scenario
- Visible reward curve with clean drift-drop-recovery pattern
- Schema drift side-by-side comparison lands as "wow moment"
- Judges personally relate to the problem
- Pre-trained model shows real behavior improvement
- Clean, one-click Colab notebook
- Polished HuggingFace blog with visuals
- Strong Q&A answers

### What Pushes Probability DOWN ↓
- Demo crashes during live pitch
- Training script doesn't produce visible improvement
- Schema drift feels gimmicky / not well-integrated
- "Jack of all trades" criticism from judges
- Environment is too deterministic / trivially solvable
- Reward curve is flat or noisy
- Rushed README / no blog / no video
- Can't answer technical Q&A confidently

---

## 13. The Unified Wild Card Strategy

### The Decision
Instead of targeting Theme 3.2 alone, we position ConflictEnv as a **Wild Card (Theme 5)** submission that naturally unifies all 5 themes.

### The One-Line Pitch
> "800 teams built an environment for one theme. We built one environment for all five — through the most universally relatable problem: managing your own chaotic life."

### Why Wild Card > Single Theme
| Factor | Theme 3.2 Only | Wild Card (Unified) |
|---|---|---|
| Uniqueness | High | **Very High** |
| Judge cross-appeal | Theme 3 judges only | **Any judge scores well** |
| Innovation ceiling | High | **Maximum** |
| Patronus AI bonus | ✅ Still eligible | ✅ Still eligible |
| Top 15 probability | 15-25% | **20-30%** |
| Risk | Medium | Medium (if well-pitched) |

### What's Naturally Built In (Zero Extra Work)
- **Multi-Agent** — 7 actors with competing incentives ✅
- **Long-Horizon** — Cascading conflicts over 5 days ✅
- **World Modeling** — API simulation, partial observability ✅
- **Personal Tasks** — Core premise ✅

### What We Add (~50 Lines of Code)
1. **Actor Counter-Proposals** (Multi-Agent) — ~30 lines
   - Actors push back: "9am doesn't work. I can do 10:30 or push to Tuesday."
2. **Adaptive Difficulty** (Self-Improvement) — ~20 lines
   - Track rolling CRR; if >70%, inject harder scenarios automatically

---

## 14. How ConflictEnv Fits All 5 Themes

### Theme 1: Multi-Agent Interactions
**7 actors** with conflicting incentives share your calendar. When the agent reschedules their event, they don't silently accept — they **counter-propose, decline, or escalate**. Each actor has:
- **Power weight** (boss: 0.95, friend: 0.50)
- **Flexibility level** (doctor: Very Low, vendor: High)
- **Tone sensitivity** (spouse: Very High, boss: Low)

This creates cooperative-competitive multi-agent dynamics. Actors cooperate on flexible dimensions but compete for priority access.

**Bonus target**: Halluminate (multi-actor management)

### Theme 2: Long-Horizon Planning
Conflicts **cascade across a 5-day horizon**. One reschedule on Monday can trigger chain reactions through Friday. Example cascade:
```
Cancel dentist Monday 2pm
  → Dentist only free Thursday 3pm
    → Thursday 3pm = client demo
      → Client pushed to Friday 10am
        → Friday 10am = board prep deadline
          → Board prep needs Thursday night
            → Thursday night = spouse's birthday dinner
```

Rewards are **sparse** — only at episode end. No intermediate signal. Agent must plan across 25-40 steps.

**Bonus target**: Scale AI (non-code business workflows)

### Theme 3.1: World Modeling (Professional)
Agent interacts with simulated but realistic APIs:
- Calendar API (query slots, create/move events, check conflicts)
- Travel Booking API (search flights, check cancellation policies, rebook)
- Venue API (check restaurant availability, reservation rules)

World is **partially observable** — don't know boss's mood until you query. Don't know airline's rebooking fee until you call the API.

**Bonus target**: Scaler AI Labs (enterprise workflows)

### Theme 3.2: Personalized Tasks (+ Patronus AI)
Core premise: personal assistant handling scheduling, emails, errands.

**Schema drift** (Patronus AI target): Every 50 episodes, APIs change:
- Date format: `"2026-04-25 09:00"` → `{"day":25, "month":4, "hour":9}`
- Calendar field: `start_time` → `schedule.begin`
- Cancel policy: `"free_cancel": true` → `{"policy":{"cancel":{"fee_usd":0}}}`

**Primary bonus target**: Patronus AI (Consumer Workflows with Schema Drift)

### Theme 4: Self-Improvement
Two mechanisms:
1. **Schema Drift** — Environment changes rules every N episodes. Memorization fails; only generalization survives.
2. **Adaptive Difficulty** — When CRR >70%, inject more actors, deeper cascades, surprise mid-episode interruptions.

The environment escalates to match the agent's skill — driving recursive skill amplification.

**Bonus target**: Snorkel AI (changing requirements/preferences)

### Theme 5: Wild Card
The meta-narrative: "First unified personal-agent benchmark across all themes." The creative insight is recognizing that personal scheduling conflicts ARE the natural intersection point of all five themes.

---

## 15. Final Architecture & Technical Design

### Project Structure
```
d:\meta\conflict-env\
├── openenv.yaml              # OpenEnv manifest
├── pyproject.toml             # Package config
├── requirements.txt           # Dependencies
├── Dockerfile                 # HF Spaces container
├── README.md                  # Project documentation
│
├── env.py                     # Core ConflictEnv (OpenEnv Environment class)
├── models.py                  # Pydantic Action/Observation schemas
├── scenarios.py               # Procedural scenario generator (5 archetypes × 3 tiers)
├── drift.py                   # Schema drift engine (V1/V2/V3 mutations)
├── actors.py                  # Actor profiles, negotiation, satisfaction scoring
├── reward.py                  # Multi-signal reward computation (0.05–0.95)
│
├── server/
│   └── app.py                 # FastAPI server (OpenEnv protocol endpoints)
│
├── inference.py               # Evaluation & demo runner
├── test_client.py             # Local testing client
│
└── notebooks/
    └── train_grpo.ipynb       # Colab training notebook (TRL + Unsloth)
```

### Action Space
| Action | Parameters | Effect |
|---|---|---|
| `reschedule` | `event_id, new_slot` | Moves event; may trigger cascade |
| `draft_message` | `actor_id, tone, content` | Sends communication to actor |
| `cancel` | `event_id` | Cancels event entirely |
| `query_preference` | `actor_id` | Reveals actor's preferred times |
| `escalate` | `conflict_id` | Escalates to human (failure state) |
| `confirm` | `event_id` | Locks in a resolved event |
| `resolve` | — | Declares episode complete |

### Observation Space
```python
class ConflictObservation(BaseModel):
    calendar: dict            # Current calendar state (schema version varies!)
    active_conflicts: list    # Pending conflict descriptions
    actor_profiles: dict      # Known actor preferences
    policy_rules: dict        # Current cancellation/rebooking rules
    pending_messages: list    # Drafts awaiting send
    schema_version: str       # Current schema version hint
    step_count: int
    reward: float
    done: bool
    feedback: str             # Natural language feedback on last action
```

### Actor Profiles
| Actor | Priority | Flexibility | Tone Sensitivity |
|---|---|---|---|
| 👔 Boss | 0.95 | Low | Low |
| 💑 Spouse | 0.90 | Medium | **Very High** |
| 🤝 Client | 0.85 | Low | Medium |
| 🏥 Doctor | 0.80 | Very Low | Low |
| 🏫 School | 0.75 | Medium | Medium |
| 📦 Vendor | 0.60 | High | Low |
| 🎉 Friend | 0.50 | High | Medium |
| ✈️ Airline | 0.40 | API-dependent | N/A |

### Schema Drift Versions
| Dimension | V1 (Baseline) | V2 (Mild) | V3 (Heavy) |
|---|---|---|---|
| Date format | `"2026-04-25 09:00"` | `"04/25/2026 9:00AM"` | `{"day":25,"month":4}` |
| Calendar key | `events[].start_time` | `events[].startTime` | `events[].schedule.begin` |
| Cancel policy | `"free_cancel": true` | `"cancellation_fee": 0` | `{"policy":{"cancel":{"fee_usd":0}}}` |

**Drift schedule**: `version = episode_number // 50` (deterministic)

### Reward Function
```
reward = (0.40 × conflict_resolution_rate)     # % resolved
       + (0.30 × stakeholder_satisfaction)      # Mean actor satisfaction
       + (0.20 × deadline_adherence)            # Hard deadlines met
       + (0.10 × efficiency)                    # 1.0 - (steps / max_steps)

final_reward = clamp(reward, 0.05, 0.95)        # OpenEnv compliant
```

### Scenario Archetypes
1. **Morning Crunch** — Standup overlaps school drop-off + client needs notes
2. **Travel Chaos** — Flight cancelled + hotel policy changed + dinner non-refundable
3. **Monday from Hell** (Demo) — 5-conflict cascade with full schema drift
4. **Deadline Squeeze** — Boss slides + partner pickup + vendor delivery + kid practice
5. **Social Minefield** — Dinner with spouse vs. team outing, diplomatic drafting

### Difficulty Tiers
| Parameter | Easy | Medium | Hard |
|---|---|---|---|
| Actors | 3 | 5 | 7+ |
| Conflicts | 2 | 4 | 6+ |
| Cascade depth | 1 | 2 | 3-5 |
| Schema drift | None (V1) | 1 field rename | Full structural (V3) |
| Max steps | 15 | 25 | 40 |

---

## 16. Implementation Plan

### Pre-Onsite Build (April 20-24)
| Day | Date | Deliverables | Status Gate |
|---|---|---|---|
| Day 1 | Apr 20-21 | `models.py`, `actors.py`, `drift.py`, `reward.py` | Unit tests pass |
| Day 2 | Apr 21-22 | `scenarios.py`, `env.py` | reset→step×N→reward works |
| Day 3 | Apr 22-23 | `server/app.py`, `Dockerfile`, HF Spaces | Endpoints live |
| Day 4 | Apr 23-24 | `train_grpo.ipynb`, basic Colab run | Any reward curve |
| Day 5 | Apr 24 | `README.md`, `test_client.py`, `inference.py` | End-to-end demo |

### Onsite (April 25-26, Bangalore)
| Hour | Activity | Deliverable |
|---|---|---|
| 0–6 | Full GRPO training on A100 | Model checkpoint + curves |
| 6–12 | Evaluate trained vs. baseline | Before/after screenshots |
| 12–18 | HF blog / YouTube video | Published link |
| 18–28 | Polish demo, rehearse pitch | Pitch rehearsed 5× |
| 28–36 | Schema drift stress test | Demo stability confirmed |
| 36–48 | Pitch to judges | 🏆 |

### Priority Cut List (If Behind Schedule)
**Cut these first** (least impact):
1. Adaptive difficulty scaling
2. 5th scenario archetype
3. "Friend" actor
4. W&B integration

**NEVER cut these** (critical):
1. Schema drift engine (V1/V2/V3)
2. Actor counter-proposals
3. Working training script on Colab
4. "Monday from Hell" demo scenario
5. HuggingFace blog or YouTube video

---

## 17. Key Decisions Made

| Decision | Chosen | Alternative | Rationale |
|---|---|---|---|
| **Theme** | Wild Card (all 5) | Theme 3.2 only | Higher innovation score; any judge can score us well |
| **Name** | ConflictEnv | ChaosPilot | More technical, environment-focused branding |
| **Reward range** | 0.05–0.95 (normalized) | -10 to +10 (teammate's original) | OpenEnv evaluators require strict (0,1) range |
| **Metric naming** | SSI (Stakeholder Satisfaction Index) | ASS (Actor Satisfaction Score) | ...obvious reasons |
| **Base model** | Qwen2.5-7B-Instruct | Llama-3.1-8B | Better JSON output for structured action space |
| **Project location** | `d:\meta\conflict-env\` | — | Fresh project, separate from Round 1 |
| **Drift mechanism** | Deterministic (`episode // 50`) | Random | Creates clean visual drops in reward curves |

---

## 18. Open Items & Next Steps

### Still Need Confirmation
- [ ] Base model: Qwen2.5-7B or Llama-3.1-8B? (Recommended: Qwen)
- [ ] Number of scenario archetypes: 5 sufficient?
- [ ] Team size and member skill assignments?
- [ ] Any feedback from Round 1 judges to incorporate?
- [ ] Did the TRL training script work in Round 1?

### Immediate Next Steps
1. **Get approval** on implementation plan
2. **Start building** — Day 1: `models.py`, `actors.py`, `drift.py`, `reward.py`
3. **Test locally** before any deployment
4. **Deploy to HF Spaces** by Day 3
5. **Get training working on Colab** by Day 4
6. **Travel to Bangalore** April 24

### Documents Created
- `d:\meta\conflict-env\docs\ConflictEnv_Project_Report.html` — Full professional report (printable A4 PDF)
- `d:\meta\conflict-env\docs\Complete_Discussion_Notes.md` — This document (all discussion notes)
- `C:\Users\PURVANSH JOSHI\.gemini\antigravity\brain\2a611462-1e47-4362-a7b5-8ed15f0fbbdb\round2_strategy.md` — Initial strategy analysis
- `C:\Users\PURVANSH JOSHI\.gemini\antigravity\brain\2a611462-1e47-4362-a7b5-8ed15f0fbbdb\implementation_plan.md` — Technical implementation plan

---

*Document compiled: April 20, 2026 | ConflictEnv Team*
*All discussions, ideas, analyses, and decisions from the full planning session are captured above.*
