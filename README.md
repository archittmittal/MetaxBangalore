---
title: Conflict Env
emoji: 📈
colorFrom: green
colorTo: indigo
sdk: docker
pinned: false
---

# ConflictEnv: Dynamic Scheduling Conflict Benchmark

> **"Resolving cascading personal scheduling conflicts under massive schema drift."**

ConflictEnv is a high-fidelity Reinforcement Learning (RL) environment designed for training LLM-based personal assistants to handle complex, multi-agent scheduling conflicts. Built specifically for the **MetaxBangalore Hackathon** and targeting the **Patronus AI Bonus Prize**, it introduces dynamic **Schema Drift** that forces agents to generalize across evolving data structures.

---

## Mission Overview

In a world of constant schedule changes, traditional AI assistants fail when APIs change or conflicts cascade. ConflictEnv benchmarks an agent's ability to:
1.  **Negotiate**: Resolve overlaps across 8 distinct actor archetypes (Boss, Spouse, Client, etc.).
2.  **Adapt**: Navigate **3 Tiers of Schema Drift** (V1, V2, V3) where fields rename and structures nest.
3.  **Prioritize**: Balance hard deadlines, actor satisfaction, and resource constraints.

---

## Core Architecture

- **`conflict_env/`**: The heart of the benchmark—an OpenEnv-compliant package.
- **`scenarios.py`**: 5 archetypes (Morning Crunch, Travel Chaos, etc.) with deterministic seed support.
- **`actors.py`**: Multi-agent system where actors have unique "Flexibility" and "Tone Sensitivity."
- **`drift.py`**: The Schema Drift Engine that mutates observations in real-time.
- **`reward.py`**: Fine-grained reward system accounting for CRR (Conflict Resolution Rate) and SSI (Social Satisfaction Index).

---

## Protocol Compliance

ConflictEnv implements the **OpenEnv HTTP API Protocol**. We've optimized the server implementation to ensure robust performance across stateless and stateful requests:

- **Endpoint Registry**:
    - `POST /reset`: Initialize new scenarios with customized difficulty.
    - `POST /step`: Execute complex actions (reschedule, draft_message, resolve).
    - `GET /state`: Inspect the full environment state (Patched for Pydantic serialization).
    - `GET /health`: System integrity check.

> [!IMPORTANT]
> The server includes a robust monkey-patch for the `openenv-core` state handler to ensure that custom fields (like `obs`, `difficulty`, and `scenario_name`) are correctly serialized and returned in the `/state` endpoint.

---

## Getting Started

### 1. Installation
Ensure you have Python 3.8+ and install the dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the OpenEnv Server
Start the FastAPI server (Default port 7860):
```bash
python server/app.py
```

### 3. Verify Protocol Compliance
Run the automated verification suite to ensure all endpoints are aligned with the OpenEnv specification:
```bash
python tests/test_api.py
```

---

## Patronus AI Bonus Prize Features

- **Theme 1 (Multi-Agent)**: Agents must negotiate with multiple actors, each with varying "satisfaction" levels that impact the final reward.
- **Theme 3 (Schema Drift)**: Observations evolve from V1 (baseline) to V3 (deeply nested/renamed) over the course of training epochs, testing the agent's structural robustness.

---

## License
BSD-3-Clause

---
*Created for the MetaxBangalore Hackathon 2026.*
