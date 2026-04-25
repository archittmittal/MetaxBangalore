# Graphify Analysis Output: ConflictEnv

This directory contains the structural analysis of the **ConflictEnv** codebase, generated via Graphify. ConflictEnv is a high-fidelity Reinforcement Learning environment designed for training LLMs to resolve cascading scheduling conflicts under schema drift.

## 📂 Directory Contents

| File | Description |
| :--- | :--- |
| [**GRAPH_REPORT.md**](./GRAPH_REPORT.md) | A detailed summary of the codebase, including detected communities, "God Nodes" (core abstractions), and structural insights. |
| [**graph.html**](./graph.html) | An interactive, browser-based visualization of the code dependency graph. |
| [**graph.json**](./graph.json) | The raw data representation of the graph (nodes, edges, and community assignments). |

---

## 🏗️ Architectural Overview

Based on the graph analysis, the system is organized into several distinct functional communities centered around the `ConflictEnv` bridge.

```mermaid
graph TB
    subgraph "Interface (Comm 6)"
        API["FastAPI / OpenEnv Server"]
        CreateEnv["create_conflict_env()"]
    end

    subgraph "Core Env (Comm 5 & 9)"
        Env["ConflictEnv (God Node)"]
        Reset[".reset()"]
        Step[".step()"]
        State[".state()"]
    end

    subgraph "Scenarios (Comm 3)"
        ScenGen["Scenario Generator"]
        ScenArch["Archetypes: Morning Crunch, Travel Chaos, etc."]
    end

    subgraph "Actor System (Comm 2)"
        ActorSys["Multi-Agent System"]
        Actors["8 Archetypes (Boss, Spouse, etc.)"]
        Satis["Satisfaction Tracking"]
    end

    subgraph "Schema Drift (Comm 1)"
        DriftEngine["Drift Engine"]
        ApplyDrift["apply_drift()"]
        Versions["V1 / V2 / V3 Mappings"]
    end

    subgraph "Rewards (Comm 4)"
        RewardComp["Reward Computation"]
        SSI["SSI (Social Satisfaction Index)"]
        CRR["CRR (Conflict Resolution Rate)"]
    end

    %% Relationships
    API --> CreateEnv
    CreateEnv --> Env
    Env --> Reset
    Env --> Step
    
    Reset --> ScenGen
    Step --> ActorSys
    Step --> DriftEngine
    Step --> RewardComp
    
    ScenGen --> ScenArch
    ActorSys --> Actors
    ActorSys --> Satis
    DriftEngine --> ApplyDrift
    RewardComp --> SSI
    RewardComp --> CRR

    %% God Node Highlights
    classDef godNode fill:#f96,stroke:#333,stroke-width:4px;
    class Env,Actors godNode;
```

---

## 🔍 Key Insights from the Graph

*   **Central Hubs**: `ConflictEnv` and `Actor` are the primary "God Nodes," serving as the connective tissue between the simulation logic, reward systems, and the external API.
*   **Modular Decoupling**: The **Schema Drift Engine** (Community 1) and **Reward Computation** (Community 4) are highly cohesive but modular, allowing for independent tuning of the benchmark's difficulty and scoring.
*   **Cascading Logic**: The analysis reveals strong inferred connections between `ConflictAction` and the `Scenario` archetypes, highlighting how agent decisions ripple through the multi-agent schedule.

---
*Analysis generated on 2026-04-22 for the MetaxBangalore Hackathon.*
