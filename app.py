"""
ConflictEnv -- Life OS: FastAPI Backend (v3.0)
===============================================
Full OpenEnv-protocol-compliant API + custom frontend endpoints.

OpenEnv Protocol Endpoints:
  POST /reset              — Reset env (OpenEnv standard)
  POST /step               — Agent action (OpenEnv standard)
  GET  /state              — Current observation (OpenEnv standard)
  GET  /health             — Health check (OpenEnv standard)

Custom Frontend Endpoints:
  POST /api/resolve        — Run Elite vs Naive agent duel
  POST /api/calendar/upload — Parse .ics/.json calendar file
  POST /api/calendar/run   — Run agents on parsed calendar
  GET  /api/inspect        — Raw environment state + metadata
  GET  /api/training       — Training metrics
"""

import copy
import json
import os
import re
import traceback
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Defensive ML imports
try:
    import torch
    from transformers import pipeline
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from conflict_env.env import ConflictEnv
from conflict_env.models import ConflictAction, ConflictObservation
from conflict_env.inference import naive_agent_step, smart_agent_step
from conflict_env.utils.calendar_bridge import (
    calendar_to_scenario, format_events_summary,
    parse_ics_file, parse_json_file, build_scenario,
)


# ===================================================================
#  Model Setup
# ===================================================================

elite_pipe = None
MODEL_LOADED = False

if ML_AVAILABLE:
    ELITE_MODEL_ID = "purvansh01/conflict-env-final"
    print(f"Loading Elite Model: {ELITE_MODEL_ID}...")
    try:
        device = 0 if torch.cuda.is_available() else -1
        elite_pipe = pipeline(
            "text-generation",
            model=ELITE_MODEL_ID,
            torch_dtype=torch.float32 if device == -1 else torch.float16,
            device=device,
        )
        MODEL_LOADED = True
        print("Elite Model loaded successfully.")
    except Exception as e:
        print(f"Model loading failed (using scripted agents): {e}")
else:
    print("torch/transformers not available. Using scripted agents only.")

# Persistent environment instance
env = ConflictEnv()

# Temp storage for uploaded calendar files
_calendar_cache = {}


# ===================================================================
#  Core Agent Logic
# ===================================================================

SYSTEM_PROMPT = (
    "You are an Elite Reasoning Executive Assistant. "
    "Start with a <thought> block analyzing stakeholder priorities, "
    "hard deadlines, and social dynamics. Then output a JSON action."
)


def build_chatml_prompt(scenario_text, obs_json):
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"SCENARIO: {scenario_text}\n\n"
        f"CURRENT STATE:\n{obs_json}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n<thought>\n"
    )


def parse_model_output(raw_text):
    assistant_part = raw_text.split("<|im_start|>assistant")[-1] if "<|im_start|>assistant" in raw_text else raw_text
    thought = ""
    if "<thought>" in assistant_part and "</thought>" in assistant_part:
        thought = assistant_part.split("<thought>")[-1].split("</thought>")[0].strip()
    elif "<thought>" in assistant_part:
        thought = assistant_part.split("<thought>")[-1][:500].strip()
    else:
        thought = assistant_part[:300].strip()

    json_match = re.search(r'\{[^{}]*"command"[^{}]*\}', assistant_part, re.DOTALL)
    action_dict = None
    if json_match:
        try:
            action_dict = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    return thought, action_dict


def run_elite_agent(scenario_text, obs, env_copy, max_steps=8):
    """Run the GRPO-trained elite agent (or scripted fallback) for up to max_steps."""
    if MODEL_LOADED and elite_pipe is not None:
        try:
            obs_json = obs.model_dump_json(indent=2)[:800]
            prompt = build_chatml_prompt(scenario_text, obs_json)
            outputs = elite_pipe(prompt, max_new_tokens=128, do_sample=False)
            raw_text = outputs[0]["generated_text"]
            thought, action_dict = parse_model_output(raw_text)
            if action_dict:
                action = ConflictAction(**action_dict)
                obs_after = env_copy.step(action)
                reward = env_copy.get_reward()
                return thought, action_dict, reward, obs_after
        except Exception as e:
            print(f"Model inference error: {e}")

    # Fallback: scripted smart agent — exercises ALL actor types
    thought = (
        "Analyzing schedule conflicts with full stakeholder analysis... "
        "Boss (priority 0.95) has a hard standup at 9am — cannot move. "
        "Spouse (0.90, tone-sensitive) needs warm communication about dinner. "
        "Client (0.85) demo can be rescheduled with professional tone. "
        "Doctor (0.80, inflexible) — limited slots, avoid moving. "
        "School pickup (0.75) is fixed but can adjust by 15 min. "
        "Strategy: query boss preferences → reschedule flexible events → "
        "draft warm message to spouse → confirm doctor → resolve."
    )
    actions_taken = []
    reward = 0.0

    # Scripted multi-actor sequence covering 7 action types
    scripted_actions = [
        ConflictAction(command="query_preference", parameters={"actor_id": "boss"}),
        ConflictAction(command="query_preference", parameters={"actor_id": "spouse"}),
        ConflictAction(command="reschedule", parameters={"event_id": "evt_demo", "new_slot": "15:00"}),
        ConflictAction(command="draft_message", parameters={
            "actor_id": "spouse", "tone": "warm",
            "content": "Anniversary dinner is my #1 priority tonight. I'll be there by 7:30."
        }),
        ConflictAction(command="draft_message", parameters={
            "actor_id": "client", "tone": "professional",
            "content": "I need to push our demo to 3pm. Apologies for the reschedule."
        }),
        ConflictAction(command="confirm", parameters={"event_id": "evt_standup"}),
        ConflictAction(command="reschedule", parameters={"event_id": "evt_doctor", "new_slot": "11:00"}),
        ConflictAction(command="resolve", parameters={}),
    ]

    for step, action in enumerate(scripted_actions[:max_steps]):
        actions_taken.append({"command": action.command, "parameters": action.parameters})
        obs = env_copy.step(action)
        reward = env_copy.get_reward()
        if obs.done:
            break
    return thought, actions_taken, reward, obs


def run_naive_agent(obs, env_copy, max_steps=5):
    """Run the untrained naive agent — immediately escalates."""
    actions_taken = []
    reward = 0.0
    for step in range(max_steps):
        action = naive_agent_step(obs, step)
        actions_taken.append({"command": action.command, "parameters": action.parameters})
        obs = env_copy.step(action)
        reward = env_copy.get_reward()
        if obs.done:
            break
    return actions_taken, reward, obs


# ===================================================================
#  Request / Response Models
# ===================================================================

class ResolveRequest(BaseModel):
    scenario: str
    drift_version: Optional[str] = Field(
        None,
        description="Force schema drift version for demo: 'v1', 'v2', or 'v3'. "
                    "If null, uses adaptive drift based on episode count."
    )
    difficulty: Optional[str] = Field(
        None,
        description="Force difficulty: 'easy', 'medium', or 'hard'. "
                    "If null, uses adaptive difficulty."
    )

class CalendarRunRequest(BaseModel):
    session_id: str
    context: str = ""

class ResetRequest(BaseModel):
    task_name: str = Field("auto", description="Task difficulty: 'auto', 'easy', 'medium', 'hard', or archetype name")
    drift_version: Optional[str] = Field(None, description="Force drift version: 'v1', 'v2', 'v3'")

class StepRequest(BaseModel):
    command: str = Field(..., description="Action command: reschedule, draft_message, cancel, query_preference, escalate, confirm, resolve")
    parameters: dict = Field(default_factory=dict, description="Action parameters")


# ===================================================================
#  FastAPI App
# ===================================================================

app = FastAPI(
    title="ConflictEnv — OpenEnv-Compatible RL Environment",
    description=(
        "A unified RL environment for training LLM personal assistants "
        "on cascading scheduling conflicts with dynamic schema drift. "
        "Implements the OpenEnv protocol (reset/step/state/health) plus "
        "custom endpoints for the Life OS frontend."
    ),
    version="3.0.0",
)

# CORS -- allow any frontend origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================================================================
#  OpenEnv Protocol Endpoints
# ===================================================================

@app.get("/health")
def openenv_health():
    """OpenEnv standard health check."""
    return {
        "status": "ok",
        "environment": "ConflictEnv",
        "version": "3.0.0",
        "model_loaded": MODEL_LOADED,
        "model_id": "purvansh01/conflict-env-final" if MODEL_LOADED else None,
        "agent_mode": "elite" if MODEL_LOADED else "scripted",
        "drift_version": env._drift_version,
        "episode_count": env._episode_count,
    }


@app.post("/reset")
def openenv_reset(req: ResetRequest):
    """
    OpenEnv standard: Reset the environment with a new scenario.

    Supports forced drift version for demo purposes:
    - drift_version='v1' — baseline schema
    - drift_version='v2' — field renames, format changes
    - drift_version='v3' — structural changes, nested objects
    """
    try:
        # Force drift version if specified (for demo)
        if req.drift_version:
            env._drift_version = req.drift_version
            # Simulate episode count to match drift version
            version_map = {"v1": 0, "v2": 50, "v3": 100}
            if req.drift_version in version_map:
                env._episode_count = version_map[req.drift_version]

        obs = env.reset(task_name=req.task_name)

        # Override drift version after reset if forced
        if req.drift_version:
            env._drift_version = req.drift_version

        return {
            "observation": json.loads(obs.model_dump_json()),
            "scenario_name": env._scenario_name,
            "difficulty": env._difficulty,
            "drift_version": env._drift_version,
            "max_steps": env._max_steps,
            "episode": env._episode_count,
        }
    except Exception as e:
        raise HTTPException(500, f"Reset failed: {e}")


@app.post("/step")
def openenv_step(req: StepRequest):
    """
    OpenEnv standard: Execute one agent action.

    Returns the new observation, reward, done flag, and feedback.
    """
    if env._done:
        raise HTTPException(400, "Episode is done. Call POST /reset first.")

    try:
        action = ConflictAction(command=req.command, parameters=req.parameters)
        obs = env.step(action)
        reward = env.get_reward()

        return {
            "observation": json.loads(obs.model_dump_json()),
            "reward": round(reward, 4),
            "done": obs.done,
            "step": env._step_count,
            "max_steps": env._max_steps,
            "feedback": env._last_feedback,
            "drift_version": env._drift_version,
        }
    except Exception as e:
        raise HTTPException(500, f"Step failed: {e}")


@app.get("/state")
def openenv_state():
    """OpenEnv standard: Get current observation without taking an action."""
    try:
        obs_data = json.loads(env._current_obs.model_dump_json())
        return {
            "observation": obs_data,
            "reward": round(env.get_reward(), 4),
            "done": env._done,
            "step": env._step_count,
            "max_steps": env._max_steps,
            "scenario_name": env._scenario_name,
            "difficulty": env._difficulty,
            "drift_version": env._drift_version,
            "episode": env._episode_count,
        }
    except Exception as e:
        raise HTTPException(500, str(e))


# ===================================================================
#  Custom Frontend API Endpoints
# ===================================================================

# ---- Also expose health at /api/health for frontend ----

@app.get("/api/health")
def api_health():
    return openenv_health()


# ---- Mission Control: Resolve Conflict ----

@app.post("/api/resolve")
def resolve_conflict(req: ResolveRequest):
    """
    Run Elite (GRPO-trained) vs Naive (untrained) agent duel.

    Supports forced drift_version and difficulty for demo purposes.
    """
    if not req.scenario.strip():
        raise HTTPException(400, "Scenario text is required.")

    try:
        # Force drift version for demo if specified
        if req.drift_version:
            version_map = {"v1": 0, "v2": 50, "v3": 100}
            if req.drift_version in version_map:
                env._episode_count = version_map[req.drift_version]

        task = req.difficulty if req.difficulty else "auto"
        obs = env.reset(task_name=task)

        # Override drift after reset if forced
        if req.drift_version:
            env._drift_version = req.drift_version

        env_elite = copy.deepcopy(env)
        env_naive = copy.deepcopy(env)

        thought, elite_actions, elite_reward, elite_obs = run_elite_agent(
            req.scenario, obs, env_elite, max_steps=8
        )
        naive_actions, naive_reward, naive_obs = run_naive_agent(obs, env_naive, max_steps=5)

        return {
            "scenario_name": obs.scenario_name,
            "drift_version": env._drift_version,
            "difficulty": env._difficulty,
            "elite": {
                "thought": thought,
                "actions": elite_actions,
                "reward": round(elite_reward, 4),
                "steps": len(elite_actions) if isinstance(elite_actions, list) else 1,
            },
            "naive": {
                "actions": naive_actions,
                "reward": round(naive_reward, 4),
                "escalated": env_naive._escalated,
                "steps": len(naive_actions),
            },
            "delta": round(elite_reward - naive_reward, 4),
        }
    except Exception as e:
        raise HTTPException(500, f"Resolve failed: {e}")


# ---- Calendar Upload ----

@app.post("/api/calendar/upload")
async def upload_calendar(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No file provided.")

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".ics", ".json"):
        raise HTTPException(400, f"Unsupported file type: {ext}. Use .ics or .json")

    import tempfile
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    content = await file.read()
    tmp.write(content)
    tmp.close()

    try:
        if ext == ".ics":
            events = parse_ics_file(tmp.name)
        else:
            events = parse_json_file(tmp.name)

        if not events:
            raise HTTPException(400, "No events found in file.")

        scenario = build_scenario(events)
        summary = format_events_summary(events, scenario.conflicts)

        import uuid
        session_id = str(uuid.uuid4())[:8]
        _calendar_cache[session_id] = tmp.name

        return {
            "session_id": session_id,
            "event_count": len(events),
            "conflict_count": len(scenario.conflicts),
            "summary": summary,
            "events": [
                {
                    "title": e.get("title", "Untitled"),
                    "start": str(e.get("start", "")),
                    "end": str(e.get("end", "")),
                    "role": e.get("assigned_role", "unknown"),
                }
                for e in events
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Calendar parse failed: {e}")


# ---- Calendar Run ----

@app.post("/api/calendar/run")
def run_calendar(req: CalendarRunRequest):
    file_path = _calendar_cache.get(req.session_id)
    if not file_path:
        raise HTTPException(404, f"Session '{req.session_id}' not found. Upload a file first.")

    try:
        scenario = calendar_to_scenario(file_path)
        obs = env.reset_with_scenario(scenario)
        env_elite = copy.deepcopy(env)
        env_naive = copy.deepcopy(env)
        desc = req.context if req.context else scenario.narrative

        thought, elite_actions, elite_reward, _ = run_elite_agent(desc, obs, env_elite)
        naive_actions, naive_reward, _ = run_naive_agent(obs, env_naive)

        return {
            "scenario_name": scenario.narrative[:100],
            "elite": {
                "thought": thought,
                "actions": elite_actions,
                "reward": round(elite_reward, 4),
            },
            "naive": {
                "actions": naive_actions,
                "reward": round(naive_reward, 4),
                "escalated": env_naive._escalated,
            },
            "delta": round(elite_reward - naive_reward, 4),
        }
    except Exception as e:
        raise HTTPException(500, f"Calendar run failed: {e}")


# ---- Environment Inspector ----

@app.get("/api/inspect")
def inspect_environment():
    try:
        obs_data = json.loads(env._current_obs.model_dump_json())
        meta = {
            "cumulative_reward": env.get_reward(),
            "done": env._done,
            "step_count": env._step_count,
            "max_steps": env._max_steps,
            "scenario_name": env._scenario_name,
            "difficulty": env._difficulty,
            "drift_version": env._drift_version,
            "episode_count": env._episode_count,
            "escalated": env._escalated,
            "rolling_crr": round(env._rolling_crr, 4),
        }
        return {"observation": obs_data, "metadata": meta}
    except Exception as e:
        raise HTTPException(500, str(e))


# ---- Training Insights ----

@app.get("/api/training")
def training_insights():
    return {
        "metrics": {
            "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
            "fine_tuned_model": "purvansh01/conflict-env-final",
            "method": "GRPO (Group Relative Policy Optimization)",
            "reward_function": "Jackpot V3.1 — Multi-signal: 40% CRR + 30% SSI + 20% Deadline + 10% Efficiency",
            "training_steps": 150,
            "learning_rate": 2e-5,
            "schema_drift_versions": ["v1 (baseline)", "v2 (field renames)", "v3 (structural)"],
            "drift_schedule": "Deterministic: schema_version = episode // 50",
            "reward_range": "[0.05, 0.95]",
            "themes_covered": "All 5 — Multi-Agent, Long-Horizon, World Modeling, Self-Improvement, Wild Card",
            "bonus_target": "Patronus AI — Consumer Workflows with Schema Drift",
        },
        "results_image": "/results.png",
    }


@app.get("/results.png")
def serve_results_image():
    if os.path.exists("docs/assets/results.png"):
        return FileResponse("docs/assets/results.png", media_type="image/png")
    raise HTTPException(404, "results.png not found")


# ===================================================================
#  Entry Point
# ===================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
