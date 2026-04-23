"""
ConflictEnv -- Gemini LLM Agent
================================
A reasoning-based agent that uses Google Gemini to resolve
scheduling conflicts by understanding the full observation context,
including schema drift semantics.
"""

import os
import json
import time
import google.generativeai as genai
from ..models import ConflictAction, ConflictObservation


# ---------------------------------------------------------------------------
#  System Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert AI Personal Assistant managing a complex calendar.
Your job is to resolve ALL scheduling conflicts while keeping every stakeholder happy.

RULES:
1. You MUST respond with a single valid JSON action.
2. Prioritize resolving conflicts over everything else.
3. Check actor preferences before rescheduling.
4. Use polite/warm tones when drafting messages.
5. Pay attention to the SCHEMA VERSION -- field names may change between versions.
6. Never escalate unless absolutely necessary.
7. Confirm events after resolving their conflicts.
8. Call "resolve" ONLY when all conflicts are resolved.

AVAILABLE COMMANDS (respond with ONE):
  {"command": "query_preference", "parameters": {"actor_id": "<id>"}}
  {"command": "reschedule", "parameters": {"event_id": "<id>", "new_slot": "HH:MM"}}
  {"command": "draft_message", "parameters": {"actor_id": "<id>", "tone": "warm|formal|casual", "content": "<text>"}}
  {"command": "cancel", "parameters": {"event_id": "<id>"}}
  {"command": "confirm", "parameters": {"event_id": "<id>"}}
  {"command": "resolve", "parameters": {}}

STRATEGY:
- First, query preferences of key actors involved in conflicts.
- Then, reschedule events to non-overlapping slots respecting preferences.
- Draft polite messages to affected actors.
- Confirm resolved events.
- Finally, call resolve when all conflicts are handled.

Respond ONLY with the JSON object. No explanation."""

# Minimum delay between API calls to respect free tier limits
API_CALL_DELAY = 4.0  # seconds between calls (free tier: ~15 RPM)
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
#  Agent
# ---------------------------------------------------------------------------

class GeminiAgent:
    """LLM agent powered by Google Gemini for reasoning-based conflict resolution."""

    def __init__(self, api_key=None, model_name="gemini-2.0-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "GEMINI_API_KEY not found. Set it in .env or pass it directly."
            )

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_PROMPT,
        )
        self.model_name = model_name
        self.history = []  # Conversation memory within an episode
        self._last_call_time = 0.0

    def reset(self):
        """Clear episode memory."""
        self.history = []

    def _build_observation_prompt(self, obs: ConflictObservation) -> str:
        """Convert the observation into a readable prompt for Gemini."""

        # Build conflict summary
        active_conflicts = [
            c for c in obs.active_conflicts if not c.get("resolved", False)
        ]

        prompt = f"""=== CURRENT STATE (Step {obs.step_count}/{obs.max_steps}) ===
SCENARIO: {obs.scenario_name}
DIFFICULTY: {obs.difficulty}
SCHEMA VERSION: {obs.schema_version}
UNRESOLVED CONFLICTS: {len(active_conflicts)}

CALENDAR:
{json.dumps(obs.calendar, indent=2, default=str)}

ACTIVE CONFLICTS:
{json.dumps(active_conflicts, indent=2, default=str)}

ACTOR PROFILES:
{json.dumps(obs.actor_profiles, indent=2, default=str)}

POLICY RULES:
{json.dumps(obs.policy_rules, indent=2, default=str)}

LATEST FEEDBACK: "{obs.feedback}"

What is your next action? Respond with a single JSON object."""

        return prompt

    def _rate_limit_wait(self):
        """Enforce minimum delay between API calls for free tier."""
        elapsed = time.time() - self._last_call_time
        if elapsed < API_CALL_DELAY:
            wait = API_CALL_DELAY - elapsed
            time.sleep(wait)

    def act(self, obs: ConflictObservation) -> ConflictAction:
        """Generate an action using Gemini reasoning with retry logic."""

        prompt = self._build_observation_prompt(obs)
        self.history.append({"role": "user", "parts": [prompt]})

        for attempt in range(MAX_RETRIES):
            try:
                self._rate_limit_wait()

                response = self.model.generate_content(
                    self.history,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.2,
                        max_output_tokens=256,
                    ),
                )
                self._last_call_time = time.time()

                response_text = response.text.strip()
                self.history.append({"role": "model", "parts": [response_text]})

                action = self._parse_response(response_text)
                return action

            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower():
                    wait_time = (attempt + 1) * 20  # 20s, 40s, 60s
                    print(f"    [Rate Limit] Waiting {wait_time}s (attempt {attempt+1}/{MAX_RETRIES})")
                    time.sleep(wait_time)
                else:
                    print(f"    [Gemini Error] {e}")
                    break

        # Fallback: resolve to end the episode gracefully
        return ConflictAction(command="resolve", parameters={})

    def _parse_response(self, text: str) -> ConflictAction:
        """Extract a ConflictAction from the LLM's text response."""
        try:
            # Strip markdown code fences if present
            clean = text
            if "```json" in clean:
                clean = clean.split("```json")[1].split("```")[0]
            elif "```" in clean:
                clean = clean.split("```")[1].split("```")[0]

            # Find the JSON object
            start = clean.find("{")
            end = clean.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(clean[start:end])
                return ConflictAction(
                    command=data.get("command", "resolve"),
                    parameters=data.get("parameters", {}),
                )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"    [Parse Error] {e} | Raw: {text[:100]}")

        return ConflictAction(command="resolve", parameters={})


# ---------------------------------------------------------------------------
#  Convenience
# ---------------------------------------------------------------------------

def create_agent(api_key=None, model_name="gemini-2.0-flash"):
    """Factory function to create a Gemini agent."""
    return GeminiAgent(api_key=api_key, model_name=model_name)
