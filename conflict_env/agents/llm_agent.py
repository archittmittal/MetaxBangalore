import json
from ..models import ConflictAction, ConflictObservation

class BaseLLMAgent:
    """Base class for LLM-based agents."""
    def act(self, obs: ConflictObservation) -> ConflictAction:
        raise NotImplementedError

class SimplePromptAgent(BaseLLMAgent):
    """
    An agent that generates a prompt for an LLM and parses the response.
    Note: This class provides the prompt; the actual API call should be 
    implemented in the training/eval loop or a subclass.
    """
    
    def generate_prompt(self, obs: ConflictObservation) -> str:
        """Create a detailed prompt from the observation."""
        
        prompt = f"""
### CONFLICTENV MISSION CONTROL ###
You are an AI Personal Assistant resolving scheduling conflicts.

SCENARIO: {obs.scenario_name}
DIFFICULTY: {obs.difficulty}
SCHEMA VERSION: {obs.schema_version}
STEP: {obs.step_count}/{obs.max_steps}

CURRENT CALENDAR:
{json.dumps(obs.calendar, indent=2)}

ACTIVE CONFLICTS:
{json.dumps(obs.active_conflicts, indent=2)}

ACTOR PROFILES:
{json.dumps(obs.actor_profiles, indent=2)}

LATEST FEEDBACK:
"{obs.feedback}"

AVAILABLE COMMANDS:
- reschedule(event_id, new_slot)
- draft_message(actor_id, tone, content)
- cancel(event_id)
- query_preference(actor_id)
- confirm(event_id)
- resolve()

GOAL: Resolve all conflicts with high stakeholder satisfaction.
SCHEMA DRIFT WARNING: Be careful with field names. Check the SCHEMA VERSION hint.

OUTPUT FORMAT:
Respond with a single JSON object matching this schema:
{{"command": "...", "parameters": {{...}}}}
"""
        return prompt

    def parse_response(self, response_text: str) -> ConflictAction:
        """Parse the JSON response from the LLM."""
        try:
            # Simple extraction in case of conversational fluff
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            data = json.loads(response_text[start:end])
            return ConflictAction(**data)
        except Exception:
            # Fallback to resolve if parsing fails
            return ConflictAction(command="resolve", parameters={})

    def act(self, obs: ConflictObservation) -> ConflictAction:
        # In a real scenario, this would call the API.
        # For the baseline demo, we'll return a placeholder.
        print(self.generate_prompt(obs))
        return ConflictAction(command="resolve", parameters={})
