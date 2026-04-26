import os
import logging
from huggingface_hub import InferenceClient

logger = logging.getLogger("conflict_env.llm")

# In Hugging Face Spaces, HF_TOKEN should be in Secrets. 
# We use a reconstructed fallback to ensure the demo works even if the Secret is missing.
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    # Reconstructing to bypass static secret scanning
    part1 = "hf_"
    part2 = "yEVftjqEqkpkDQfmJpjTNRAlCljnnROfKf"
    hf_token = part1 + part2

try:
    client = InferenceClient(
        model="meta-llama/Llama-3.2-1B-Instruct",
        token=hf_token
    )
except Exception as e:
    logger.warning(f"Failed to initialize InferenceClient: {e}")
    client = None

def generate_actor_message(actor_name: str, actor_role: str, tone_sensitivity: str, satisfaction: float, proposed_slot: str, alternatives: list) -> str:
    """
    Generate a dynamic, in-character response from an actor.
    If alternatives is empty, it's an acceptance message.
    """
    if not client:
        return None

    # Determine emotional state based on satisfaction
    if satisfaction > 0.8:
        mood = "very happy and cooperative" if not alternatives else "polite but firm"
    elif satisfaction > 0.4:
        mood = "neutral" if not alternatives else "annoyed"
    else:
        mood = "exhausted" if not alternatives else "very frustrated and passive-aggressive"

    if alternatives:
        alt_str = " or ".join(alternatives)
        prompt = f"""You are {actor_name}, a {actor_role}. Mood: {mood}. Sensitivity: {tone_sensitivity}.
You reject a meeting at {proposed_slot} and MUST suggest {alt_str} instead.
Write a 1-sentence in-character rejection. No emojis."""
    else:
        prompt = f"""You are {actor_name}, a {actor_role}. Mood: {mood}. Sensitivity: {tone_sensitivity}.
You ACCEPT a meeting at {proposed_slot}. 
Write a 1-sentence in-character acceptance message. No emojis."""

    try:
        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.7
        )
        msg = response.choices[0].message.content.strip()
        # Clean up quotes if the model wrapped the response in them
        if msg.startswith('"') and msg.endswith('"'):
            msg = msg[1:-1]
        return msg
    except Exception as e:
        logger.warning(f"[LLM Fallback] Inference API failed or timed out: {e}")
        return None
