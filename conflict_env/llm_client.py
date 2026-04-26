import os
import logging
from huggingface_hub import InferenceClient

logger = logging.getLogger("conflict_env.llm")

# Initialize client. In Hugging Face Spaces, HF_TOKEN is automatically available if set in Secrets.
hf_token = os.environ.get("HF_TOKEN")

try:
    client = InferenceClient(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        token=hf_token
    )
except Exception as e:
    logger.warning(f"Failed to initialize InferenceClient: {e}")
    client = None

def generate_actor_message(actor_name: str, actor_role: str, tone_sensitivity: str, satisfaction: float, proposed_slot: str, alternatives: list) -> str:
    """
    Generate a dynamic, in-character response from an actor rejecting a time slot.
    Returns None if the API fails, allowing a fallback to the hardcoded string.
    """
    if not client:
        return None

    # Determine emotional state based on satisfaction
    if satisfaction > 0.8:
        mood = "polite but firm"
    elif satisfaction > 0.4:
        mood = "annoyed"
    else:
        mood = "very frustrated and passive-aggressive"

    alt_str = " or ".join(alternatives)
    prompt = f"""You are {actor_name}, a {actor_role}. Your tone sensitivity is {tone_sensitivity}. Your current mood is {mood}.
The scheduling assistant just proposed a meeting at {proposed_slot}, but you cannot make it.
You MUST suggest {alt_str} instead.
Write a very short (1-2 sentences), in-character message rejecting the proposed time and suggesting the alternatives. Do not use hashtags or emojis. Keep it realistic."""

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
