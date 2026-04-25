import re
import json

def reward_format_check(completions, **kwargs):
    """
    Ensures the model follows the <thought> ... </thought> {JSON} protocol.
    """
    responses = [c[0]["content"] for c in completions]
    rewards = []
    for response in responses:
        score = 0.0
        # 1. Check for thought tags
        if "<thought>" in response and "</thought>" in response:
            score += 15.0
        
        # 2. Check for JSON structure after thought
        # Look for the first '{' after the last '</thought>'
        try:
            thought_end = response.rfind("</thought>")
            if thought_end != -1:
                json_part = response[thought_end + len("</thought>"):].strip()
                if json_part.startswith("{") and json_part.endswith("}"):
                    # Optional: Verify it's actually valid JSON
                    json.loads(json_part)
                    score += 15.0
        except:
            pass # JSON was invalid
            
        rewards.append(score)
    return rewards

def reward_reasoning_quality(completions, **kwargs):
    """
    Rewards longer, more analytical thought processes.
    """
    responses = [c[0]["content"] for c in completions]
    rewards = []
    for response in responses:
        # Extract thought block
        match = re.search(r"<thought>(.*?)</thought>", response, re.DOTALL)
        if match:
            thought_content = match.group(1).strip()
            # Reward length (up to a limit) to encourage depth
            length_score = min(len(thought_content.split()) / 50.0, 5.0) 
            
            # Bonus for strategy keywords
            keywords = ["prioritize", "conflict", "deadline", "mitigate", "social"]
            keyword_score = sum(1.0 for k in keywords if k in thought_content.lower())
            
            rewards.append(length_score + keyword_score)
        else:
            rewards.append(0.0)
    return rewards
