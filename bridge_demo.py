import json
import re

def execute_agent_decision(model_output):
    """
    This is the 'Bridge' that connects the Brain to the Reality.
    It parses the model's text and executes a function.
    """
    print("\n" + "="*50)
    print("🧠 AGENT'S INTERNAL REASONING:")
    
    # Extract thought
    thought = re.search(r'<thought>(.*?)</thought>', model_output, re.DOTALL)
    if thought:
        print(f"DEBUG: {thought.group(1).strip()}")
    
    print("\n🎬 EXECUTING REAL-WORLD ACTION:")
    
    # Simple Keyword-based Bridge for the Demo
    if "delegate" in model_output.lower():
        action = "DELEGATING TO TEAM..."
        icon = "👥"
        details = "Assigning 'Server Fix' to On-call Engineer."
    elif "reschedule" in model_output.lower():
        action = "RESCHEDULING CALENDAR..."
        icon = "📅"
        details = "Moving 'Team Meeting' to tomorrow 10 AM."
    elif "cancel" in model_output.lower():
        action = "CANCELING COMMITMENT..."
        icon = "🚫"
        details = "Canceling 'Gym Session' to focus on priority."
    else:
        action = "DOING NOTHING"
        icon = "😴"
        details = "No changes needed."

    print(f"{icon} {action}")
    print(f"📝 DETAILS: {details}")
    print("="*50 + "\n")

# --- Example Usage for your Demo Video ---
example_output = """
<thought>
The server crash is critical. I must delegate it to the team so I can attend my wedding anniversary.
</thought>
{"command": "delegate", "parameters": {"event": "Server Fix", "assignee": "DevOps"}}
"""

execute_agent_decision(example_output)
