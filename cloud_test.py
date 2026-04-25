import requests
import json
import re
import time

# Configuration
LOCAL_SERVER = "http://127.0.0.1:7860"
MODEL_ID = "purvansh01/conflict-env-final"
HF_TOKEN = os.getenv("HF_TOKEN")

def call_hf_router_api(sys_prompt, user_prompt):
    """Calls HF using the Global Router endpoint (OpenAI-compatible)."""
    # This is the modern, global router URL discovered in the docs
    api_url = "https://router.huggingface.co/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 600, # Increased for longer reasoning
        "temperature": 0.1
    }
    
    print(f"Calling Cloud Model via Global Router ({api_url})...")
    response = requests.post(api_url, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    elif response.status_code == 503:
        print("Model is loading, waiting 30 seconds...")
        time.sleep(30)
        return call_hf_router_api(sys_prompt, user_prompt)
    else:
        raise Exception(f"API Error ({response.status_code}): {response.text}")

def run_victory_test():
    print(f"Resetting Local Environment at {LOCAL_SERVER}...")
    try:
        reset_resp = requests.post(f"{LOCAL_SERVER}/reset", json={"task_name": "anniversary_dinner_conflict"})
    except:
        print("Local server NOT found. Make sure 'run_local_server.py' is running!")
        return

    obs = reset_resp.json()["observation"]
    print("\n" + "="*50)
    print("LOCAL SCENARIO LOADED:")
    print(obs)
    print("="*50 + "\n")

    sys_prompt = "You are an Elite Executive Assistant. You MUST start with <thought> reasoning and end with a JSON action."
    user_input = f"Scenario: Custom Scenario\nDetails: {obs}\n\nRespond in the following format: <thought> reasoning </thought> {{\"command\": \"...\", \"parameters\": {{...}}}}"
    
    response = call_hf_router_api(sys_prompt, user_input)
    
    print("\nMODEL REASONING & ACTION:")
    print("-" * 30)
    print(response)
    print("-" * 30)

    # Extract Action
    action_match = re.search(r"(\{.*?\})", response, re.DOTALL)
    if action_match:
        action_json = action_match.group(1).strip()
        print(f"Extracted Action: {action_json}")
        
        print(f"Sending Action to Local Environment...")
        step_resp = requests.post(f"{LOCAL_SERVER}/step", json=json.loads(action_json))
        
        if step_resp.status_code == 200:
            result = step_resp.json()
            print(f"Environment Result: {result['observation']}")
            state_resp = requests.get(f"{LOCAL_SERVER}/state")
            reward = state_resp.json().get("reward", 0)
            print(f"Final Reward: {reward}")
            
            if reward > 0.8:
                print("\n🏆 LOCAL TEST PASSED: Cloud model successfully controlled local environment!")
            else:
                print("\nTEST COMPLETE: Reward was lower than expected.")
        else:
            print(f"Local server rejected action: {step_resp.text}")
    else:
        print("No JSON action found in model response.")

if __name__ == "__main__":
    try:
        run_victory_test()
    except Exception as e:
        print(f"Error: {e}")
