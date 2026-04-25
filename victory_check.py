import os, sys, re, json
# Ensure we can import conflict_env
sys.path.insert(0, os.getcwd())

from unsloth import FastLanguageModel
from conflict_env.env import ConflictEnv
from conflict_env.models import ConflictAction, VALID_COMMANDS

# Model name
model_name = "purvansh01/conflict-env-final"

print(f"🔄 Loading FINAL model: {model_name}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

def test_conflict_resolution():
    env = ConflictEnv()
    # Use the specific anniversary scenario
    obs = env.reset(task_name="anniversary_dinner_conflict")
    scenario_text = obs.observation
    
    print("\n" + "="*50)
    print("🔥 TESTING SCENARIO:")
    print(scenario_text)
    print("="*50 + "\n")
    
    sys_prompt = (
        "You are an Elite Executive Assistant. You MUST start with <thought> reasoning and end with a JSON action.\n"
        f"VALID COMMANDS: {', '.join(sorted(VALID_COMMANDS))}."
    )
    
    user_input = f"""Scenario: High Stakes Conflict
Details: {scenario_text}

Respond in the following format: <thought> reasoning </thought> {{"command": "...", "parameters": {{...}}}}"""

    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_input}
        ],
        tokenize=False, add_generation_prompt=True,
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=500, temperature=0.1) # Low temp for deterministic test
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    print("🤖 MODEL RESPONSE:")
    print("-" * 30)
    print(response)
    print("-" * 30)
    
    # Extract Action
    action_match = re.search(r"(\{.*?\})", response, re.DOTALL)
    if action_match:
        try:
            action_json = action_match.group(1).strip()
            action = ConflictAction.model_validate_json(action_json)
            print(f"✅ Parsed Command: {action.command}")
            
            # Step in the environment
            next_obs = env.step(action)
            reward = env.get_reward()
            
            print(f"🌍 Environment Feedback: {next_obs.observation}")
            print(f"💰 Environment Reward: {reward}")
            
            if reward > 0.8:
                print("\n🏆 TEST PASSED: Conflict Resolved Successfully!")
            else:
                print("\n⚠️ TEST COMPLETED: Partial Resolution or Low Reward.")
                
        except Exception as e:
            print(f"❌ Failed to parse or execute action: {e}")
    else:
        print("❌ No JSON action found in model response.")

if __name__ == "__main__":
    test_conflict_resolution()
