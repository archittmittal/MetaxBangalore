from gradio_client import Client
import json
import re

# Configuration
SPACE_ID = "purvansh01/conflict-env"
HF_TOKEN = os.getenv("HF_TOKEN")

def test_live_hf_demo():
    print(f"--- TESTING LIVE HUGGING FACE SPACE: {SPACE_ID} ---")
    try:
        # Fixed: Using 'token' instead of 'hf_token'
        client = Client(SPACE_ID, token=HF_TOKEN)
        
        scenario = "It is my 10th anniversary dinner tonight at 7 PM. But the production server just crashed at 6:45 PM. My boss is calling, and my spouse is waiting at the restaurant."
        
        print(f"Sending Scenario to Cloud Agent...")
        # Note: In most 'ConflictEnv' demos, we use a single input.
        # I will attempt to call the predict function.
        result = client.predict(
            scenario,
            api_name="/predict"
        )
        
        print("\nCLOUD AGENT RESPONSE:")
        print("-" * 50)
        print(result)
        print("-" * 50)
        
        if "<thought>" in str(result) and "{" in str(result):
            print("\nINTEGRATION SUCCESS: The cloud model is alive and reasoning!")
        else:
            print("\nPARTIAL SUCCESS: Received response but format was unexpected.")
            
    except Exception as e:
        print(f"\nINTEGRATION FAILED: {e}")

if __name__ == "__main__":
    test_live_hf_demo()
