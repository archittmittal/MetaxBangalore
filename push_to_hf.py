import os
from huggingface_hub import HfApi

# Configuration
SPACE_ID = "purvansh01/conflict-env"
HF_TOKEN = os.getenv("HF_TOKEN")

def deploy_to_hf():
    api = HfApi(token=HF_TOKEN)
    
    print(f"STARTING TOTAL SYNC TO SPACE: {SPACE_ID}")
    
    # Upload everything in the current directory (respecting .hfignore)
    try:
        api.upload_folder(
            folder_path=".",
            repo_id=SPACE_ID,
            repo_type="space",
            delete_patterns=["*"], # Clean up any old mess in the space
            commit_message="Full Stack Sync: Merged Upstream + Reasoning Logic"
        )
        print("\nDEPLOYMENT COMPLETE! The Space is now building with the FULL stack.")
        print("Wait 2-3 minutes for the build to finish.")
    except Exception as e:
        print(f"\nDEPLOYMENT FAILED: {e}")

if __name__ == "__main__":
    deploy_to_hf()
