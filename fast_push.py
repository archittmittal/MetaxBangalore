import os
from huggingface_hub import HfApi

SPACE_ID = "purvansh01/conflict-env"
HF_TOKEN = os.getenv("HF_TOKEN")

def fast_deploy():
    api = HfApi(token=HF_TOKEN)
    files_to_push = ["app.py", "README.md", "openenv.yaml", "Dockerfile", "requirements.txt"]
    
    print(f"Pushing CRITICAL files to {SPACE_ID}...")
    for f in files_to_push:
        if os.path.exists(f):
            print(f"  -> {f}")
            api.upload_file(
                path_or_fileobj=f,
                path_in_repo=f,
                repo_id=SPACE_ID,
                repo_type="space"
            )
    
    # Also push the conflict_env folder
    print("Pushing conflict_env/ folder...")
    api.upload_folder(
        folder_path="conflict_env",
        path_in_repo="conflict_env",
        repo_id=SPACE_ID,
        repo_type="space"
    )
    
    print("\nDONE! Critical files and core logic pushed.")

if __name__ == "__main__":
    fast_deploy()
