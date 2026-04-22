"""
ConflictEnv -- FastAPI Server (OpenEnv Protocol)
=================================================
Serves the ConflictEnv environment via HTTP endpoints.

Endpoints:
  POST /reset    -> ConflictObservation (scenario setup)
  POST /step     -> ConflictObservation (action result)
  GET  /state    -> ConflictObservation (current state)
  GET  /health   -> {"status": "ok"}
"""

import sys
import os

# Ensure parent directory is on the path so we can import env, models, etc.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Protocol Adjustment ---
# The base create_app hardcodes State as the response model for /state.
# To prevent stripping of custom fields, we monkey-patch the State class 
# in the openenv modules BEFORE create_app is called.
import openenv.core.env_server.types as openenv_types
import openenv.core.env_server.http_server as openenv_http
from openenv.core.env_server.http_server import create_app
from env import ConflictEnv
from models import ConflictAction, ConflictObservation, ConflictState
import uvicorn

openenv_types.State = ConflictState
openenv_http.State = ConflictState

def create_conflict_env() -> ConflictEnv:
    """Factory function for the OpenEnv server."""
    return ConflictEnv()


app = create_app(
    create_conflict_env,
    ConflictAction,
    ConflictObservation,
    env_name="conflict_env",
    max_concurrent_envs=1,
)


def main():
    """Entry point for running the server directly."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
