import uvicorn
from server.app import app

if __name__ == "__main__":
    print("Starting ConflictEnv Local Server on http://localhost:7860")
    print("This server is now ready to receive actions from the reasoning agent!")
    uvicorn.run(app, host="127.0.0.1", port=7860)
