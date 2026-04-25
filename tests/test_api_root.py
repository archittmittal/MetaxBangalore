"""Quick API test -- skips model loading, tests FastAPI routes with scripted agents."""
import os
os.environ["SKIP_MODEL"] = "1"

# Monkey-patch to skip model loading
import app as _app
_app.ML_AVAILABLE = False
_app.MODEL_LOADED = False
_app.elite_pipe = None

from fastapi.testclient import TestClient
import json

client = TestClient(_app.app)

print("--- FastAPI API Tests ---\n")

# 1. Health check
print("[1] GET /api/health")
r = client.get("/api/health")
print(f"  Status: {r.status_code}")
print(f"  Body: {json.dumps(r.json(), indent=2)}")
assert r.status_code == 200
print("  ✓ PASS\n")

# 2. Resolve conflict
print("[2] POST /api/resolve")
r = client.post("/api/resolve", json={"scenario": "Boss meeting at 9AM clashes with school drop-off"})
print(f"  Status: {r.status_code}")
data = r.json()
print(f"  Elite reward: {data['elite']['reward']}")
print(f"  Naive reward: {data['naive']['reward']}")
print(f"  Delta: {data['delta']}")
print(f"  Scenario: {data['scenario_name']}")
assert r.status_code == 200
assert "elite" in data and "naive" in data
print("  ✓ PASS\n")

# 3. Inspect
print("[3] GET /api/inspect")
r = client.get("/api/inspect")
print(f"  Status: {r.status_code}")
data = r.json()
print(f"  Scenario: {data['metadata']['scenario_name']}")
print(f"  Step count: {data['metadata']['step_count']}")
assert r.status_code == 200
print("  ✓ PASS\n")

# 4. Training
print("[4] GET /api/training")
r = client.get("/api/training")
print(f"  Status: {r.status_code}")
data = r.json()
print(f"  Model: {data['metrics']['base_model']}")
assert r.status_code == 200
print("  ✓ PASS\n")

# 5. Empty scenario
print("[5] POST /api/resolve (empty - should 400)")
r = client.post("/api/resolve", json={"scenario": ""})
print(f"  Status: {r.status_code}")
assert r.status_code == 400
print("  ✓ PASS\n")

print("=" * 40)
print("All API tests passed! Ready to deploy.")
