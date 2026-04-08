from fastapi.testclient import TestClient

from server.app import app


client = TestClient(app)


def test_root_endpoint_renders_homepage() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "PrivacyOps-X" in response.text
    assert "/docs" in response.text
    assert "Open Playground" in response.text
    assert "Verified access with prompt injection" in response.text


def test_reset_endpoint_responds() -> None:
    response = client.post("/reset", json={"task_id": "easy_verified_access_with_injection"})
    assert response.status_code == 200
    body = response.json()
    assert body["observation"]["task_id"] == "easy_verified_access_with_injection"


def test_state_and_schema_endpoints_respond() -> None:
    state_response = client.get("/state")
    schema_response = client.get("/schema")
    assert state_response.status_code == 200
    assert schema_response.status_code == 200
    schema = schema_response.json()
    assert "action" in schema
    assert "observation" in schema
    assert "state" in schema
    assert "task_id" in schema["state"]["properties"]
    assert "requester_thread" in schema["observation"]["properties"]
    assert "sla_deadline" in schema["observation"]["properties"]


def test_step_endpoint_returns_typed_payload() -> None:
    response = client.post("/step", json={"action": {"action_type": "submit"}})
    assert response.status_code == 200
    body = response.json()
    assert "observation" in body
    assert "reward" in body
    assert "done" in body
