from fastapi.testclient import TestClient

from backend.main import app


def test_models_endpoint_defaults_to_from_scratch() -> None:
    client = TestClient(app)

    response = client.get("/models")

    assert response.status_code == 200
    data = response.json()
    assert data["active"] == "from-scratch"
    assert data["active_label"] == "Mon modèle (from scratch)"
    assert data["active_checkpoint"]
    assert {model["id"] for model in data["models"]} == {
        "from-scratch",
        "pretrained",
    }


def test_health_reports_active_model_without_loading_gpt() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["active_model"]["id"] == "from-scratch"
    assert data["active_model"]["checkpoint"]
