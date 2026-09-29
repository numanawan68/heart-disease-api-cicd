import sys
from pathlib import Path

from fastapi.testclient import TestClient


ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app import app

client = TestClient(app)


# --------------------------------------------------
# Health Check Test
# --------------------------------------------------

def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True


# --------------------------------------------------
# Home Page Test
# --------------------------------------------------

def test_home_page():
    response = client.get("/")

    assert response.status_code == 200
    assert "Heart Disease Prediction" in response.text


# --------------------------------------------------
# Prediction Test
# --------------------------------------------------

def test_prediction():
    data = {
        "male": "male",
        "age": "45",
        "currentSmoker": "no",
        "cigsPerDay": "0",
        "BPMeds": "no",
        "prevalentStroke": "no",
        "prevalentHyp": "no",
        "diabetes": "no",
        "totChol": "200",
        "sysBP": "120",
        "diaBP": "80",
        "BMI": "24.5",
        "heartRate": "72",
        "glucose": "85"
    }

    response = client.post("/predict", data=data)

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "explanation" in result
    assert "shap_plot" in result


# --------------------------------------------------
# Prediction Output Test
# --------------------------------------------------

def test_prediction_output():
    data = {
        "male": "female",
        "age": "55",
        "currentSmoker": "yes",
        "cigsPerDay": "10",
        "BPMeds": "no",
        "prevalentStroke": "no",
        "prevalentHyp": "yes",
        "diabetes": "no",
        "totChol": "220",
        "sysBP": "140",
        "diaBP": "90",
        "BMI": "28.0",
        "heartRate": "78",
        "glucose": "100"
    }

    response = client.post("/predict", data=data)

    assert response.status_code == 200

    result = response.json()

    assert isinstance(result["prediction"], str)
    assert isinstance(result["explanation"], list)
    assert len(result["explanation"]) > 0
    assert result["shap_plot"] == "/static/shap_plot.png"


# --------------------------------------------------
# Missing Field Test
# --------------------------------------------------

def test_prediction_missing_field():
    data = {
        "male": "male",
        "age": "45",
        "currentSmoker": "no",
        "cigsPerDay": "0",
        "BPMeds": "no",
        "prevalentStroke": "no",
        "prevalentHyp": "no",
        "diabetes": "no",
        "totChol": "200",
        "sysBP": "120",
        "diaBP": "80",
        "BMI": "24.5",
        "heartRate": "72"
        # glucose intentionally missing
    }

    response = client.post("/predict", data=data)

    assert response.status_code == 422


# --------------------------------------------------
# Invalid Numeric Input Test
# --------------------------------------------------

def test_prediction_invalid_numeric_input():
    data = {
        "male": "male",
        "age": "invalid",
        "currentSmoker": "no",
        "cigsPerDay": "0",
        "BPMeds": "no",
        "prevalentStroke": "no",
        "prevalentHyp": "no",
        "diabetes": "no",
        "totChol": "200",
        "sysBP": "120",
        "diaBP": "80",
        "BMI": "24.5",
        "heartRate": "72",
        "glucose": "85"
    }

    response = client.post("/predict", data=data)

    assert response.status_code == 422

