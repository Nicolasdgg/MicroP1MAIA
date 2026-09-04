"""
Pruebas unitarias y de integración para la API REST de inferencia clínica (FastAPI).
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from app.api import app, FEATURE_NAMES, AASM_CLASSES

client = TestClient(app)

def test_api_root():
    """Verifica que el endpoint raíz responda con estado online."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "SomnoScope REST API"
    assert data["status"] == "online"

def test_api_health():
    """Verifica el health check y que el modelo esté cargado."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["n_features_expected"] == 25

def test_api_predict_features():
    """Verifica la inferencia a partir de un vector de 25 características."""
    dummy_features = list(np.random.randn(25))
    response = client.post("/predict/features", json={"features": dummy_features})
    assert response.status_code == 200
    data = response.json()
    assert "stage_idx" in data
    assert data["stage_label"] in AASM_CLASSES
    assert 0.0 <= data["confidence"] <= 1.0
    assert len(data["probabilities"]) == 5

def test_api_predict_epoch():
    """Verifica la inferencia directa sobre una señal bioeléctrica continua (3000 puntos)."""
    dummy_signal = list(np.random.randn(3000))
    response = client.post("/predict/epoch", json={"signal": dummy_signal, "sfreq": 100.0})
    assert response.status_code == 200
    data = response.json()
    assert data["stage_label"] in AASM_CLASSES
    assert 0.0 <= data["confidence"] <= 1.0
