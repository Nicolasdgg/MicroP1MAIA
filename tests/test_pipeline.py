"""
Pruebas unitarias para el pipeline de procesamiento, extracción de características
y funciones de evaluación clínica de polisomnografía.
"""

import pytest
import numpy as np
import joblib
from pathlib import Path

from src.features.spectral import extract_epoch_features, FEATURE_NAMES, FREQ_BANDS
from src.models.evaluate import compute_sleep_metrics
from src.data.loader import AASM_CLASSES, CLASS_TO_IDX

def test_feature_extraction_shape():
    """Verifica que el extractor retorne exactamente las 25 características definidas."""
    dummy_signal = np.random.randn(3000).astype(np.float32)
    feats = extract_epoch_features(dummy_signal, sfreq=100.0)
    assert len(feats) == len(FEATURE_NAMES)
    assert len(feats) == 25
    assert not np.isnan(feats).any()
    assert not np.isinf(feats).any()

def test_spectral_relative_powers_sum():
    """Verifica que las potencias relativas sumen aproximadamente 1.0."""
    dummy_signal = np.random.randn(3000).astype(np.float32)
    feats = extract_epoch_features(dummy_signal, sfreq=100.0)
    # delta_rel (idx 6) a beta_rel (idx 10)
    rel_powers = feats[6:11]
    assert np.isclose(np.sum(rel_powers), 1.0, atol=1e-3)

def test_compute_sleep_metrics():
    """Verifica el cálculo de métricas clínicas estándar."""
    y_true = np.array([0, 1, 2, 3, 4, 2, 3, 0])
    y_pred = np.array([0, 1, 2, 3, 4, 2, 0, 0])
    metrics = compute_sleep_metrics(y_true, y_pred)
    
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert "cohen_kappa" in metrics
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert -1.0 <= metrics["cohen_kappa"] <= 1.0

def test_trained_model_inference():
    """Verifica que el modelo guardado genere predicciones coherentes en las 5 clases AASM."""
    model_path = Path("models/best_sleep_model.pkl")
    if model_path.exists():
        model = joblib.load(model_path)
        dummy_feats = np.random.randn(5, 25).astype(np.float32)
        preds = model.predict(dummy_feats)
        assert len(preds) == 5
        for p in preds:
            assert p in range(5)
