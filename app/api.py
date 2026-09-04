"""
SomnoScope REST API - Servicio de Inferencia Clínica para Estadificación de Sueño.
Desarrollado con FastAPI para servir los modelos supervisados empaquetados.
"""

import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel, Field
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import mne

# Ruta base del repositorio
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data.loader import AASM_CLASSES, IDX_TO_CLASS, CLASS_TO_IDX
from src.features.spectral import extract_epoch_features, FEATURE_NAMES

app = FastAPI(
    title="SomnoScope Inference API",
    description="API REST para inferencia y estadificación automática de polisomnografía (AASM: W, N1, N2, N3, REM).",
    version="2.0.0"
)

# Permitir CORS amplio para consumo desde cualquier cliente o dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carga en memoria del modelo supervisado empaquetado
MODEL_PATH = repo_root / "models" / "best_sleep_model.pkl"
model = None
if MODEL_PATH.exists():
    try:
        model = joblib.load(MODEL_PATH)
        print(f"[OK] Modelo cargado exitosamente desde: {MODEL_PATH}")
    except Exception as e:
        print(f"[WARN] No se pudo cargar el modelo: {e}")

# ----------------- ESQUEMAS PYDANTIC -----------------
class FeaturesInput(BaseModel):
    features: List[float] = Field(..., description="Vector de 25 características espectrales y estadísticas")

class EpochSignalInput(BaseModel):
    signal: List[float] = Field(..., description="Muestras continuas de la época de 30 segundos (ej. 3000 puntos a 100 Hz)")
    sfreq: Optional[float] = Field(100.0, description="Frecuencia de muestreo en Hz")

class PredictionOutput(BaseModel):
    stage_idx: int
    stage_label: str
    confidence: float
    probabilities: Dict[str, float]

class RecordingPredictionOutput(BaseModel):
    total_epochs: int
    duration_hours: float
    tib_hours: float
    tst_hours: float
    sleep_efficiency_pct: float
    sol_min: float
    waso_min: float
    predicted_stages: List[str]
    stage_distribution: Dict[str, int]

# ----------------- ENDPOINTS -----------------
@app.get("/")
def read_root():
    return {
        "service": "SomnoScope REST API",
        "status": "online",
        "version": "2.0.0",
        "model_loaded": model is not None,
        "docs_url": "/docs"
    }

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return {
        "status": "healthy",
        "model": "LightGBM (Optimizado)",
        "classes": AASM_CLASSES,
        "n_features_expected": len(FEATURE_NAMES)
    }

@app.post("/predict/features", response_model=PredictionOutput)
def predict_from_features(data: FeaturesInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    feats = np.array(data.features).reshape(1, -1)
    if feats.shape[1] != len(FEATURE_NAMES):
        raise HTTPException(
            status_code=400,
            detail=f"Se esperaban {len(FEATURE_NAMES)} características, recibidas {feats.shape[1]}"
        )
    
    pred_idx = int(model.predict(feats)[0])
    probs = model.predict_proba(feats)[0]
    
    prob_dict = {AASM_CLASSES[i]: float(probs[i]) for i in range(len(AASM_CLASSES))}
    return PredictionOutput(
        stage_idx=pred_idx,
        stage_label=IDX_TO_CLASS[pred_idx],
        confidence=float(probs[pred_idx]),
        probabilities=prob_dict
    )

@app.post("/predict/epoch", response_model=PredictionOutput)
def predict_single_epoch(data: EpochSignalInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    sig = np.array(data.signal, dtype=np.float32)
    feats = extract_epoch_features(sig, sfreq=data.sfreq).reshape(1, -1)
    
    pred_idx = int(model.predict(feats)[0])
    probs = model.predict_proba(feats)[0]
    
    prob_dict = {AASM_CLASSES[i]: float(probs[i]) for i in range(len(AASM_CLASSES))}
    return PredictionOutput(
        stage_idx=pred_idx,
        stage_label=IDX_TO_CLASS[pred_idx],
        confidence=float(probs[pred_idx]),
        probabilities=prob_dict
    )

@app.post("/predict/recording", response_model=RecordingPredictionOutput)
async def predict_recording_file(
    file: UploadFile = File(...),
    target_channel: str = Form("EEG Fpz-Cz")
):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        available_ch = raw.ch_names
        ch_sel = target_channel if target_channel in available_ch else available_ch[0]
        raw.pick([ch_sel])
        
        if raw.info['sfreq'] != 100.0:
            raw.resample(100.0, verbose=False)
        raw.filter(0.5, 35.0, fir_design='firwin', verbose=False)
        
        n_samples = len(raw.times)
        epoch_len = 3000
        n_epochs = n_samples // epoch_len
        if n_epochs == 0:
            raise HTTPException(status_code=400, detail="El archivo no contiene épocas de 30s válidas.")
        
        data = raw.get_data()[0, :n_epochs * epoch_len]
        X_epochs = data.reshape(n_epochs, epoch_len).astype(np.float32)
        
        # Extracción y predicción en batch
        features = np.array([extract_epoch_features(x) for x in X_epochs])
        preds = model.predict(features)
        
        # Cálculo de arquitectura de sueño
        tib_hours = float(n_epochs * 30 / 3600)
        sleep_epochs = int(np.sum(preds != 0))
        tst_hours = float(sleep_epochs * 30 / 3600)
        eff = float((tst_hours / tib_hours) * 100) if tib_hours > 0 else 0.0
        
        first_sleep = np.where(preds != 0)[0]
        sol = float(first_sleep[0] * 0.5) if len(first_sleep) > 0 else 0.0
        waso_epochs = int(np.sum(preds[int(first_sleep[0]):] == 0)) if len(first_sleep) > 0 else 0
        waso = float(waso_epochs * 0.5)
        
        stage_labels = [IDX_TO_CLASS[p] for p in preds]
        dist = {c: int(np.sum(preds == CLASS_TO_IDX[c])) for c in AASM_CLASSES}
        
        return RecordingPredictionOutput(
            total_epochs=n_epochs,
            duration_hours=round(tib_hours, 2),
            tib_hours=round(tib_hours, 2),
            tst_hours=round(tst_hours, 2),
            sleep_efficiency_pct=round(eff, 1),
            sol_min=round(sol, 1),
            waso_min=round(waso, 1),
            predicted_stages=stage_labels,
            stage_distribution=dist
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando archivo EDF: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
