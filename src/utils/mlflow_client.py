"""
Módulo de cliente y utilidades de seguimiento para MLflow.
Soporta servidor remoto en AWS EC2 (54.235.50.255:5000) con fallback local robusto.
"""

import os
import urllib.request
from pathlib import Path
from typing import Dict, Any, Optional
import mlflow

DEFAULT_REMOTE_URI = "http://54.235.50.255:5000"
LOCAL_FALLBACK_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "Microproyecto_Sleep_Staging"

def is_remote_mlflow_available(uri: str = DEFAULT_REMOTE_URI, timeout: float = 3.0) -> bool:
    """Verifica si el servidor remoto de MLflow está respondiendo en la IP de EC2."""
    try:
        with urllib.request.urlopen(uri, timeout=timeout) as resp:
            return resp.status in (200, 302, 500)
    except Exception:
        return False

def configure_mlflow(
    preferred_uri: Optional[str] = None, 
    experiment_name: str = EXPERIMENT_NAME
) -> str:
    """
    Configura el tracking URI de MLflow (remoto en EC2 si está disponible, o local).
    Retorna la URI configurada activa.
    """
    target_uri = preferred_uri or os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_REMOTE_URI)
    
    if target_uri.startswith("http") and not is_remote_mlflow_available(target_uri):
        print(f"[Aviso MLflow] Servidor remoto {target_uri} no disponible. Usando fallback local: {LOCAL_FALLBACK_URI}")
        target_uri = LOCAL_FALLBACK_URI
    else:
        print(f"[MLflow] Conectado exitosamente al servidor de tracking: {target_uri}")
        
    mlflow.set_tracking_uri(target_uri)
    mlflow.set_experiment(experiment_name)
    return target_uri
