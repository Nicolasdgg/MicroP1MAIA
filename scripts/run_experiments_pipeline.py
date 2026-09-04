"""
Pipeline maestro para ejecución, evaluación comparativa y registro de experimentos en MLflow.
Ejecuta Random Forest, LightGBM y TinySleepNet (1D-CNN + BiLSTM) con partición estricta por sujeto.
"""

import sys
import time
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# Agregar raíz al sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data.loader import load_dataset_by_subjects
from src.features.spectral import extract_features_dataset
from src.models.classical import train_random_forest, train_lightgbm
from src.models.deep_learning import train_deep_sleep_net
from src.utils.mlflow_client import configure_mlflow

def main():
    print("=" * 70)
    print(" INICIANDO PIPELINE DE EXPERIMENTACIÓN - ENTREGA 2 (MAIA) ")
    print("=" * 70)
    
    data_dir = repo_root / "data" / "raw"
    models_dir = repo_root / "models"
    figures_dir = repo_root / "reports" / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Configurar MLflow
    tracking_uri = configure_mlflow()
    print(f"Tracking URI activo: {tracking_uri}")
    
    # 2. Cargar datos de polisomnografía (con caché para acelerar iteraciones)
    cache_path = repo_root / "data" / "processed" / "cache_dataset.npz"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    if cache_path.exists():
        print(f"\n[1/4] Cargando datos preprocesados desde caché: {cache_path.name}...")
        data_cache = np.load(cache_path)
        X_raw = data_cache["X_raw"]
        y = data_cache["y"]
        subjects = data_cache["subjects"]
        print(f"Total épocas cargadas desde caché: {len(y):,}")
    else:
        print("\n[1/4] Cargando registros y segmentando épocas de 30s...")
        t0 = time.time()
        X_raw, y, subjects = load_dataset_by_subjects(data_dir, max_records=6)
        print(f"Total épocas cargadas: {len(y):,} en {time.time() - t0:.1f} s")
        np.savez_compressed(cache_path, X_raw=X_raw, y=y, subjects=subjects)
        print(f"Datos guardados en caché: {cache_path.name}")
        
    print(f"Distribución de estadios: W={np.sum(y==0)}, N1={np.sum(y==1)}, N2={np.sum(y==2)}, N3={np.sum(y==3)}, REM={np.sum(y==4)}")
    
    # 3. Partición Estricta por Sujeto (GroupShuffleSplit Anti-Data Leakage)
    print("\n[2/4] Realizando partición por sujeto (Train / Test)...")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=42)
    train_idx, val_idx = next(gss.split(X_raw, y, groups=subjects))
    
    train_subjs = np.unique(subjects[train_idx])
    val_subjs = np.unique(subjects[val_idx])
    print(f"Sujetos Entrenamiento ({len(train_subjs)}): {list(train_subjs)}")
    print(f"Sujetos Validación/Test ({len(val_subjs)}): {list(val_subjs)}")
    
    # Datos para Deep Learning (señal cruda 3000 muestras)
    X_raw_train, y_train = X_raw[train_idx], y[train_idx]
    X_raw_val, y_val = X_raw[val_idx], y[val_idx]
    
    # 4. Extracción de características espectrales para Modelos Clásicos
    print("\n[3/4] Extrayendo características espectrales (Welch PSD, ratios, Hjorth)...")
    t0 = time.time()
    X_feat_train = extract_features_dataset(X_raw_train)
    X_feat_val = extract_features_dataset(X_raw_val)
    print(f"Matriz de características: {X_feat_train.shape} en {time.time() - t0:.1f} s")
    
    results = []
    
    # 5. Entrenamiento Modelo 1: Random Forest
    print("\n---> Entrenando Modelo 1A: Random Forest...")
    rf_model, rf_metrics = train_random_forest(
        X_feat_train, y_train, X_feat_val, y_val, 
        n_estimators=100, max_depth=12, figures_dir=figures_dir
    )
    results.append({"Modelo": "Random Forest", **rf_metrics})
    joblib.dump(rf_model, models_dir / "model_random_forest.pkl")
    
    # 6. Entrenamiento Modelo 2: LightGBM
    print("\n---> Entrenando Modelo 1B: LightGBM...")
    lgb_model, lgb_metrics = train_lightgbm(
        X_feat_train, y_train, X_feat_val, y_val, 
        n_estimators=150, learning_rate=0.05, figures_dir=figures_dir
    )
    results.append({"Modelo": "LightGBM", **lgb_metrics})
    joblib.dump(lgb_model, models_dir / "model_lightgbm.pkl")
    
    # 7. Entrenamiento Modelo 3: TinySleepNet (Deep Learning)
    print("\n---> Entrenando Modelo 2: TinySleepNet (1D-CNN + BiLSTM)...")
    dl_model, dl_metrics = train_deep_sleep_net(
        X_raw_train, y_train, X_raw_val, y_val, 
        epochs=8, batch_size=64, learning_rate=0.001, figures_dir=figures_dir
    )
    results.append({"Modelo": "TinySleepNet (CNN+BiLSTM)", **dl_metrics})
    import torch
    torch.save(dl_model.state_dict(), models_dir / "model_tinysleepnet.pt")
    
    # 8. Consolidar Tabla Comparativa
    df_results = pd.DataFrame(results)
    results_path = repo_root / "reports" / "tabla_comparativa_modelos.csv"
    df_results.to_csv(results_path, index=False)
    
    print("\n" + "=" * 70)
    print(" RESUMEN COMPARATIVO DE MODELOS SUPERVISADOS ")
    print("=" * 70)
    print(df_results[["Modelo", "accuracy", "f1_macro", "cohen_kappa", "f1_N1", "f1_N2", "f1_N3", "f1_REM"]].to_string(index=False))
    print("=" * 70)
    print(f"[OK] Experimentos registrados exitosamente. Resultados guardados en {results_path.name}")

if __name__ == "__main__":
    main()
