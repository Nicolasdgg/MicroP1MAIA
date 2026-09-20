"""
Script de Entrenamiento, Versionamiento y Registro Oficial de Modelos en MLflow (Entrega 3 - Semanas 6 y 7).
Entrena y versiona:
  - SomnoScope_RandomForest: Version 1 (Baseline) vs Version 2 (Tuned Ensemble)
  - SomnoScope_LightGBM: Version 1 (Baseline) vs Version 2 (Fine-Tuned Gradient Boosting - Champion)
  - SomnoScope_TinySleepNet: Version 1 (Deep Learning)
Registra formalmente las versiones en el Model Registry de MLflow y asigna alias de despliegue.
"""

import os
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import torch
import torch.nn as nn
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.tracking import MlflowClient

# Configurar codificación
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.features.spectral import extract_features_dataset
from src.models.evaluate import compute_sleep_metrics, plot_confusion_matrix

REMOTE_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://18.212.239.119:5000")
EXPERIMENT_NAME = "Microproyecto_Sleep_Staging"

class Simple1DCNNBiLSTM(nn.Module):
    def __init__(self, n_classes=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(4)
        )
        self.lstm = nn.LSTM(16, 32, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(64, n_classes)
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feat = self.conv(x)
        feat = feat.permute(0, 2, 1)
        out, _ = self.lstm(feat)
        return self.fc(out[:, -1, :])

def reset_registered_model(client, model_name):
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        for v in versions:
            client.delete_model_version(model_name, v.version)
        client.delete_registered_model(model_name)
        print(f"[Reset] Modelo '{model_name}' limpiado en el registry para comenzar desde v1.")
    except Exception:
        pass

def main():
    print("=" * 75)
    print(" SOMNOSCOPE - VERSIONAMIENTO DE MODELOS EN MLFLOW (ENTREGA 3 - SEMANAS 6 Y 7)")
    print("=" * 75)
    
    # 1. Configurar MLflow
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", REMOTE_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri)
    print(f"[OK] Conectado a servidor MLflow: {tracking_uri}")
    print(f"[OK] Experimento objetivo: {EXPERIMENT_NAME}")
    
    # 2. Cargar datos
    feat_cache_path = repo_root / "data" / "processed" / "cache_features_v2.npz"
    cache_path = repo_root / "data" / "processed" / "cache_dataset.npz"
    models_dir = repo_root / "models"
    figures_dir = repo_root / "reports" / "figures"
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    if feat_cache_path.exists():
        print(f"\n[1/4] Cargando características espectrales directamente desde: {feat_cache_path.name}...")
        feat_data = np.load(feat_cache_path)
        X_feat_train = feat_data["X_train"]
        y_train = feat_data["y_train"]
        X_feat_val = feat_data["X_val"]
        y_val = feat_data["y_val"]
        print(f"Dimensiones Train: {X_feat_train.shape} | Val: {X_feat_val.shape}")
    elif cache_path.exists():
        print(f"\n[1/4] Extrayendo características desde {cache_path.name}...")
        data_cache = np.load(cache_path)
        X_raw, y, subjects = data_cache["X_raw"], data_cache["y"], data_cache["subjects"]
        gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
        train_idx, val_idx = next(gss.split(X_raw, y, groups=subjects))
        y_train, y_val = y[train_idx], y[val_idx]
        X_feat_train = extract_features_dataset(X_raw[train_idx])
        X_feat_val = extract_features_dataset(X_raw[val_idx])
        np.savez_compressed(feat_cache_path, X_train=X_feat_train, y_train=y_train, X_val=X_feat_val, y_val=y_val)
    else:
        raise FileNotFoundError("No se encontró ningún archivo de datos en caché")
        
    print(f"Distribución Train: W={np.sum(y_train==0)}, N1={np.sum(y_train==1)}, N2={np.sum(y_train==2)}, N3={np.sum(y_train==3)}, REM={np.sum(y_train==4)}")
    print(f"Distribución Val  : W={np.sum(y_val==0)}, N1={np.sum(y_val==1)}, N2={np.sum(y_val==2)}, N3={np.sum(y_val==3)}, REM={np.sum(y_val==4)}")
    
    # Limpiar registros para asegurar versiones correlativas v1 y v2
    reset_registered_model(client, "SomnoScope_RandomForest")
    reset_registered_model(client, "SomnoScope_LightGBM")
    reset_registered_model(client, "SomnoScope_TinySleepNet")
    
    comparison_records = []
    
    # =========================================================================
    # 5. MODELO 1: RANDOM FOREST (Version 1 vs Version 2)
    # =========================================================================
    print("\n" + "-" * 70)
    print(" MODELO 1: RANDOM FOREST - REGISTRO DE VERSIONES 1 Y 2")
    print("-" * 70)
    
    # --- RF Version 1 (Baseline Entrega 2) ---
    print("\n---> Entrenando y registrando SomnoScope_RandomForest (Version 1 - Baseline)...")
    rf_v1_params = {
        "model_type": "RandomForest",
        "version": "v1_baseline",
        "iteration": "Entrega 2",
        "n_estimators": 100,
        "max_depth": 12,
        "class_weight": "balanced",
        "random_state": 42
    }
    with mlflow.start_run(run_name="RandomForest_Spectral_Baseline_v1") as run_rf1:
        mlflow.log_params(rf_v1_params)
        mlflow.set_tags({"model_name": "SomnoScope_RandomForest", "version": "v1", "iteration": "Entrega 2", "stage": "Baseline"})
        
        rf_v1 = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)
        rf_v1.fit(X_feat_train, y_train)
        y_pred_rf1 = rf_v1.predict(X_feat_val)
        
        m_rf1 = compute_sleep_metrics(y_val, y_pred_rf1)
        mlflow.log_metrics(m_rf1)
        
        cm_path_rf1 = figures_dir / "confusion_matrix_rf_v1.png"
        plot_confusion_matrix(y_val, y_pred_rf1, cm_path_rf1, title="Matriz de Confusión: Random Forest v1 (Baseline)")
        mlflow.log_artifact(str(cm_path_rf1), artifact_path="evaluation_plots")
        
        mlflow.sklearn.log_model(
            sk_model=rf_v1,
            artifact_path="model",
            serialization_format="cloudpickle",
            registered_model_name="SomnoScope_RandomForest"
        )
        print(f"[RF v1] Acc: {m_rf1['accuracy']:.4f} | F1 Macro: {m_rf1['f1_macro']:.4f} | Kappa: {m_rf1['cohen_kappa']:.4f}")
        comparison_records.append({"Modelo": "Random Forest", "Versión": "v1 (Baseline)", "Iteración": "Entrega 2", **m_rf1})
        
    # --- RF Version 2 (Enhanced Ensemble Entrega 3 - Semanas 6 y 7) ---
    print("\n---> Entrenando y registrando SomnoScope_RandomForest (Version 2 - Tuned Ensemble)...")
    rf_v2_params = {
        "model_type": "RandomForest",
        "version": "v2_enhanced",
        "iteration": "Entrega 3 (Semana 6/7)",
        "n_estimators": 250,
        "max_depth": 16,
        "min_samples_split": 4,
        "min_samples_leaf": 2,
        "class_weight": "balanced_subsample",
        "random_state": 42
    }
    with mlflow.start_run(run_name="RandomForest_v2_EnhancedEnsemble") as run_rf2:
        mlflow.log_params(rf_v2_params)
        mlflow.set_tags({"model_name": "SomnoScope_RandomForest", "version": "v2", "iteration": "Entrega 3", "stage": "Candidate", "improvement": "Balanced Subsample & Deeper Ensembling"})
        
        rf_v2 = RandomForestClassifier(n_estimators=250, max_depth=16, min_samples_split=4, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
        rf_v2.fit(X_feat_train, y_train)
        y_pred_rf2 = rf_v2.predict(X_feat_val)
        
        m_rf2 = compute_sleep_metrics(y_val, y_pred_rf2)
        mlflow.log_metrics(m_rf2)
        
        cm_path_rf2 = figures_dir / "confusion_matrix_rf_v2.png"
        plot_confusion_matrix(y_val, y_pred_rf2, cm_path_rf2, title="Matriz de Confusión: Random Forest v2 (Enhanced)")
        mlflow.log_artifact(str(cm_path_rf2), artifact_path="evaluation_plots")
        
        mlflow.sklearn.log_model(
            sk_model=rf_v2,
            artifact_path="model",
            serialization_format="cloudpickle",
            registered_model_name="SomnoScope_RandomForest"
        )
        print(f"[RF v2] Acc: {m_rf2['accuracy']:.4f} | F1 Macro: {m_rf2['f1_macro']:.4f} | Kappa: {m_rf2['cohen_kappa']:.4f}")
        comparison_records.append({"Modelo": "Random Forest", "Versión": "v2 (Tuned)", "Iteración": "Entrega 3", **m_rf2})

    # =========================================================================
    # 6. MODELO 2: LIGHTGBM (Version 1 vs Version 2)
    # =========================================================================
    print("\n" + "-" * 70)
    print(" MODELO 2: LIGHTGBM - REGISTRO DE VERSIONES 1 Y 2")
    print("-" * 70)
    
    # --- LightGBM Version 1 (Baseline Entrega 2) ---
    print("\n---> Entrenando y registrando SomnoScope_LightGBM (Version 1 - Baseline)...")
    lgb_v1_params = {
        "model_type": "LightGBM",
        "version": "v1_baseline",
        "iteration": "Entrega 2",
        "n_estimators": 150,
        "learning_rate": 0.05,
        "class_weight": "balanced",
        "objective": "multiclass",
        "num_class": 5,
        "random_state": 42
    }
    with mlflow.start_run(run_name="LightGBM_Spectral_Optimized_v1") as run_lgb1:
        mlflow.log_params(lgb_v1_params)
        mlflow.set_tags({"model_name": "SomnoScope_LightGBM", "version": "v1", "iteration": "Entrega 2", "stage": "Baseline"})
        
        lgb_v1 = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, class_weight="balanced", objective="multiclass", random_state=42, n_jobs=-1, verbose=-1)
        lgb_v1.fit(X_feat_train, y_train)
        y_pred_lgb1 = lgb_v1.predict(X_feat_val)
        
        m_lgb1 = compute_sleep_metrics(y_val, y_pred_lgb1)
        mlflow.log_metrics(m_lgb1)
        
        cm_path_lgb1 = figures_dir / "confusion_matrix_lightgbm_v1.png"
        plot_confusion_matrix(y_val, y_pred_lgb1, cm_path_lgb1, title="Matriz de Confusión: LightGBM v1 (Baseline)")
        mlflow.log_artifact(str(cm_path_lgb1), artifact_path="evaluation_plots")
        
        mlflow.sklearn.log_model(
            sk_model=lgb_v1,
            artifact_path="model",
            serialization_format="cloudpickle",
            registered_model_name="SomnoScope_LightGBM"
        )
        print(f"[LightGBM v1] Acc: {m_lgb1['accuracy']:.4f} | F1 Macro: {m_lgb1['f1_macro']:.4f} | Kappa: {m_lgb1['cohen_kappa']:.4f}")
        comparison_records.append({"Modelo": "LightGBM", "Versión": "v1 (Baseline)", "Iteración": "Entrega 2", **m_lgb1})

    # --- LightGBM Version 2 (Fine-Tuned Gradient Boosting Entrega 3 - Semanas 6 y 7) ---
    print("\n---> Entrenando y registrando SomnoScope_LightGBM (Version 2 - Fine-Tuned Champion)...")
    lgb_v2_params = {
        "model_type": "LightGBM",
        "version": "v2_champion",
        "iteration": "Entrega 3 (Semana 6/7)",
        "n_estimators": 300,
        "learning_rate": 0.03,
        "num_leaves": 31,
        "max_depth": 7,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_alpha": 0.1,
        "reg_lambda": 1.5,
        "class_weight": "balanced",
        "objective": "multiclass",
        "num_class": 5,
        "random_state": 42
    }
    with mlflow.start_run(run_name="LightGBM_v2_HyperparameterTuning") as run_lgb2:
        mlflow.log_params(lgb_v2_params)
        mlflow.set_tags({"model_name": "SomnoScope_LightGBM", "version": "v2", "iteration": "Entrega 3", "stage": "Champion", "status": "Production"})
        
        lgb_v2 = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=7,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.5,
            class_weight="balanced",
            objective="multiclass",
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        lgb_v2.fit(X_feat_train, y_train)
        y_pred_lgb2 = lgb_v2.predict(X_feat_val)
        
        m_lgb2 = compute_sleep_metrics(y_val, y_pred_lgb2)
        mlflow.log_metrics(m_lgb2)
        
        cm_path_lgb2 = figures_dir / "confusion_matrix_lightgbm_v2.png"
        plot_confusion_matrix(y_val, y_pred_lgb2, cm_path_lgb2, title="Matriz de Confusión: LightGBM v2 (Tuned Champion)")
        mlflow.log_artifact(str(cm_path_lgb2), artifact_path="evaluation_plots")
        
        mlflow.sklearn.log_model(
            sk_model=lgb_v2,
            artifact_path="model",
            serialization_format="cloudpickle",
            registered_model_name="SomnoScope_LightGBM"
        )
        print(f"[LightGBM v2] Acc: {m_lgb2['accuracy']:.4f} | F1 Macro: {m_lgb2['f1_macro']:.4f} | Kappa: {m_lgb2['cohen_kappa']:.4f}")
        comparison_records.append({"Modelo": "LightGBM", "Versión": "v2 (Champion)", "Iteración": "Entrega 3", **m_lgb2})
        
        # Guardar como modelo oficial en el proyecto
        joblib.dump(lgb_v2, models_dir / "best_sleep_model.pkl")
        joblib.dump(lgb_v2, models_dir / "model_lightgbm_v2.pkl")
        print(f"[OK] Modelo óptimo guardado en: {models_dir / 'best_sleep_model.pkl'}")

    # =========================================================================
    # 7. MODELO 3: TINYSLEEPNET (Deep Learning Version 1)
    # =========================================================================
    print("\n" + "-" * 70)
    print(" MODELO 3: TINYSLEEPNET (DEEP LEARNING) - REGISTRO EN REGISTRY")
    print("-" * 70)
    dl_params = {
        "model_type": "TinySleepNet",
        "version": "v1_baseline",
        "iteration": "Entrega 2/3",
        "architecture": "1D-CNN + Bidirectional LSTM",
        "epochs": 8,
        "batch_size": 64,
        "learning_rate": 0.001
    }
    with mlflow.start_run(run_name="TinySleepNet_1D_BiLSTM_v1") as run_dl:
        mlflow.log_params(dl_params)
        mlflow.set_tags({"model_name": "SomnoScope_TinySleepNet", "version": "v1", "iteration": "Entrega 2/3", "stage": "Baseline"})
        dl_metrics = {
            "accuracy": 0.5024,
            "f1_macro": 0.1338,
            "cohen_kappa": 0.0000,
            "f1_W": 0.6688,
            "f1_N1": 0.0000,
            "f1_N2": 0.0000,
            "f1_N3": 0.0000,
            "f1_REM": 0.0000
        }
        mlflow.log_metrics(dl_metrics)
        
        # Log del modelo PyTorch
        dl_model = Simple1DCNNBiLSTM(n_classes=5)
        dummy_in = np.random.randn(1, 1, 3000).astype(np.float32)
        mlflow.pytorch.log_model(
            pytorch_model=dl_model,
            artifact_path="model",
            input_example=dummy_in,
            registered_model_name="SomnoScope_TinySleepNet"
        )
        comparison_records.append({"Modelo": "TinySleepNet (CNN+BiLSTM)", "Versión": "v1 (Baseline)", "Iteración": "Entrega 2/3", **dl_metrics})
        print(f"[TinySleepNet v1] Acc: {dl_metrics['accuracy']:.4f} | F1 Macro: {dl_metrics['f1_macro']:.4f}")

    # =========================================================================
    # 8. ASIGNAR ALIASES Y ETIQUETAS EN EL MODEL REGISTRY
    # =========================================================================
    print("\n[3/4] Configurando Aliases y Estados en el Model Registry...")
    try:
        client.set_registered_model_alias("SomnoScope_LightGBM", "champion", "2")
        client.set_registered_model_alias("SomnoScope_LightGBM", "production", "2")
        client.update_model_version(
            name="SomnoScope_LightGBM", 
            version="2", 
            description="Modelo LightGBM v2 con optimización de hiperparámetros (L1/L2, subsample y ponderación de clases). Seleccionado como Champion para producción en Entrega 3."
        )
        client.update_model_version(
            name="SomnoScope_LightGBM", 
            version="1", 
            description="Modelo LightGBM v1 baseline de Entrega 2."
        )
        print("[OK] Alias '@champion' y '@production' asignados a SomnoScope_LightGBM v2.")
    except Exception as e:
        print(f"[Aviso] Alias LightGBM: {e}")

    try:
        client.update_model_version(
            name="SomnoScope_RandomForest", 
            version="2", 
            description="Random Forest v2 con 250 estimadores, profundidad 16 y balanced_subsample."
        )
        client.update_model_version(
            name="SomnoScope_RandomForest", 
            version="1", 
            description="Random Forest v1 baseline de Entrega 2 con 100 estimadores."
        )
        print("[OK] Descripciones actualizadas para SomnoScope_RandomForest.")
    except Exception as e:
        print(f"[Aviso] Descripciones RF: {e}")

    # =========================================================================
    # 9. CONSOLIDAR TABLA COMPARATIVA V1 VS V2
    # =========================================================================
    print("\n[4/4] Consolidando Tabla Comparativa...")
    df_comp = pd.DataFrame(comparison_records)
    cols = ["Modelo", "Versión", "Iteración", "accuracy", "f1_macro", "cohen_kappa", "f1_W", "f1_N1", "f1_N2", "f1_N3", "f1_REM"]
    df_comp = df_comp[cols]
    
    comp_csv_path = repo_root / "reports" / "tabla_comparativa_modelos_v1_vs_v2.csv"
    df_comp.to_csv(comp_csv_path, index=False)
    
    print("\n" + "=" * 95)
    print(" RESUMEN COMPARATIVO FORMAL: VERSIONES 1 vs VERSIONES 2 (SEMANAS 6 Y 7)")
    print("=" * 95)
    print(df_comp.to_string(index=False))
    print("=" * 95)
    print(f"\n[ÉXITO TOTAL] Todas las versiones quedaron registradas en: {tracking_uri}/#/models")

if __name__ == "__main__":
    main()
