"""
Entrenamiento y evaluación de Modelos Clásicos de Machine Learning (Random Forest y LightGBM)
basados en características espectrales de EEG y seguimiento en MLflow.
"""

from pathlib import Path
from typing import Dict, Tuple, Any
import joblib
import numpy as np
import mlflow
import mlflow.sklearn
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

from src.models.evaluate import compute_sleep_metrics, plot_confusion_matrix
from src.utils.mlflow_client import configure_mlflow

def train_random_forest(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_val: np.ndarray, 
    y_val: np.ndarray,
    n_estimators: int = 150,
    max_depth: int = 15,
    figures_dir: Path = Path("reports/figures")
) -> Tuple[RandomForestClassifier, Dict[str, float]]:
    """Entrena un modelo Random Forest con pesos balanceados y registra en MLflow."""
    configure_mlflow()
    
    params = {
        "model_type": "RandomForest",
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "class_weight": "balanced",
        "random_state": 42
    }
    
    with mlflow.start_run(run_name="RandomForest_Spectral_Baseline"):
        mlflow.log_params(params)
        
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_val)
        metrics = compute_sleep_metrics(y_val, y_pred)
        mlflow.log_metrics(metrics)
        
        # Matriz de confusión
        cm_path = figures_dir / "confusion_matrix_random_forest.png"
        plot_confusion_matrix(y_val, y_pred, cm_path, title="Matriz de Confusión: Random Forest")
        mlflow.log_artifact(str(cm_path), artifact_path="evaluation_plots")
        
        # Guardar modelo en MLflow
        mlflow.sklearn.log_model(clf, artifact_path="model_rf", serialization_format="cloudpickle")
        
        print(f"[Random Forest] Acc: {metrics['accuracy']:.4f} | F1 Macro: {metrics['f1_macro']:.4f} | Kappa: {metrics['cohen_kappa']:.4f}")
        return clf, metrics

def train_lightgbm(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    X_val: np.ndarray, 
    y_val: np.ndarray,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    figures_dir: Path = Path("reports/figures")
) -> Tuple[lgb.LGBMClassifier, Dict[str, float]]:
    """Entrena un modelo LightGBM con pesos balanceados y registra en MLflow."""
    configure_mlflow()
    
    params = {
        "model_type": "LightGBM",
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "class_weight": "balanced",
        "objective": "multiclass",
        "num_class": 5,
        "random_state": 42
    }
    
    with mlflow.start_run(run_name="LightGBM_Spectral_Optimized"):
        mlflow.log_params(params)
        
        clf = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            class_weight="balanced",
            objective="multiclass",
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_val)
        metrics = compute_sleep_metrics(y_val, y_pred)
        mlflow.log_metrics(metrics)
        
        # Matriz de confusión
        cm_path = figures_dir / "confusion_matrix_lightgbm.png"
        plot_confusion_matrix(y_val, y_pred, cm_path, title="Matriz de Confusión: LightGBM")
        mlflow.log_artifact(str(cm_path), artifact_path="evaluation_plots")
        
        # Guardar modelo en MLflow
        mlflow.sklearn.log_model(clf, artifact_path="model_lightgbm", serialization_format="cloudpickle")
        
        print(f"[LightGBM] Acc: {metrics['accuracy']:.4f} | F1 Macro: {metrics['f1_macro']:.4f} | Kappa: {metrics['cohen_kappa']:.4f}")
        return clf, metrics
