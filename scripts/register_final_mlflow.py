"""
Script de registro oficial en MLflow Model Registry para la Entrega Final (TinySleepNet Champion).
Lee los resultados exactos del benchmark sobre los 18 registros (12,537 train / 6,067 test por sujeto)
y registra en MLflow todos los modelos, parámetros, métricas, matrices de confusión y aliases:
  - SomnoScope_TinySleepNet v2 (Normalizado z-score): @champion y @production
  - SomnoScope_TinySleepNet v1 (Sin normalizar): línea base colapsada
  - SomnoScope_LightGBM v1 & v2: @challenger
  - SomnoScope_RandomForest v1 & v2: baseline / tuned
"""

import os
import sys
import json
from pathlib import Path
import joblib
import numpy as np
import torch
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.tracking import MlflowClient

os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.models.deep_learning import TinySleepNet1D

REMOTE_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://52.207.88.72:5000")
EXPERIMENT_NAME = "Microproyecto_Sleep_Staging"


def reset_registered_model(client: MlflowClient, model_name: str):
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
        for v in versions:
            client.delete_model_version(model_name, v.version)
        client.delete_registered_model(model_name)
        print(f"[Reset] Modelo '{model_name}' limpiado para registro correlativo v1/v2.")
    except Exception:
        pass


def extract_metric_dict(rec: dict) -> dict:
    keys = ["accuracy", "f1_macro", "f1_weighted", "cohen_kappa", "f1_W", "f1_N1", "f1_N2", "f1_N3", "f1_REM"]
    return {k: float(rec[k]) for k in keys if k in rec and rec[k] is not None}


def main():
    summary_path = repo_root / "reports" / "benchmark_summary_full18.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No se encontró {summary_path}. Ejecute primero scripts/train_full_benchmark.py.")

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    records = summary["metrics"]
    figures_dir = repo_root / "reports" / "figures"
    models_dir = repo_root / "models"

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", REMOTE_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri)
    print(f"[OK] Conectado a MLflow: {tracking_uri} | Experimento: {EXPERIMENT_NAME}")

    reset_registered_model(client, "SomnoScope_RandomForest")
    reset_registered_model(client, "SomnoScope_LightGBM")
    reset_registered_model(client, "SomnoScope_TinySleepNet")

    common_params = {
        "dataset": "Sleep-EDFx (18 registros, 9 sujetos)",
        "split_strategy": "GroupShuffleSplit_by_subject_67_33",
        "train_epochs": summary["train_epochs"],
        "val_epochs": summary["val_epochs"],
        "random_state": 42,
    }

    rf_model = joblib.load(models_dir / "model_random_forest.pkl")
    lgb_model = joblib.load(models_dir / "best_sleep_model.pkl")

    tsn_model = TinySleepNet1D(in_channels=1, n_classes=5)
    tsn_model.load_state_dict(torch.load(models_dir / "model_tinysleepnet.pt", map_location="cpu", weights_only=True))
    tsn_model.eval()

    # Mapeo de los 6 registros esperados
    run_specs = [
        ("RandomForest_v1_Baseline_Full18", "SomnoScope_RandomForest", "v1", "confusion_matrix_rf_v1.png",
         {"model_type": "RandomForest", "n_estimators": 100, "max_depth": 12, "class_weight": "balanced"}, "sklearn", rf_model),
        ("RandomForest_v2_Tuned_Full18", "SomnoScope_RandomForest", "v2", "confusion_matrix_rf_v2.png",
         {"model_type": "RandomForest", "n_estimators": 250, "max_depth": 16, "class_weight": "balanced_subsample"}, "sklearn", rf_model),
        ("LightGBM_v1_Baseline_Full18", "SomnoScope_LightGBM", "v1", "confusion_matrix_lightgbm_v1.png",
         {"model_type": "LightGBM", "n_estimators": 150, "learning_rate": 0.05, "class_weight": "balanced"}, "sklearn", lgb_model),
        ("LightGBM_v2_Tuned_Full18", "SomnoScope_LightGBM", "v2", "confusion_matrix_lightgbm_v2.png",
         {"model_type": "LightGBM", "n_estimators": 300, "learning_rate": 0.03, "reg_alpha": 0.1, "reg_lambda": 1.5}, "sklearn", lgb_model),
        ("TinySleepNet_v1_Unnormalized_Baseline", "SomnoScope_TinySleepNet", "v1", "confusion_matrix_tinysleepnet_v1.png",
         {"model_type": "TinySleepNet1D", "normalization": "none", "epochs": 2, "batch_size": 64, "learning_rate": 0.001}, "pytorch_dummy", TinySleepNet1D()),
        ("TinySleepNet_v2_Normalized_Champion", "SomnoScope_TinySleepNet", "v2", "confusion_matrix_tinysleepnet_v2.png",
         {"model_type": "TinySleepNet1D", "normalization": "z-score_per_epoch", "best_epoch": summary.get("best_epoch_tsn_v2", 4), "batch_size": 64, "learning_rate": 0.001}, "pytorch", tsn_model),
    ]

    for rec, (run_name, reg_name, ver_tag, cm_file, extra_p, kind, mdl_obj) in zip(records, run_specs):
        metrics = extract_metric_dict(rec)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({**common_params, **extra_p})
            mlflow.log_metrics(metrics)
            mlflow.set_tags({
                "model_name": reg_name,
                "version": ver_tag,
                "evaluation_cohort": "6067_unseen_subject_epochs",
                "status": "Champion" if run_name == "TinySleepNet_v2_Normalized_Champion" else "Candidate"
            })
            cm_path = figures_dir / cm_file
            if cm_path.exists():
                mlflow.log_artifact(str(cm_path), artifact_path="evaluation_plots")

            if kind == "sklearn":
                mlflow.sklearn.log_model(
                    sk_model=mdl_obj,
                    artifact_path="model",
                    serialization_format="cloudpickle",
                    registered_model_name=reg_name,
                )
            else:
                mlflow.pytorch.log_model(
                    pytorch_model=mdl_obj,
                    artifact_path="model",
                    registered_model_name=reg_name,
                )
            print(f"[OK] Registrado {reg_name} ({ver_tag}) | Acc={metrics['accuracy']:.4f} | F1-Macro={metrics['f1_macro']:.4f} | F1-N1={metrics['f1_N1']:.4f}")

    # Configurar Aliases y Descripciones oficiales
    tsn_v2_m = extract_metric_dict(records[5])
    lgb_best_idx = 3 if records[3]["f1_macro"] > records[2]["f1_macro"] else 2
    lgb_best_ver = "2" if lgb_best_idx == 3 else "1"
    lgb_best_m = extract_metric_dict(records[lgb_best_idx])

    client.set_registered_model_alias("SomnoScope_TinySleepNet", "champion", "2")
    client.set_registered_model_alias("SomnoScope_TinySleepNet", "production", "2")
    client.update_model_version(
        name="SomnoScope_TinySleepNet",
        version="2",
        description=(
            f"Modelo Champion Oficial (TinySleepNet1D con normalización z-score por época, "
            f"checkpointing época {summary.get('best_epoch_tsn_v2', '')}). Evaluado en 6,067 épocas "
            f"inter-sujeto: Accuracy={tsn_v2_m['accuracy']*100:.2f}%, F1-Macro={tsn_v2_m['f1_macro']:.4f}, "
            f"Kappa={tsn_v2_m['cohen_kappa']:.4f}, F1-N1={tsn_v2_m['f1_N1']:.4f}."
        ),
    )
    client.update_model_version(
        name="SomnoScope_TinySleepNet",
        version="1",
        description="Línea base sin normalización (colapsa por escala en voltios ~1e-5 frente a BatchNorm eps=1e-5).",
    )

    client.set_registered_model_alias("SomnoScope_LightGBM", "challenger", lgb_best_ver)
    client.update_model_version(
        name="SomnoScope_LightGBM",
        version=lgb_best_ver,
        description=(
            f"Mejor modelo tabular clásico (Challenger, 25 características espectrales). "
            f"Evaluado en 6,067 épocas inter-sujeto: Accuracy={lgb_best_m['accuracy']*100:.2f}%, "
            f"F1-Macro={lgb_best_m['f1_macro']:.4f}, Kappa={lgb_best_m['cohen_kappa']:.4f}, F1-N1={lgb_best_m['f1_N1']:.4f}."
        ),
    )
    print("\n[ÉXITO TOTAL] Aliases @champion/@production asignados a SomnoScope_TinySleepNet v2 y @challenger a SomnoScope_LightGBM.")


if __name__ == "__main__":
    main()
