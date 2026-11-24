"""
Registro oficial de Validación Cruzada 3-Fold GroupKFold (23 sujetos, 48,378 épocas) en MLflow.
Crea corridas anidadas (parent run con Media ± DE + 3 child runs por Fold) dentro del experimento
'Microproyecto_Sleep_Staging' y promueve en el Model Registry:
  - SomnoScope_TinySleepNet (nueva versión Champion 23 sujetos) -> @champion y @production
  - SomnoScope_LightGBM (nueva versión Challenger 23 sujetos) -> @challenger
  - SomnoScope_RandomForest (nueva versión 23 sujetos)
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


def main():
    cv_path = repo_root / "reports" / "cv_groupkfold_23subjects.json"
    if not cv_path.exists():
        raise FileNotFoundError(f"No se encontró {cv_path}. Ejecute primero scripts/run_groupkfold_cv_23subjects.py.")

    with open(cv_path, "r", encoding="utf-8") as f:
        cv = json.load(f)

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", REMOTE_TRACKING_URI)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient(tracking_uri)
    print(f"[OK] Conectado a MLflow: {tracking_uri} | Experimento: {EXPERIMENT_NAME}")

    models_dir = repo_root / "models"
    figures_dir = repo_root / "reports" / "figures"

    rf_model = joblib.load(models_dir / "model_random_forest.pkl")
    lgb_model = joblib.load(models_dir / "best_sleep_model.pkl")
    tsn_model = TinySleepNet1D(in_channels=1, n_classes=5)
    tsn_model.load_state_dict(torch.load(models_dir / "model_tinysleepnet.pt", map_location="cpu", weights_only=True))
    tsn_model.eval()

    model_specs = [
        ("Random Forest v1", "CV3_GroupKFold_RandomForest_23Subj", "SomnoScope_RandomForest", "sklearn", rf_model, None),
        ("LightGBM v1", "CV3_GroupKFold_LightGBM_v1_23Subj", None, None, None, None),
        ("LightGBM v2", "CV3_GroupKFold_LightGBM_v2_23Subj", "SomnoScope_LightGBM", "sklearn", lgb_model, "confusion_matrix_cv_lightgbm_23subj.png"),
        ("TinySleepNet v2 (Normalizado)", "CV3_GroupKFold_TinySleepNet_Champion_23Subj", "SomnoScope_TinySleepNet", "pytorch", tsn_model, "confusion_matrix_cv_tinysleepnet_23subj.png"),
    ]

    registered_versions = {}

    for m_key, parent_run_name, reg_name, kind, mdl_obj, cm_file in model_specs:
        s = cv["summary_mean_std"][m_key]
        with mlflow.start_run(run_name=parent_run_name):
            mlflow.log_params({
                "model_family": m_key,
                "dataset": "Sleep-EDF-20 + Telemetry (23 sujetos, 45 noches)",
                "cv_strategy": "GroupKFold_by_subject",
                "n_splits": cv["n_splits"],
                "n_subjects_total": cv["n_subjects_total"],
                "n_epochs_total": cv["n_epochs_total"],
                "normalization": "z-score_per_epoch" if "TinySleepNet" in m_key else "spectral_25_features",
            })

            # Métricas promedio y desviación estándar en el Parent Run
            parent_metrics = {
                "accuracy": s["accuracy_mean"],
                "accuracy_std": s["accuracy_std"],
                "f1_macro": s["f1_macro_mean"],
                "f1_macro_std": s["f1_macro_std"],
                "cohen_kappa": s["cohen_kappa_mean"],
                "cohen_kappa_std": s["cohen_kappa_std"],
                "f1_W": s["f1_W_mean"],
                "f1_N1": s["f1_N1_mean"],
                "f1_N1_std": s["f1_N1_std"],
                "f1_N2": s["f1_N2_mean"],
                "f1_N3": s["f1_N3_mean"],
                "f1_N3_std": s["f1_N3_std"],
                "f1_REM": s["f1_REM_mean"],
            }
            mlflow.log_metrics(parent_metrics)
            mlflow.set_tags({
                "evaluation_protocol": "3-Fold_GroupKFold_23_Subjects",
                "cohort_size": "48378_epochs",
                "role": "Champion" if "TinySleepNet" in m_key else ("Challenger" if m_key == "LightGBM v2" else "Baseline"),
            })

            # Child Runs (1 por cada Fold)
            for fold_entry in cv["folds"]:
                f_num = fold_entry["fold"]
                f_m = fold_entry["metrics_by_model"][m_key]
                with mlflow.start_run(run_name=f"{parent_run_name}_Fold{f_num}", nested=True):
                    mlflow.log_params({
                        "fold": f_num,
                        "val_subjects": ",".join(fold_entry["val_subjects"]),
                        "train_epochs": fold_entry["train_epochs"],
                        "val_epochs": fold_entry["val_epochs"],
                    })
                    mlflow.log_metrics({
                        k: float(v) for k, v in f_m.items()
                        if k in ["accuracy", "f1_macro", "f1_weighted", "cohen_kappa", "f1_W", "f1_N1", "f1_N2", "f1_N3", "f1_REM", "best_epoch"]
                    })

            if cm_file and (figures_dir / cm_file).exists():
                mlflow.log_artifact(str(figures_dir / cm_file), artifact_path="evaluation_plots")
            if (repo_root / "reports" / "tabla_cv_groupkfold_23subjects.csv").exists():
                mlflow.log_artifact(str(repo_root / "reports" / "tabla_cv_groupkfold_23subjects.csv"), artifact_path="cv_reports")

            if reg_name and mdl_obj is not None:
                if kind == "sklearn":
                    mv = mlflow.sklearn.log_model(
                        sk_model=mdl_obj,
                        artifact_path="model",
                        serialization_format="cloudpickle",
                        registered_model_name=reg_name,
                    )
                else:
                    dummy_in = np.zeros((1, 1, 3000), dtype=np.float32)
                    mv = mlflow.pytorch.log_model(
                        pytorch_model=mdl_obj,
                        artifact_path="model",
                        input_example=dummy_in,
                        registered_model_name=reg_name,
                    )
                # Obtener último número de versión creada
                versions = client.search_model_versions(f"name='{reg_name}'")
                latest_v = str(max(int(v.version) for v in versions))
                registered_versions[reg_name] = latest_v
                print(
                    f"[OK] Registrado {reg_name} v{latest_v} | "
                    f"CV3 Acc={s['accuracy_mean']*100:.2f}±{s['accuracy_std']*100:.2f}% | "
                    f"F1-Mac={s['f1_macro_mean']:.4f}±{s['f1_macro_std']:.4f} | "
                    f"F1-N1={s['f1_N1_mean']:.4f}±{s['f1_N1_std']:.4f}"
                )

    # Actualizar aliases oficiales en Model Registry
    tsn_s = cv["summary_mean_std"]["TinySleepNet v2 (Normalizado)"]
    lgb_s = cv["summary_mean_std"]["LightGBM v2"]

    if "SomnoScope_TinySleepNet" in registered_versions:
        v_tsn = registered_versions["SomnoScope_TinySleepNet"]
        client.set_registered_model_alias("SomnoScope_TinySleepNet", "champion", v_tsn)
        client.set_registered_model_alias("SomnoScope_TinySleepNet", "production", v_tsn)
        client.update_model_version(
            name="SomnoScope_TinySleepNet",
            version=v_tsn,
            description=(
                f"Champion Oficial entrenado en 23 sujetos (Sleep-EDF-20 + Telemetry, 48,378 épocas) y validado "
                f"con 3-Fold GroupKFold: Acc={tsn_s['accuracy_mean']*100:.2f}±{tsn_s['accuracy_std']*100:.2f}%, "
                f"F1-Macro={tsn_s['f1_macro_mean']:.4f}±{tsn_s['f1_macro_std']:.4f}, "
                f"Kappa={tsn_s['cohen_kappa_mean']:.4f}±{tsn_s['cohen_kappa_std']:.4f}, "
                f"F1-N1={tsn_s['f1_N1_mean']:.4f}±{tsn_s['f1_N1_std']:.4f}."
            ),
        )

    if "SomnoScope_LightGBM" in registered_versions:
        v_lgb = registered_versions["SomnoScope_LightGBM"]
        client.set_registered_model_alias("SomnoScope_LightGBM", "challenger", v_lgb)
        client.update_model_version(
            name="SomnoScope_LightGBM",
            version=v_lgb,
            description=(
                f"Challenger Clásico (25 features espectrales) entrenado en 23 sujetos y validado con "
                f"3-Fold GroupKFold: Acc={lgb_s['accuracy_mean']*100:.2f}±{lgb_s['accuracy_std']*100:.2f}%, "
                f"F1-Macro={lgb_s['f1_macro_mean']:.4f}±{lgb_s['f1_macro_std']:.4f}, "
                f"F1-N1={lgb_s['f1_N1_mean']:.4f}±{lgb_s['f1_N1_std']:.4f}."
            ),
        )

    print("\n[ÉXITO TOTAL] Validación Cruzada 3-Fold GroupKFold y nuevas versiones @champion/@challenger registradas en MLflow.")


if __name__ == "__main__":
    main()
