"""
Validación Cruzada Estricta por Sujeto (3-Fold GroupKFold) sobre la cohorte ampliada
Sleep-EDF-20 + Telemetry (23 sujetos únicos, 45 noches, 48,378 épocas de 30s).

Evalúa en igualdad de condiciones sobre exactamente los mismos 3 folds (0% fuga de datos):
  - Random Forest v1 (Baseline)
  - LightGBM v1 (Baseline)
  - LightGBM v2 (Tuned)
  - TinySleepNet1D v2 (Normalizado z-score por época + checkpointing)

Calcula Media ± Desviación Estándar por métrica, matriz de confusión agregada Out-Of-Fold (OOF)
sobre las 48,378 épocas inéditas y guarda todos los resultados en reports/.
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

os.environ["PYTHONIOENCODING"] = "utf-8"

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.features.spectral import extract_features_dataset
from src.models.deep_learning import train_deep_sleep_net, predict_deep_sleep_net
from src.models.evaluate import compute_sleep_metrics, plot_confusion_matrix


def summarize_folds(fold_metrics_list):
    keys = ["accuracy", "f1_macro", "cohen_kappa", "f1_W", "f1_N1", "f1_N2", "f1_N3", "f1_REM"]
    summary = {}
    for k in keys:
        vals = [float(m[k]) for m in fold_metrics_list]
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return summary


def main():
    t0_all = time.time()
    print("=" * 90)
    print(" VALIDACIÓN CRUZADA 3-FOLD GROUPKFOLD — 23 SUJETOS (48,378 ÉPOCAS, ANTI-DATA LEAKAGE)")
    print("=" * 90)

    local_ssd_dir = Path(os.path.expanduser("~")) / "somnoscope_sleepedf20_cache"
    cache_raw_path = local_ssd_dir / "cache_sleepedf20_plus_telemetry.npz"
    cache_feat_split_path = local_ssd_dir / "cache_features_sleepedf20.npz"
    cache_feat_full_path = local_ssd_dir / "cache_features_full_23subj.npz"

    if not cache_raw_path.exists():
        raise FileNotFoundError(f"No se encontró el caché preprocesado en {cache_raw_path}")

    d = np.load(cache_raw_path)
    X_raw, y, subjects = d["X_raw"], d["y"], d["subjects"]
    n_total = len(y)
    unique_subjs = sorted(np.unique(subjects).tolist())
    print(f"[1/4] Dataset cargado desde SSD: {n_total:,} épocas | {len(unique_subjs)} sujetos únicos.")

    # Reconstruir matriz completa de 25 features en <1s reutilizando el split cacheado
    if cache_feat_full_path.exists():
        X_feat = np.load(cache_feat_full_path)["X_feat"]
        print(f"[2/4] Características espectrales completas cargadas: {X_feat.shape}")
    elif cache_feat_split_path.exists():
        print("[2/4] Reconstruyendo matriz completa de 25 features desde caché SSD (0s extracción)...")
        gss_prev = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=42)
        tr_prev, va_prev = next(gss_prev.split(X_raw, y, groups=subjects))
        df_s = np.load(cache_feat_split_path)
        X_feat = np.zeros((n_total, df_s["X_train"].shape[1]), dtype=np.float32)
        X_feat[tr_prev] = df_s["X_train"]
        X_feat[va_prev] = df_s["X_val"]
        np.savez_compressed(cache_feat_full_path, X_feat=X_feat)
        print(f"      Reconstrucción verificada: {X_feat.shape}")
    else:
        print("[2/4] Extrayendo 25 características espectrales...")
        X_feat = extract_features_dataset(X_raw)
        np.savez_compressed(cache_feat_full_path, X_feat=X_feat)

    gkf = GroupKFold(n_splits=3)
    folds = list(gkf.split(X_raw, y, groups=subjects))

    reports_dir = repo_root / "reports"
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    model_folds = {
        "Random Forest v1": [],
        "LightGBM v1": [],
        "LightGBM v2": [],
        "TinySleepNet v2 (Normalizado)": [],
    }
    oof_preds_tsn = np.zeros(n_total, dtype=int)
    oof_preds_lgb2 = np.zeros(n_total, dtype=int)
    fold_metadata = []

    print("\n[3/4] Ejecutando los 3 Folds de GroupKFold (evaluando 23/23 sujetos inéditos)...")
    for f_idx, (tr_idx, va_idx) in enumerate(folds, 1):
        tr_subjs = sorted(np.unique(subjects[tr_idx]).tolist())
        va_subjs = sorted(np.unique(subjects[va_idx]).tolist())
        print("\n" + "-" * 85)
        print(f" FOLD {f_idx}/3 | Train: {len(tr_subjs)} sujetos ({len(tr_idx):,} épocas) | Test: {len(va_subjs)} sujetos ({len(va_idx):,} épocas)")
        print(f" Test sujetos Fold {f_idx}: {va_subjs}")
        print("-" * 85, flush=True)

        X_f_tr, y_tr = X_feat[tr_idx], y[tr_idx]
        X_f_va, y_va = X_feat[va_idx], y[va_idx]
        X_r_tr, X_r_va = X_raw[tr_idx], X_raw[va_idx]

        # 1. Random Forest v1
        rf = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)
        rf.fit(X_f_tr, y_tr)
        m_rf = compute_sleep_metrics(y_va, rf.predict(X_f_va))
        model_folds["Random Forest v1"].append(m_rf)
        print(f"  [RF v1]   Acc={m_rf['accuracy']:.4f} | F1-Mac={m_rf['f1_macro']:.4f} | Kappa={m_rf['cohen_kappa']:.4f} | F1-N1={m_rf['f1_N1']:.4f} | F1-N3={m_rf['f1_N3']:.4f}", flush=True)

        # 2. LightGBM v1
        lgb1 = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, class_weight="balanced", objective="multiclass", random_state=42, n_jobs=-1, verbose=-1)
        lgb1.fit(X_f_tr, y_tr)
        m_lgb1 = compute_sleep_metrics(y_va, lgb1.predict(X_f_va))
        model_folds["LightGBM v1"].append(m_lgb1)
        print(f"  [LGBM v1] Acc={m_lgb1['accuracy']:.4f} | F1-Mac={m_lgb1['f1_macro']:.4f} | Kappa={m_lgb1['cohen_kappa']:.4f} | F1-N1={m_lgb1['f1_N1']:.4f} | F1-N3={m_lgb1['f1_N3']:.4f}", flush=True)

        # 3. LightGBM v2
        lgb2 = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.03, num_leaves=31, max_depth=7,
            subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.5,
            class_weight="balanced", objective="multiclass", random_state=42, n_jobs=-1, verbose=-1
        )
        lgb2.fit(X_f_tr, y_tr)
        p_lgb2 = lgb2.predict(X_f_va)
        oof_preds_lgb2[va_idx] = p_lgb2
        m_lgb2 = compute_sleep_metrics(y_va, p_lgb2)
        model_folds["LightGBM v2"].append(m_lgb2)
        print(f"  [LGBM v2] Acc={m_lgb2['accuracy']:.4f} | F1-Mac={m_lgb2['f1_macro']:.4f} | Kappa={m_lgb2['cohen_kappa']:.4f} | F1-N1={m_lgb2['f1_N1']:.4f} | F1-N3={m_lgb2['f1_N3']:.4f}", flush=True)

        # 4. TinySleepNet v2 Normalizado (8 épocas + checkpointing)
        t_dl = time.time()
        tsn_model, m_tsn_best = train_deep_sleep_net(
            X_r_tr, y_tr, X_r_va, y_va,
            epochs=8, batch_size=64, learning_rate=0.001,
            figures_dir=figures_dir, random_state=42 + f_idx, log_to_mlflow=False
        )
        p_tsn, _ = predict_deep_sleep_net(tsn_model, X_r_va)
        oof_preds_tsn[va_idx] = p_tsn
        m_tsn = compute_sleep_metrics(y_va, p_tsn)
        m_tsn["best_epoch"] = float(m_tsn_best.get("best_epoch", 0))
        model_folds["TinySleepNet v2 (Normalizado)"].append(m_tsn)
        print(
            f"  [TSN v2]  Acc={m_tsn['accuracy']:.4f} | F1-Mac={m_tsn['f1_macro']:.4f} | "
            f"Kappa={m_tsn['cohen_kappa']:.4f} | F1-N1={m_tsn['f1_N1']:.4f} | F1-N3={m_tsn['f1_N3']:.4f} "
            f"(ep {int(m_tsn['best_epoch'])}, {time.time() - t_dl:.1f}s)",
            flush=True
        )

        fold_metadata.append({
            "fold": f_idx,
            "train_subjects": tr_subjs,
            "val_subjects": va_subjs,
            "train_epochs": int(len(tr_idx)),
            "val_epochs": int(len(va_idx)),
            "metrics_by_model": {
                "Random Forest v1": m_rf,
                "LightGBM v1": m_lgb1,
                "LightGBM v2": m_lgb2,
                "TinySleepNet v2 (Normalizado)": m_tsn,
            }
        })

    # 4. Consolidar Media ± DE y Matrices OOF Agregadas (48,378 épocas)
    print("\n[4/4] Consolidando resultados 3-Fold GroupKFold y matrices OOF...")
    oof_metrics_tsn = compute_sleep_metrics(y, oof_preds_tsn)
    oof_metrics_lgb2 = compute_sleep_metrics(y, oof_preds_lgb2)

    plot_confusion_matrix(
        y, oof_preds_tsn,
        figures_dir / "confusion_matrix_cv_tinysleepnet_23subj.png",
        title="Matriz de Confusión Agregada 3-Fold GroupKFold\nTinySleepNet v2 (48,378 épocas, 23 sujetos)"
    )
    plot_confusion_matrix(
        y, oof_preds_lgb2,
        figures_dir / "confusion_matrix_cv_lightgbm_23subj.png",
        title="Matriz de Confusión Agregada 3-Fold GroupKFold\nLightGBM v2 (48,378 épocas, 23 sujetos)"
    )

    summary_rows = []
    summaries_dict = {}
    for m_name, f_list in model_folds.items():
        s = summarize_folds(f_list)
        summaries_dict[m_name] = s
        summary_rows.append({"Modelo": m_name, **s})

    df_cv = pd.DataFrame(summary_rows)
    df_cv.to_csv(reports_dir / "tabla_cv_groupkfold_23subjects.csv", index=False)

    cv_payload = {
        "n_splits": 3,
        "split_strategy": "GroupKFold_by_subject",
        "n_subjects_total": len(unique_subjs),
        "n_epochs_total": int(n_total),
        "subjects": unique_subjs,
        "elapsed_seconds": round(time.time() - t0_all, 1),
        "summary_mean_std": summaries_dict,
        "oof_aggregate_tinysleepnet_v2": oof_metrics_tsn,
        "oof_aggregate_lightgbm_v2": oof_metrics_lgb2,
        "folds": fold_metadata,
    }
    with open(reports_dir / "cv_groupkfold_23subjects.json", "w", encoding="utf-8") as f:
        json.dump(cv_payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 98)
    print(" RESUMEN OFICIAL 3-FOLD GROUPKFOLD (23 SUJETOS / 48,378 ÉPOCAS — MEDIA ± DESV. ESTÁNDAR)")
    print("=" * 98)
    for m_name, s in summaries_dict.items():
        print(
            f"{m_name:31s} | "
            f"Acc: {s['accuracy_mean']*100:5.2f}±{s['accuracy_std']*100:4.2f}% | "
            f"F1-Mac: {s['f1_macro_mean']:.4f}±{s['f1_macro_std']:.4f} | "
            f"Kappa: {s['cohen_kappa_mean']:.4f}±{s['cohen_kappa_std']:.4f} | "
            f"F1-N1: {s['f1_N1_mean']:.4f}±{s['f1_N1_std']:.4f} | "
            f"F1-N3: {s['f1_N3_mean']:.4f}±{s['f1_N3_std']:.4f}"
        )
    print("=" * 98)
    print(f"OOF Global TinySleepNet v2 (48,378 épocas): Acc={oof_metrics_tsn['accuracy']*100:.2f}% | F1-Mac={oof_metrics_tsn['f1_macro']:.4f} | Kappa={oof_metrics_tsn['cohen_kappa']:.4f} | F1-N1={oof_metrics_tsn['f1_N1']:.4f}")
    print(f"Tiempo total CV: {time.time() - t0_all:.1f}s")


if __name__ == "__main__":
    main()
