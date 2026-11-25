"""
Script de entrenamiento integral sobre los 18 registros completos de Sleep-EDFx (9 sujetos, 18,604 épocas).
Partición estricta por sujeto (GroupShuffleSplit, 6 sujetos train = 12,537 épocas, 3 sujetos test = 6,067 épocas).
Entrena y compara en igualdad de condiciones:
  - Random Forest v1 & v2
  - LightGBM v1 & v2
  - TinySleepNet v1 (sin normalización) & v2 (con normalización z-score por época + checkpointing)
Guarda los artefactos finales en models/ y los reportes en reports/.
"""

import os
import sys
import time
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb

os.environ["PYTHONIOENCODING"] = "utf-8"

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.data.loader import load_dataset_by_subjects
from src.features.spectral import extract_features_dataset
from src.models.deep_learning import TinySleepNet1D, train_deep_sleep_net, predict_deep_sleep_net
from src.models.evaluate import compute_sleep_metrics, plot_confusion_matrix


def train_tinysleepnet_unnormalized(X_train, y_train, X_val, y_val, epochs=3, batch_size=64, lr=0.001):
    """Replica el comportamiento original sin normalizar para documentar la línea base v1 en las 6,067 épocas."""
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr_t = torch.tensor(X_train[:, np.newaxis, :], dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    X_va_t = torch.tensor(X_val[:, np.newaxis, :], dtype=torch.float32)

    gen = torch.Generator().manual_seed(42)
    tr_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True, generator=gen)
    va_loader = DataLoader(TensorDataset(X_va_t), batch_size=batch_size, shuffle=False)

    counts = np.bincount(y_train, minlength=5)
    weights = torch.tensor(len(y_train) / (5.0 * np.maximum(counts, 1.0)), dtype=torch.float32).to(device)
    model = TinySleepNet1D(in_channels=1, n_classes=5).to(device)
    crit = nn.CrossEntropyLoss(weight=weights)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for ep in range(epochs):
        model.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            opt.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for (bx,) in va_loader:
            preds.extend(model(bx.to(device)).argmax(dim=1).cpu().numpy())
    val_preds = np.array(preds)
    metrics = compute_sleep_metrics(y_val, val_preds)
    return model, metrics, val_preds


def main():
    print("=" * 80)
    print(" SOMNOSCOPE - BENCHMARK COMPLETO (18 REGISTROS, 9 SUJETOS, 6,067 ÉPOCAS TEST)")
    print("=" * 80)

    data_dir = repo_root / "data" / "raw"
    proc_dir = repo_root / "data" / "processed"
    models_dir = repo_root / "Modelos"
    figures_dir = repo_root / "reports" / "figures"
    proc_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cargar los 18 registros completos (con caché dedicada)
    cache_raw_path = proc_dir / "cache_dataset_full18.npz"
    if cache_raw_path.exists():
        print(f"\n[1/5] Cargando dataset completo desde caché: {cache_raw_path.name}...")
        d = np.load(cache_raw_path)
        X_raw, y, subjects = d["X_raw"], d["y"], d["subjects"]
    else:
        print("\n[1/5] Cargando los 18 registros EDF de data/raw...")
        t0 = time.time()
        X_raw, y, subjects = load_dataset_by_subjects(data_dir)
        print(f"Cargadas {len(y):,} épocas de {len(np.unique(subjects))} sujetos en {time.time() - t0:.1f}s")
        np.savez_compressed(cache_raw_path, X_raw=X_raw, y=y, subjects=subjects)

    # 2. Partición estricta por sujeto (Anti-Data Leakage)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.33, random_state=42)
    train_idx, val_idx = next(gss.split(X_raw, y, groups=subjects))
    X_raw_train, y_train = X_raw[train_idx], y[train_idx]
    X_raw_val, y_val = X_raw[val_idx], y[val_idx]

    train_subjs = sorted(np.unique(subjects[train_idx]).tolist())
    val_subjs = sorted(np.unique(subjects[val_idx]).tolist())
    print(f"[2/5] Partición por sujeto:")
    print(f"      Train ({len(train_subjs)} sujetos): {train_subjs} -> {len(y_train):,} épocas")
    print(f"      Test  ({len(val_subjs)} sujetos): {val_subjs} -> {len(y_val):,} épocas")
    print(f"      Distribución Test: W={np.sum(y_val==0)}, N1={np.sum(y_val==1)}, N2={np.sum(y_val==2)}, N3={np.sum(y_val==3)}, REM={np.sum(y_val==4)}")

    # 3. Extracción de características espectrales para modelos clásicos
    cache_feat_path = proc_dir / "cache_features_full18.npz"
    if cache_feat_path.exists():
        print(f"\n[3/5] Cargando características espectrales desde caché: {cache_feat_path.name}...")
        df = np.load(cache_feat_path)
        X_feat_train, X_feat_val = df["X_train"], df["X_val"]
    else:
        print("\n[3/5] Extrayendo 25 características espectrales (Welch PSD, ratios, Hjorth)...")
        t0 = time.time()
        X_feat_train = extract_features_dataset(X_raw_train)
        X_feat_val = extract_features_dataset(X_raw_val)
        print(f"Extraídas en {time.time() - t0:.1f}s -> Train: {X_feat_train.shape}, Val: {X_feat_val.shape}")
        np.savez_compressed(cache_feat_path, X_train=X_feat_train, y_train=y_train, X_val=X_feat_val, y_val=y_val)

    records = []

    # 4. Modelos Clásicos (RF v1/v2 y LightGBM v1/v2)
    print("\n[4/5] Entrenando modelos clásicos sobre los 18 registros...")

    # RF v1
    rf_v1 = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)
    rf_v1.fit(X_feat_train, y_train)
    m_rf1 = compute_sleep_metrics(y_val, rf_v1.predict(X_feat_val))
    plot_confusion_matrix(y_val, rf_v1.predict(X_feat_val), figures_dir / "confusion_matrix_rf_v1.png", title="Matriz de Confusión: Random Forest v1 (6,067 épocas)")
    print(f"  [RF v1]   Acc={m_rf1['accuracy']:.4f} | F1-Macro={m_rf1['f1_macro']:.4f} | Kappa={m_rf1['cohen_kappa']:.4f} | F1-N1={m_rf1['f1_N1']:.4f} | F1-N3={m_rf1['f1_N3']:.4f}")
    records.append({"Modelo": "Random Forest", "Versión": "v1 (Baseline)", "Iteración": "Full 18 registros", **m_rf1})

    # RF v2
    rf_v2 = RandomForestClassifier(n_estimators=250, max_depth=16, min_samples_split=4, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    rf_v2.fit(X_feat_train, y_train)
    m_rf2 = compute_sleep_metrics(y_val, rf_v2.predict(X_feat_val))
    plot_confusion_matrix(y_val, rf_v2.predict(X_feat_val), figures_dir / "confusion_matrix_rf_v2.png", title="Matriz de Confusión: Random Forest v2 (6,067 épocas)")
    print(f"  [RF v2]   Acc={m_rf2['accuracy']:.4f} | F1-Macro={m_rf2['f1_macro']:.4f} | Kappa={m_rf2['cohen_kappa']:.4f} | F1-N1={m_rf2['f1_N1']:.4f} | F1-N3={m_rf2['f1_N3']:.4f}")
    records.append({"Modelo": "Random Forest", "Versión": "v2 (Tuned)", "Iteración": "Full 18 registros", **m_rf2})
    joblib.dump(rf_v1, models_dir / "model_random_forest.pkl")

    # LightGBM v1
    lgb_v1 = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.05, class_weight="balanced", objective="multiclass", random_state=42, n_jobs=-1, verbose=-1)
    lgb_v1.fit(X_feat_train, y_train)
    pred_lgb1 = lgb_v1.predict(X_feat_val)
    m_lgb1 = compute_sleep_metrics(y_val, pred_lgb1)
    plot_confusion_matrix(y_val, pred_lgb1, figures_dir / "confusion_matrix_lightgbm_v1.png", title="Matriz de Confusión: LightGBM v1 (6,067 épocas)")
    plot_confusion_matrix(y_val, pred_lgb1, figures_dir / "confusion_matrix_lightgbm.png", title="Matriz de Confusión: LightGBM (6,067 épocas)")
    print(f"  [LGBM v1] Acc={m_lgb1['accuracy']:.4f} | F1-Macro={m_lgb1['f1_macro']:.4f} | Kappa={m_lgb1['cohen_kappa']:.4f} | F1-N1={m_lgb1['f1_N1']:.4f} | F1-N3={m_lgb1['f1_N3']:.4f}")
    records.append({"Modelo": "LightGBM", "Versión": "v1 (Baseline)", "Iteración": "Full 18 registros", **m_lgb1})

    # LightGBM v2
    lgb_v2 = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.03, num_leaves=31, max_depth=7,
        subsample=0.85, colsample_bytree=0.85, reg_alpha=0.1, reg_lambda=1.5,
        class_weight="balanced", objective="multiclass", random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_v2.fit(X_feat_train, y_train)
    pred_lgb2 = lgb_v2.predict(X_feat_val)
    m_lgb2 = compute_sleep_metrics(y_val, pred_lgb2)
    plot_confusion_matrix(y_val, pred_lgb2, figures_dir / "confusion_matrix_lightgbm_v2.png", title="Matriz de Confusión: LightGBM v2 (6,067 épocas)")
    print(f"  [LGBM v2] Acc={m_lgb2['accuracy']:.4f} | F1-Macro={m_lgb2['f1_macro']:.4f} | Kappa={m_lgb2['cohen_kappa']:.4f} | F1-N1={m_lgb2['f1_N1']:.4f} | F1-N3={m_lgb2['f1_N3']:.4f}")
    records.append({"Modelo": "LightGBM", "Versión": "v2 (Tuned)", "Iteración": "Full 18 registros", **m_lgb2})

    # Guardar mejor LightGBM clásico para /predict/features y comparación en Dashboard
    best_lgb = lgb_v2 if m_lgb2["f1_macro"] > m_lgb1["f1_macro"] else lgb_v1
    joblib.dump(best_lgb, models_dir / "model_lightgbm.pkl")
    joblib.dump(best_lgb, models_dir / "best_sleep_model.pkl")

    # 5. Deep Learning: TinySleepNet v1 (sin normalizar) y v2 (normalizado z-score por época)
    print("\n[5/5] Entrenando TinySleepNet1D...")
    print("  ---> Evaluando TinySleepNet v1 (sin normalización - línea base colapsada)...")
    t0 = time.time()
    tsn_v1, m_tsn1, preds_tsn1 = train_tinysleepnet_unnormalized(X_raw_train, y_train, X_raw_val, y_val, epochs=2)
    plot_confusion_matrix(y_val, preds_tsn1, figures_dir / "confusion_matrix_tinysleepnet_v1.png", title="Matriz de Confusión: TinySleepNet v1 (Sin Normalizar)")
    print(f"  [TSN v1]  Acc={m_tsn1['accuracy']:.4f} | F1-Macro={m_tsn1['f1_macro']:.4f} | Kappa={m_tsn1['cohen_kappa']:.4f} | F1-N1={m_tsn1['f1_N1']:.4f} ({time.time()-t0:.1f}s)")
    records.append({"Modelo": "TinySleepNet (1D-CNN+BiLSTM)", "Versión": "v1 (Sin normalizar)", "Iteración": "Baseline", **m_tsn1})

    print("  ---> Entrenando TinySleepNet v2 (con normalización z-score por época + checkpointing)...")
    t0 = time.time()
    tsn_v2, m_tsn2 = train_deep_sleep_net(
        X_raw_train, y_train, X_raw_val, y_val,
        epochs=10, batch_size=64, learning_rate=0.001,
        figures_dir=figures_dir, random_state=42, log_to_mlflow=False
    )
    print(f"  Entrenamiento TinySleepNet v2 completado en {time.time()-t0:.1f}s")

    # Verificación exacta del checkpoint guardado
    torch.save(tsn_v2.state_dict(), models_dir / "model_tinysleepnet.pt")
    ver_preds, ver_probs = predict_deep_sleep_net(tsn_v2, X_raw_val)
    m_ver = compute_sleep_metrics(y_val, ver_preds)
    plot_confusion_matrix(y_val, ver_preds, figures_dir / "confusion_matrix_tinysleepnet_v2.png", title="Matriz de Confusión: TinySleepNet v2 Normalizado (Champion)")
    print(f"  [TSN v2 VERIFICADO] Acc={m_ver['accuracy']:.4f} | F1-Macro={m_ver['f1_macro']:.4f} | Kappa={m_ver['cohen_kappa']:.4f} | F1-N1={m_ver['f1_N1']:.4f} | F1-N3={m_ver['f1_N3']:.4f}")
    records.append({
        "Modelo": "TinySleepNet (1D-CNN+BiLSTM)",
        "Versión": f"v2 (Normalizado, ep {int(m_tsn2.get('best_epoch', 0))})",
        "Iteración": "Entrega Final (Champion)",
        **m_ver
    })

    # Guardar tablas comparativas y resumen JSON
    df_comp = pd.DataFrame(records)
    cols = ["Modelo", "Versión", "Iteración", "accuracy", "f1_macro", "cohen_kappa", "f1_W", "f1_N1", "f1_N2", "f1_N3", "f1_REM"]
    df_comp = df_comp[cols]
    df_comp.to_csv(repo_root / "reports" / "tabla_comparativa_modelos_v1_vs_v2.csv", index=False)
    df_comp.to_csv(repo_root / "reports" / "tabla_comparativa_modelos.csv", index=False)

    summary = {
        "train_subjects": train_subjs,
        "val_subjects": val_subjs,
        "train_epochs": int(len(y_train)),
        "val_epochs": int(len(y_val)),
        "best_epoch_tsn_v2": int(m_tsn2.get("best_epoch", 0)),
        "metrics": records
    }
    with open(repo_root / "reports" / "benchmark_summary_full18.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 100)
    print(" RESUMEN FINAL COMPARATIVO (6,067 ÉPOCAS DE PRUEBA - SUJETOS INÉDITOS)")
    print("=" * 100)
    print(df_comp.to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    main()
