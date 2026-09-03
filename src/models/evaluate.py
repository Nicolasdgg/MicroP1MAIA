"""
Módulo de evaluación y cálculo de métricas para clasificación de estadios de sueño AASM.
Genera métricas globales (F1 Macro, Kappa, Accuracy), métricas por estadio y matriz de confusión.
"""

from typing import Dict, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix
)

from src.data.loader import AASM_CLASSES

def compute_sleep_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula las métricas clínicas estándar de polisomnografía."""
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # F1 por clase
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=range(len(AASM_CLASSES)), zero_division=0)
    
    metrics = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "cohen_kappa": float(kappa),
    }
    
    for i, cls_name in enumerate(AASM_CLASSES):
        metrics[f"f1_{cls_name}"] = float(f1_per_class[i])
        
    return metrics

def plot_confusion_matrix(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    output_path: Path, 
    title: str = "Matriz de Confusión Normalizada"
) -> Path:
    """Genera y guarda el gráfico de la matriz de confusión normalizada."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(AASM_CLASSES)), normalize='true')
    
    plt.figure(figsize=(7, 6), dpi=300)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=".2f", 
        cmap="Blues", 
        xticklabels=AASM_CLASSES, 
        yticklabels=AASM_CLASSES,
        cbar=True
    )
    plt.title(title, fontsize=12, fontweight='bold', pad=12)
    plt.xlabel("Estadio Predicho", fontsize=10, fontweight='bold')
    plt.ylabel("Estadio Real (AASM)", fontsize=10, fontweight='bold')
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    return output_path
