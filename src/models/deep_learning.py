"""
Arquitectura y entrenamiento de Deep Learning: 1D-CNN + BiLSTM (Inspirada en TinySleepNet)
para la clasificación automática de épocas de 30 segundos de EEG (100 Hz, 3000 muestras).
"""

import copy
from pathlib import Path
from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import mlflow
import mlflow.pytorch

from src.models.evaluate import compute_sleep_metrics, plot_confusion_matrix
from src.utils.mlflow_client import configure_mlflow

class TinySleepNet1D(nn.Module):
    """
    Red neuronal convolucional y recurrente para señales temporales de polisomnografía.
    Combina representación de características espectrales locales (1D-CNN) con
    dinámica temporal secuencial (BiLSTM).
    """
    def __init__(self, in_channels: int = 1, n_classes: int = 5):
        super().__init__()
        
        # Extractor de características espaciotemporales (1D-CNN)
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=50, stride=6, padding=25),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(0.2)
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )
        
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=4, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
            nn.Dropout(0.3)
        )
        
        # Módulo Recurrente Bi-direccional
        self.bilstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Clasificador de 5 estadios AASM
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, 1, 3000)
        c1 = self.conv_block1(x)
        c2 = self.conv_block2(c1)
        c3 = self.conv_block3(c2)  # (batch_size, 128, 32)
        
        # Transponer para LSTM: (batch_size, seq_len=32, features=128)
        lstm_in = c3.permute(0, 2, 1)
        lstm_out, _ = self.bilstm(lstm_in)
        
        # Tomar el último paso temporal o pooling global
        feat = lstm_out[:, -1, :]  # (batch_size, 128)
        logits = self.fc(feat)     # (batch_size, 5)
        return logits

def normalize_epochs(X: np.ndarray) -> np.ndarray:
    """Z-score por época: (x - media) / desviación estándar de cada época."""
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim == 1:
        mu = arr.mean()
        sd = arr.std() + 1e-12
        return ((arr - mu) / sd).astype(np.float32)
    mu = arr.mean(axis=1, keepdims=True)
    sd = arr.std(axis=1, keepdims=True) + 1e-12  # la señal viene en voltios (~1e-5)
    return ((arr - mu) / sd).astype(np.float32)


def predict_deep_sleep_net(
    model: TinySleepNet1D,
    X_epochs: np.ndarray,
    batch_size: int = 128,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ejecuta inferencia por lotes con TinySleepNet1D aplicando normalización z-score por época.
    Retorna (y_pred, y_probs) con formas (N,) y (N, 5).
    """
    arr = np.asarray(X_epochs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
    X_norm = normalize_epochs(arr)[:, np.newaxis, :]

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(X_norm), batch_size):
            batch = torch.tensor(X_norm[start:start + batch_size], dtype=torch.float32, device=device)
            logits = model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

    y_probs = np.concatenate(all_probs, axis=0)
    y_pred = np.argmax(y_probs, axis=1).astype(int)
    return y_pred, y_probs


def train_deep_sleep_net(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    figures_dir: Path = Path("reports/figures"),
    random_state: int = 42,
    log_to_mlflow: bool = True,
) -> Tuple[TinySleepNet1D, Dict[str, float]]:
    """Entrena la red TinySleepNet1D con pesos de clase, checkpointing de mejor época y registro opcional en MLflow."""
    if log_to_mlflow:
        configure_mlflow()

    torch.manual_seed(random_state)
    np.random.seed(random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_state)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Deep Learning] Entrenando en dispositivo: {device}")

    # Preparar tensores. La señal MNE viene en voltios (~1e-5): se normaliza cada época con z-score
    # para que la red pueda aprender (sin esto colapsa hacia una sola clase).
    X_train_t = torch.tensor(normalize_epochs(X_train)[:, np.newaxis, :], dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(normalize_epochs(X_val)[:, np.newaxis, :], dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)

    gen = torch.Generator().manual_seed(random_state)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True, generator=gen)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

    # Calcular pesos de clase para mitigar el desbalance severo de N1
    class_counts = np.bincount(y_train, minlength=5)
    class_weights = torch.tensor(
        len(y_train) / (5.0 * np.maximum(class_counts, 1.0)),
        dtype=torch.float32
    ).to(device)

    model = TinySleepNet1D(in_channels=1, n_classes=5).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    params = {
        "model_type": "TinySleepNet1D_CNN_BiLSTM",
        "normalization": "z-score_per_epoch",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "class_weights_used": True,
        "random_state": random_state,
        "device": str(device),
    }

    def _run_training():
        best_val_f1 = -1.0
        best_metrics: Dict[str, float] = {}
        best_state_dict = copy.deepcopy(model.state_dict())
        best_val_preds = None
        best_epoch = 1

        for ep in range(epochs):
            model.train()
            total_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                out = model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # Evaluación de validación
            model.eval()
            all_preds = []
            with torch.no_grad():
                for batch_x, _ in val_loader:
                    batch_x = batch_x.to(device)
                    out = model(batch_x)
                    preds = torch.argmax(out, dim=1).cpu().numpy()
                    all_preds.extend(preds)

            val_preds = np.array(all_preds)
            metrics = compute_sleep_metrics(y_val, val_preds)
            metrics["train_loss"] = total_loss / len(train_loader)

            if log_to_mlflow:
                mlflow.log_metrics(metrics, step=ep)

            if metrics["f1_macro"] > best_val_f1:
                best_val_f1 = metrics["f1_macro"]
                best_metrics = dict(metrics)
                best_state_dict = copy.deepcopy(model.state_dict())
                best_val_preds = val_preds.copy()
                best_epoch = ep + 1

            print(
                f"Época {ep+1:02d}/{epochs:02d} - Loss: {metrics['train_loss']:.4f} | "
                f"Val Acc: {metrics['accuracy']:.4f} | F1 Macro: {metrics['f1_macro']:.4f} | "
                f"Kappa: {metrics['cohen_kappa']:.4f} | F1-N1: {metrics['f1_N1']:.4f}"
            )

        # Restaurar los pesos exactos de la mejor época (checkpointing)
        model.load_state_dict(best_state_dict)
        best_metrics["best_epoch"] = float(best_epoch)
        print(
            f"[Checkpointing] Restaurado mejor estado (Época {best_epoch:02d}): "
            f"Acc={best_metrics['accuracy']:.4f} | F1-Macro={best_metrics['f1_macro']:.4f} | "
            f"Kappa={best_metrics['cohen_kappa']:.4f} | F1-N1={best_metrics['f1_N1']:.4f}"
        )

        # Generar matriz de confusión del mejor estado
        figures_dir.mkdir(parents=True, exist_ok=True)
        cm_path = figures_dir / "confusion_matrix_deep_learning.png"
        plot_confusion_matrix(
            y_val,
            best_val_preds if best_val_preds is not None else val_preds,
            cm_path,
            title=f"Matriz de Confusión: TinySleepNet Normalizado (Época {best_epoch})",
        )
        if log_to_mlflow:
            mlflow.log_metrics({f"best_{k}": v for k, v in best_metrics.items()})
            mlflow.log_artifact(str(cm_path), artifact_path="evaluation_plots")
            dummy_input = torch.randn(1, 1, 3000, dtype=torch.float32)
            try:
                mlflow.pytorch.log_model(model, artifact_path="model_tinysleepnet", input_example=dummy_input.numpy())
            except Exception as e:
                print(f"[Aviso MLflow] No se pudo serializar en formato pt2 ({e}), guardando state_dict.")
                torch.save(model.state_dict(), figures_dir.parent / "Modelos" / "model_tinysleepnet.pt")

        return model, best_metrics

    if log_to_mlflow:
        with mlflow.start_run(run_name="TinySleepNet_1D_BiLSTM_Normalized"):
            mlflow.log_params(params)
            return _run_training()
    else:
        return _run_training()
