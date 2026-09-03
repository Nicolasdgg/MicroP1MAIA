"""
Arquitectura y entrenamiento de Deep Learning: 1D-CNN + BiLSTM (Inspirada en TinySleepNet)
para la clasificación automática de épocas de 30 segundos de EEG (100 Hz, 3000 muestras).
"""

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

def train_deep_sleep_net(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 15,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    figures_dir: Path = Path("reports/figures")
) -> Tuple[TinySleepNet1D, Dict[str, float]]:
    """Entrena la red TinySleepNet1D con pesos de clase y registra el progreso en MLflow."""
    configure_mlflow()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Deep Learning] Entrenando en dispositivo: {device}")
    
    # Preparar tensores
    X_train_t = torch.tensor(X_train[:, np.newaxis, :], dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val[:, np.newaxis, :], dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
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
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "class_weights_used": True,
        "device": str(device)
    }
    
    with mlflow.start_run(run_name="TinySleepNet_1D_BiLSTM"):
        mlflow.log_params(params)
        
        best_val_f1 = 0.0
        best_metrics = {}
        
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
            
            # Registrar métricas por época
            mlflow.log_metrics(metrics, step=ep)
            
            if metrics["f1_macro"] > best_val_f1:
                best_val_f1 = metrics["f1_macro"]
                best_metrics = metrics
                
            print(f"Época {ep+1:02d}/{epochs:02d} - Loss: {metrics['train_loss']:.4f} | Val Acc: {metrics['accuracy']:.4f} | F1 Macro: {metrics['f1_macro']:.4f}")
            
        # Generar matriz de confusión del mejor estado
        cm_path = figures_dir / "confusion_matrix_deep_learning.png"
        plot_confusion_matrix(y_val, val_preds, cm_path, title="Matriz de Confusión: TinySleepNet (CNN+BiLSTM)")
        mlflow.log_artifact(str(cm_path), artifact_path="evaluation_plots")
        
        # Guardar modelo en MLflow
        dummy_input = torch.randn(1, 1, 3000, dtype=torch.float32)
        try:
            mlflow.pytorch.log_model(model, artifact_path="model_tinysleepnet", input_example=dummy_input.numpy())
        except Exception as e:
            print(f"[Aviso MLflow] No se pudo serializar en formato pt2 ({e}), guardando state_dict.")
            torch.save(model.state_dict(), figures_dir.parent / "models" / "model_tinysleepnet.pt")
        
        return model, best_metrics
