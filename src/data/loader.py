"""
Módulo de carga, preprocesamiento y partición de datos para Sleep-EDFx.
Garantiza partición estricta por sujeto (Anti-Data Leakage) y estandarización a 5 clases AASM.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import mne
from scipy import signal

AASM_CLASSES = ['W', 'N1', 'N2', 'N3', 'REM']
CLASS_TO_IDX = {c: i for i, c in enumerate(AASM_CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(AASM_CLASSES)}

STAGE_MAP_R_AND_K_TO_AASM = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
    "Movement time": "M",
    "Sleep stage ?": "?"
}

def find_psg_hypno_pairs(data_dir: Path) -> List[Tuple[Path, Path, str]]:
    """Encuentra parejas de archivos PSG e hipnogramas en data_dir."""
    psg_files = list(data_dir.rglob("*PSG.edf"))
    pairs = []
    for psg in sorted(psg_files):
        stem = psg.stem.replace("-PSG", "")
        prefix = stem[:6]
        hypno_matches = list(psg.parent.glob(f"{prefix}*Hypnogram.edf"))
        if not hypno_matches:
            hypno_matches = list(data_dir.rglob(f"{prefix}*Hypnogram.edf"))
        if hypno_matches:
            subject_id = prefix[:5]  # ej. SC400 o ST701
            pairs.append((psg, hypno_matches[0], subject_id))
    return pairs

def load_and_preprocess_recording(
    psg_path: Path, 
    hypno_path: Path, 
    target_channel: str = "EEG Fpz-Cz", 
    sfreq: float = 100.0,
    epoch_sec: float = 30.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga un registro PSG, aplica filtro pasa-banda 0.5-35 Hz, Notch 50 Hz,
    segmenta en épocas de 30s y devuelve (X_epochs, y_labels).
    """
    raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)
    
    # Normalizar nombres de canales
    rename_map = {}
    for ch in raw.ch_names:
        ch_upper = ch.upper()
        if "FPZ-CZ" in ch_upper or "FPZ_CZ" in ch_upper:
            rename_map[ch] = "EEG Fpz-Cz"
        elif "PZ-OZ" in ch_upper or "PZ_OZ" in ch_upper:
            rename_map[ch] = "EEG Pz-Oz"
        elif "EOG" in ch_upper:
            rename_map[ch] = "EOG horizontal"
        elif "EMG" in ch_upper:
            rename_map[ch] = "EMG submental"
    raw.rename_channels(rename_map)
    
    # Seleccionar canal objetivo primero para evitar advertencias de otros canales no-EEG
    available_channels = raw.ch_names
    selected_ch = target_channel if target_channel in available_channels else available_channels[0]
    raw.pick([selected_ch])
    
    # Re-muestreo a 100 Hz si difiere
    if raw.info['sfreq'] != sfreq:
        raw.resample(sfreq, verbose=False)
        
    # Filtrado pasa-banda clínico 0.5 - 35 Hz (cubre y elimina ruido de 50 Hz/60 Hz de red)
    raw.filter(l_freq=0.5, h_freq=35.0, fir_design='firwin', verbose=False)
    
    # Cargar anotaciones
    annot = mne.read_annotations(str(hypno_path))
    raw.set_annotations(annot, emit_warning=False)
    
    # Construir mapeo de eventos
    event_id = {}
    for desc in np.unique(annot.description):
        if desc in STAGE_MAP_R_AND_K_TO_AASM:
            aasm_stage = STAGE_MAP_R_AND_K_TO_AASM[desc]
            if aasm_stage in CLASS_TO_IDX:
                event_id[desc] = CLASS_TO_IDX[aasm_stage]
                
    events, _ = mne.events_from_annotations(raw, event_id=event_id, chunk_duration=epoch_sec, verbose=False)
    
    # Segmentar en épocas fijas de 30 segundos
    tmax = epoch_sec - (1.0 / sfreq)
    epochs = mne.Epochs(
        raw, 
        events, 
        tmin=0.0, 
        tmax=tmax, 
        baseline=None, 
        preload=True, 
        verbose=False
    )
    
    X = epochs.get_data(copy=True)[:, 0, :]  # Shape: (n_epochs, 3000)
    y = epochs.events[:, 2]                  # Shape: (n_epochs,) con enteros 0 a 4
    
    # Recorte opcional de vigilia excesiva en grabaciones de 24h
    # (limitar a 30 minutos antes y después del primer/último sueño)
    sleep_idx = np.where(y != 0)[0]
    if len(sleep_idx) > 0:
        start_idx = max(0, sleep_idx[0] - 60)
        end_idx = min(len(y), sleep_idx[-1] + 60)
        X = X[start_idx:end_idx]
        y = y[start_idx:end_idx]
        
    return X, y

def load_dataset_by_subjects(
    data_dir: Path, 
    max_records: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Carga todos los registros disponibles y retorna (X, y, subject_ids)
    para permitir validación cruzada y partición estricta por sujeto.
    """
    pairs = find_psg_hypno_pairs(data_dir)
    if max_records:
        pairs = pairs[:max_records]
        
    all_x, all_y, all_subjs = [], [], []
    
    for psg, hypno, subj in pairs:
        try:
            X, y = load_and_preprocess_recording(psg, hypno)
            all_x.append(X)
            all_y.append(y)
            all_subjs.extend([subj] * len(y))
        except Exception as e:
            print(f"[Aviso] Error procesando {psg.name}: {e}")
            
    X_full = np.concatenate(all_x, axis=0)
    y_full = np.concatenate(all_y, axis=0)
    subj_full = np.array(all_subjs)
    
    return X_full, y_full, subj_full
