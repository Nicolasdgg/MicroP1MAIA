"""
Módulo de extracción de características espectrales (Welch PSD),
ratios de frecuencia y métricas temporales/Hjorth para señales de EEG (épocas de 30s).
"""

from typing import Dict, List, Tuple
import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis

FREQ_BANDS = {
    "Delta": (0.5, 4.0),
    "Theta": (4.0, 8.0),
    "Alpha": (8.0, 12.0),
    "Sigma": (12.0, 16.0),
    "Beta": (16.0, 30.0)
}

FEATURE_NAMES = [
    "delta_abs", "theta_abs", "alpha_abs", "sigma_abs", "beta_abs", "total_power",
    "delta_rel", "theta_rel", "alpha_rel", "sigma_rel", "beta_rel",
    "ratio_delta_theta", "ratio_theta_alpha", "ratio_slow_fast",
    "stat_mean", "stat_std", "stat_variance", "stat_skew", "stat_kurtosis",
    "stat_ptp", "stat_energy", "stat_zcr",
    "hjorth_activity", "hjorth_mobility", "hjorth_complexity"
]

def extract_epoch_features(epoch_signal: np.ndarray, sfreq: float = 100.0) -> np.ndarray:
    """
    Extrae un vector de 25 características discriminativas para una época de 30 segundos (3000 muestras).
    Aplica normalización z-score para robustez ante variabilidad de impedancia inter-sujeto.
    """
    # Normalización robusta z-score
    epoch_norm = (epoch_signal - np.mean(epoch_signal)) / (np.std(epoch_signal) + 1e-8)
    feats = []
    
    # 1. Welch PSD
    freqs, psd = signal.welch(epoch_norm, fs=sfreq, nperseg=int(4.0 * sfreq), noverlap=int(2.0 * sfreq))
    df = freqs[1] - freqs[0]
    
    # Potencias absolutas por banda
    band_powers = {}
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        idx = np.where((freqs >= fmin) & (freqs < fmax))[0]
        p_abs = np.sum(psd[idx]) * df if len(idx) > 0 else 1e-12
        band_powers[band_name] = max(p_abs, 1e-12)
        feats.append(band_powers[band_name])
        
    total_power = sum(band_powers.values())
    feats.append(total_power)
    
    # Potencias relativas (%)
    delta_rel = band_powers["Delta"] / total_power
    theta_rel = band_powers["Theta"] / total_power
    alpha_rel = band_powers["Alpha"] / total_power
    sigma_rel = band_powers["Sigma"] / total_power
    beta_rel = band_powers["Beta"] / total_power
    feats.extend([delta_rel, theta_rel, alpha_rel, sigma_rel, beta_rel])
    
    # Ratios espectrales clínicos
    ratio_dt = band_powers["Delta"] / band_powers["Theta"]
    ratio_ta = band_powers["Theta"] / band_powers["Alpha"]
    ratio_sf = (band_powers["Delta"] + band_powers["Theta"]) / (band_powers["Alpha"] + band_powers["Beta"])
    feats.extend([ratio_dt, ratio_ta, ratio_sf])
    
    # 2. Estadísticas en el dominio temporal
    s_mean = np.mean(epoch_signal)
    s_std = np.std(epoch_signal)
    s_var = np.var(epoch_signal)
    s_skew = float(skew(epoch_signal))
    s_kurt = float(kurtosis(epoch_signal))
    s_ptp = np.ptp(epoch_signal)
    s_energy = np.sum(epoch_signal ** 2) / len(epoch_signal)
    
    # Zero Crossing Rate (ZCR)
    zero_crosses = np.nonzero(np.diff(epoch_signal > 0))[0]
    s_zcr = len(zero_crosses) / len(epoch_signal)
    
    feats.extend([s_mean, s_std, s_var, s_skew, s_kurt, s_ptp, s_energy, s_zcr])
    
    # 3. Parámetros de Hjorth (Actividad, Movilidad, Complejidad)
    d1 = np.diff(epoch_signal)
    d2 = np.diff(d1)
    
    activity = s_var
    mobility = np.sqrt(np.var(d1) / (activity + 1e-12))
    complexity = (np.sqrt(np.var(d2) / (np.var(d1) + 1e-12))) / (mobility + 1e-12)
    
    feats.extend([activity, mobility, complexity])
    
    return np.array(feats, dtype=np.float32)

def extract_features_dataset(X_raw: np.ndarray, sfreq: float = 100.0) -> np.ndarray:
    """
    Extrae características para una matriz completa de épocas (N, 3000) -> (N, 25).
    """
    n_epochs = len(X_raw)
    X_feats = np.zeros((n_epochs, len(FEATURE_NAMES)), dtype=np.float32)
    
    for i in range(n_epochs):
        X_feats[i] = extract_epoch_features(X_raw[i], sfreq=sfreq)
        
    return X_feats
