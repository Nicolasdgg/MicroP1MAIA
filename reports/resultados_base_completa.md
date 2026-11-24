# Resultados con la base completa de Sleep-EDF Expanded

Corrida del pipeline del repositorio sobre los **197 registros** de Sleep-EDF Expanded v1.0.0
(153 Sleep Cassette + 44 Sleep Telemetry, ~100 sujetos), descargados con `scripts/descargar_sleep_edfx.py`
y verificados contra `data/raw/SHA256SUMS.txt`. Incluye la normalización por época de TinySleepNet.

- Épocas procesadas: **237,802** (canal Fpz-Cz, 5 clases AASM, vigilia recortada a ±30 min).
- Partición por sujeto (`GroupShuffleSplit`, semilla 42), **sin validación cruzada**:
  - `run_experiments_pipeline.py`: 67 % entrenamiento / 33 % prueba.
  - `train_and_register_v2_models.py`: 70 % entrenamiento / 30 % prueba (72,233 épocas de prueba).
- Equipo: Intel i5-13400 (10 núcleos), 32 GB RAM, CPU (sin GPU). MLflow local (SQLite).

## Tiempos

| Etapa | Tiempo |
|---|---|
| Descarga de 358 archivos (7.5 GB) | 4.6 min |
| Lectura y preprocesamiento de 197 EDF | ~19 min |
| Guardado de la caché de épocas (5.5 GB) | ~4 min |
| Características + RF + LightGBM + TinySleepNet (8 epochs, CPU) | ~20 min |
| Versiones v1/v2 y registro en MLflow | 7.4 min |
| **Total sin descarga** | **~51 min** |

## Métricas en prueba

`train_and_register_v2_models.py` (partición 70/30):

| Modelo | Exactitud | F1-macro | Kappa | F1-N1 | F1-N3 | F1-REM |
|---|---|---|---|---|---|---|
| Random Forest v1 | 72.3 % | 0.673 | 0.634 | 0.354 | 0.786 | 0.589 |
| Random Forest v2 | 74.2 % | 0.680 | 0.653 | 0.323 | 0.810 | 0.601 |
| LightGBM v1 | 72.4 % | 0.676 | 0.636 | 0.368 | 0.778 | 0.591 |
| LightGBM v2 | 72.5 % | 0.677 | 0.637 | 0.372 | 0.779 | 0.590 |

`run_experiments_pipeline.py` (partición 67/33):

| Modelo | Exactitud | F1-macro | Kappa | F1-N1 | F1-N3 | F1-REM |
|---|---|---|---|---|---|---|
| Random Forest | 72.7 % | 0.677 | 0.639 | 0.364 | 0.784 | 0.597 |
| LightGBM | 72.8 % | 0.680 | 0.641 | 0.377 | 0.775 | 0.597 |
| TinySleepNet (normalizada, mejor epoch) | 77.2 % | 0.720 | 0.699 | 0.439 | 0.723 | 0.741 |

## Advertencias

- **TinySleepNet en `tabla_comparativa_modelos_v1_vs_v2.csv` no es real**: `train_and_register_v2_models.py`
  registra métricas fijas (50.2 %, Kappa 0). Su evaluación real es la de `tabla_comparativa_modelos.csv`.
- El mejor epoch de TinySleepNet y la comparación v1/v2 se eligen sobre el mismo conjunto de prueba en el que se
  reportan las métricas; las cifras son algo optimistas. Las diferencias entre modelos clásicos (< 0.01 en F1)
  no son concluyentes con una sola partición.
- El script asigna `@champion` a LightGBM v1 por código, aunque con estos datos Random Forest v2 obtiene mejores
  métricas globales. `models/best_sleep_model.pkl` queda con LightGBM v2 (lo que guarda el script).
