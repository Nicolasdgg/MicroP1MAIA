# Datos — SomnoScope (Sleep-EDF Database Expanded)

Esta carpeta contiene las **muestras ligeras y la estructura documentada** de los datos utilizados en SomnoScope. Los archivos binarios completos de polisomnografía (`.edf`, varios GB) se gestionan fuera de Git mediante **DVC (`data.dvc`)** y/o descarga directa desde PhysioNet.

## Fuente de datos
- **Repositorio oficial:** [Sleep-EDF Database Expanded (PhysioNet)](https://physionet.org/content/sleep-edfx/1.0.0/)
- **Cohorte evaluada en Validación Cruzada (`3-Fold GroupKFold`):** **23 sujetos únicos** (`20 Sleep Cassette SC400–SC419` + `3 Sleep Telemetry ST701, ST702, ST704`), **45 noches completas**, **48,378 épocas válidas de 30 s**.
- **Canal estándar empleado:** `EEG Fpz-Cz` (frecuencia de muestreo `100 Hz`, 3,000 muestras por época, filtro pasa-banda FIR `0.5–35 Hz`).

## Distribución de clases (Estándar AASM)

| `stage_id` | Estadio AASM | Descripción clínica | Épocas (23 sujetos) | Proporción |
| :---: | :---: | :--- | :---: | :---: |
| `0` | **W** | Vigilia (*Wake*, recortada ±30 min) | 8,837 | 18.27% |
| `1` | **N1** | Sueño ligero transicional (clase minoritaria) | 3,211 | 6.64% |
| `2` | **N2** | Sueño intermedio (husos de sueño / complejos K) | 20,837 | 43.07% |
| `3` | **N3** | Sueño profundo de ondas lentas (`S3 + S4`) | 6,705 | 13.86% |
| `4` | **REM** | Sueño de Movimientos Oculares Rápidos | 8,788 | 18.17% |
| — | **Total** | **Cohorte completa evaluada (OOF)** | **48,378** | **100.00%** |

## Archivos incluidos en esta carpeta
1. `metadatos_sujetos.csv`: Metadatos y conteos por estudio de la cohorte versionada en DVC.
2. `muestra_caracteristicas_30_epocas.csv`: Muestra real de 30 épocas etiquetadas con las **25 características fisiológicas** (potencias absolutas/relativas Welch PSD en bandas delta/theta/alfa/sigma/beta, razones espectrales, entropía espectral, parámetros de Hjorth y estadísticos temporales) que consume el endpoint `/predict/features` y el modelo Challenger LightGBM.

## Cómo descargar los registros `.edf` completos
- **Opción A (DVC / S3):** Configurar credenciales AWS en `.env` (ver `.env.example`) y ejecutar:
  ```bash
  dvc pull
  ```
- **Opción B (PhysioNet vía MNE-Python):** Los scripts y notebooks (`Notebooks/run_groupkfold_cv_23subjects.ipynb`) descargan automáticamente las grabaciones desde PhysioNet usando `mne.datasets.sleep_physionet.age.fetch_data`.
