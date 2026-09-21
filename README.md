# Financial ML Leakage and Predictive Signal Audit

Reproducibility package for the manuscript:

**Auditing Information Leakage and Predictive Signal in Financial Machine Learning: A Controlled Study**

Authors: Ivan Ortiz-Garces, Aracely Mera-Navarrete, Paul Sarango-Lalangui, Jefferson Beltran-Morales, and William Villegas-Ch.

## Purpose

This repository contains the controlled synthetic benchmark, scripts, generated datasets, model outputs, robustness diagnostics, figures, and manuscript materials used to audit information leakage and controlled predictive-signal recovery in a financial machine learning pipeline.

The study is methodological rather than empirical. The benchmark is synthetic and does not contain real company observations or historical financial records.

## Repository Structure

- `Auditoria/02_Descarga_Datos/`: synthetic benchmark generation and raw generated series.
- `Auditoria/03_EDA_Exploratorio/`: exploratory analysis scripts and figures.
- `Auditoria/04_Preprocessing/`: preprocessing scripts and train/validation/test datasets.
- `Auditoria/05_Modelado/`: model-training scripts and model outputs.
- `Auditoria/06_Evaluacion/`: model evaluation scripts and derived metrics.
- `Auditoria/07_Analisis/`: error analysis, SHAP analysis, and figure scripts.
- `Auditoria/09_Validacion_Senal/`: controlled signal-injection validation.
- `Auditoria/10_Robustez_Minima/`: neural multiseed checks, moving-block bootstrap diagnostics, baseline metrics, and ensemble-weight sensitivity.
- `figures/`: final manuscript figures.
- `manuscript/`: LaTeX source, compiled manuscript PDF, and bibliography used in the manuscript.
- `MANIFEST.md`: package inventory and verification note.

## Key Reproducibility Components

The package includes:

- economic-factor generation;
- GARCH-like returns for nine benchmark assets;
- OHLCV construction;
- seed assignment;
- signal injection;
- gamma-by-seed recovery loop;
- XGBoost training;
- SHAP calculation;
- neural-network multiseed sensitivity checks;
- moving-block bootstrap diagnostics for directional accuracy.

## Suggested Execution Order

The scripts are organized according to the original audit workflow:

1. Generate the controlled benchmark:
   `Auditoria/02_Descarga_Datos/generate_synthetic_data.py`

2. Run preprocessing:
   `Auditoria/04_Preprocessing/preprocessing_final.py`

3. Train models:
   `Auditoria/05_Modelado/modeling.py`

4. Evaluate model outputs:
   `Auditoria/06_Evaluacion/evaluation.py`

5. Run SHAP and result analyses:
   `Auditoria/07_Analisis/analysis.py`
   `Auditoria/07_Analisis/generate_main_xgb_shap_table.py`

6. Run controlled signal recovery:
   `Auditoria/09_Validacion_Senal/validate_signal_recovery_paired.py`

7. Run robustness diagnostics:
   `Auditoria/10_Robustez_Minima/minimum_revision_analysis.py`

## Environment

The scripts were developed for Python 3.x. Main dependencies are listed in `requirements.txt`.

TensorFlow results can vary slightly across hardware and backend versions. The manuscript reports neural sensitivity using 10 fixed seeds to reduce reliance on a single stochastic initialization.

## Data Statement

All financial series in this repository are computer-generated benchmark data. Asset identifiers such as `TECH-1` or `ENERGY-1` are synthetic labels and do not correspond to real companies.

## Citation

If using this package, please cite the manuscript and the archived Zenodo DOI once available. Citation metadata is provided in `CITATION.cff`.
