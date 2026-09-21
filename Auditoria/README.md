# Proyecto: Análisis Comparativo de Predicción de Mercados con IA
## Economía + IA + Datasets Abiertos | Frontiers in Artificial Intelligence (Q1)

**Inicio:** Agosto 2026  
**Status:** En Desarrollo  
**Revista Target:** Frontiers in Artificial Intelligence (Q1)

---

## 📁 Estructura de Carpetas (Numeradas para Trazabilidad)

```
Auditoria/
├── 01_Planificación/          ← Plan maestro, RQs, metodología
├── 02_Descarga_Datos/         ← Raw datasets (Yahoo Finance, FRED)
├── 03_EDA_Exploratorio/       ← Jupyter notebooks, visualizaciones iniciales
├── 04_Preprocessing/          ← Data cleaning, feature engineering
├── 05_Modelado/               ← Entrenamiento LSTM, Transformer, XGBoost
├── 06_Evaluación/             ← Resultados por modelo, comparativas
├── 07_Análisis_Resultados/    ← SHAP, factores económicos, insights
├── 08_Paper_Manuscrito/       ← Draft, versiones, figures finales
├── 09_Código_Reproducible/    ← GitHub structure, requirements.txt
└── TRAZABILIDAD.md            ← Log de cambios y versiones
```

---

## 📋 Contenido por Carpeta

### 01_Planificación/
**Qué va aquí:** Plan maestro, RQs, hipótesis, calendario
- `paper_economia_ai_plan.md` → Plan completo del proyecto
- `RQs_y_hipotesis.txt` → Research Questions definidas
- `timeline.md` → Cronograma detallado
- `dataset_sources.txt` → URLs y instrucciones descarga

### 02_Descarga_Datos/
**Qué va aquí:** Datos crudos descargados
- `tech_prices_raw.csv` → AAPL, MSFT, NVDA, GOOGL (2021-2026)
- `energy_prices_raw.csv` → CVX, COP, XLE (2021-2026)
- `fred_economic_raw.csv` → Datos económicos FRED
- `download_log.txt` → Fecha/hora de descarga, versiones

### 03_EDA_Exploratorio/
**Qué va aquí:** Análisis exploratorio, visualizaciones iniciales
- `01_eda_tech_energy.ipynb` → Exploratory Data Analysis
- `02_descriptive_stats.ipynb` → Estadísticas descriptivas
- `figures/` → Gráficos preliminares (precios, correlaciones, VIX)
- `eda_report.md` → Resumen hallazgos iniciales

### 04_Preprocessing/
**Qué va aquí:** Datos limpios, procesados, features
- `01_data_cleaning.ipynb` → Limpieza, handling missing values
- `02_feature_engineering.ipynb` → Technical indicators, lag features
- `tech_features_processed.csv` → Datos listos para modelado
- `energy_features_processed.csv`
- `preprocessing_log.md` → Decisiones, transformaciones aplicadas

### 05_Modelado/
**Qué va aquí:** Modelos entrenados, logs, hyperparameters
- `01_lstm_training.ipynb` → Training LSTM (Tech + Energía)
- `02_transformer_training.ipynb` → Training Transformer
- `03_xgboost_training.ipynb` → Training XGBoost
- `04_ensemble_training.ipynb` → Ensemble model
- `models/` → Archivos .h5, .pkl (modelos guardados)
- `hyperparameters.json` → Config final de todos los modelos
- `training_log.md` → Epochs, loss, convergence notes

### 06_Evaluación/
**Qué va aquí:** Métricas, predicciones, comparativas
- `01_model_evaluation.ipynb` → Evaluación métrica por modelo
- `results_tech.csv` → MAE, RMSE, R², Directional Acc, Sharpe Ratio (Tech)
- `results_energy.csv` → Métricas Energía
- `predictions_vs_actual_tech.csv` → Predicciones vs valores reales
- `predictions_vs_actual_energy.csv`
- `evaluation_summary.md` → Tabla comparativa final

### 07_Análisis_Resultados/
**Qué va aquí:** Análisis profundo, explicabilidad, insights económicos
- `01_shap_analysis.ipynb` → Feature importance (SHAP values)
- `02_sector_differences.ipynb` → Análisis comparativo Tech vs Energía
- `03_economic_factors.ipynb` → Correlación features → performance
- `04_transfer_learning.ipynb` → ¿Modelo Tech predice Energía?
- `figures/` → SHAP plots, correlation heatmaps, feature importance
- `insights.md` → Hallazgos clave, explicaciones

### 08_Paper_Manuscrito/
**Qué va aquí:** Paper en desarrollo, figuras finales, versiones
- `v01_draft.docx` → Primera versión
- `v02_with_results.docx` → Con resultados
- `v03_final.docx` → Versión para envío
- `figures/` → Figuras de alta resolución (Fig 1-8)
- `supplementary_material/` → Tablas completas, código apéndice
- `submission_checklist.md` → Requerimientos Frontiers

### 09_Código_Reproducible/
**Qué va aquí:** Estructura GitHub-ready
```
├── data/
│   ├── raw/ → Links/instrucciones descarga
│   └── processed/ → Features finales
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_analysis.ipynb
├── src/
│   ├── models.py → Arquitecturas LSTM, Transformer
│   ├── utils.py → Funciones preprocessing, métricas
│   └── config.py → Hyperparameters centralizados
├── results/
│   ├── figures/ → Gráficos finales
│   └── metrics.json
├── requirements.txt → Dependencias Python
└── README.md → Cómo replicar en 5 pasos
```

---

## 🔍 Trazabilidad: TRAZABILIDAD.md

Cada análisis/cambio importante debe registrarse en `TRAZABILIDAD.md`:

```
[2026-08-04 14:30] - Creación estructura carpetas
[2026-08-05 09:00] - Descarga Yahoo Finance data (5 años, AAPL/MSFT/NVDA/CVX/COP/XLE)
[2026-08-05 14:00] - Descarga FRED API (inflación, tasa desempleo, VIX)
[2026-08-06 10:00] - EDA completado: Tech 2.3x más volátil que Energía
[2026-08-08 15:30] - LSTM entrenado Tech: MAE=0.032, Sharpe=1.2
...
```

---

## 🚀 Flujo de Trabajo Recomendado

1. **02_Descarga_Datos/** → Descargar todos los datasets
2. **03_EDA_Exploratorio/** → Visualizar, entender los datos
3. **04_Preprocessing/** → Limpiar, crear features
4. **05_Modelado/** → Entrenar todos los modelos
5. **06_Evaluación/** → Evaluar y comparar
6. **07_Análisis_Resultados/** → Explicar PORQUÉ
7. **08_Paper_Manuscrito/** → Escribir paper
8. **09_Código_Reproducible/** → Versionar para GitHub

---

## 📊 Versioning de Análisis

Cada análisis importante genera una **subfolder numerada** dentro de su carpeta:

Ejemplo en `05_Modelado/`:
```
05_Modelado/
├── v01_baseline_lstm/
│   ├── model.h5
│   ├── config.json
│   └── results.json
├── v02_lstm_dropout_tuning/
│   ├── model.h5
│   └── results.json
├── v03_lstm_final/
│   ├── model.h5
│   ├── results.json
│   └── notes.md
```

---

## ✅ Checklist de Completitud

- [ ] 01_Planificación: Plan maestro guardado
- [ ] 02_Descarga_Datos: Todos los datasets descargados
- [ ] 03_EDA_Exploratorio: EDA completo, figuras generadas
- [ ] 04_Preprocessing: Features listas
- [ ] 05_Modelado: Todos los modelos entrenados
- [ ] 06_Evaluación: Métricas calculadas
- [ ] 07_Análisis_Resultados: Insights documentados
- [ ] 08_Paper_Manuscrito: Draft listo
- [ ] 09_Código_Reproducible: GitHub-ready

---

## 📝 Notas Importantes

1. **TRAZABILIDAD:** Toda acción va en `TRAZABILIDAD.md` con timestamp
2. **Versioning:** v01, v02, v03... para cada iteración importante
3. **Reproducibilidad:** Cada carpeta debe contener TODO lo necesario
4. **Documentación:** README en cada subfolder explicando qué hay
5. **Backups:** Los datos raw NUNCA se borran (2_Descarga_Datos es inmutable)

---

**Última actualización:** 2026-08-04  
**Responsable:** Análisis Economía + IA | Frontiers Q1
