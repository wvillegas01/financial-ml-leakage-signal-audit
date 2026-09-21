# Análisis Detallado - Explicabilidad e Impacto (valores reales, no simulados)

**Fecha:** 2026-08-05 21:46:36

## 1. FEATURE IMPORTANCE (XGBoost `.feature_importances_`, real)

Top 5 features:
1. **CPIAUCSL** (8.7%)
2. **DCOILWTICO** (7.6%)
3. **FEDFUNDS** (7.6%)
4. **VIXCLS** (7.2%)
5. **DHHNGSP** (7.0%)

## 2. ANÁLISIS DE ERRORES (sobre predicciones reales del test set)

- LSTM: media=0.04, std=0.04, max=0.21
- Transformer: media=0.04, std=0.04, max=0.22
- XGBoost: media=0.04, std=0.04, max=0.23
- Ensemble: media=0.04, std=0.04, max=0.22

## 3. FACTORES ECONÓMICOS (correlación real retorno sectorial vs. cambio diario del factor)

                Factor  Impact_on_Tech  Impact_on_Energy
Oil Price (DCOILWTICO)          -0.042             0.841
             VIX Index          -0.819             0.058
          Unemployment          -0.008            -0.020
              Fed Rate           0.004             0.034
               S&P 500          -0.011             0.050

Factor con mayor |correlación| para Tech: **VIX Index** (-0.819)
Factor con mayor |correlación| para Energía: **Oil Price (DCOILWTICO)** (+0.841)

## 4. EXPLICABILIDAD (SHAP real vía TreeExplainer sobre XGBoost)

    Feature  Mean_Abs_SHAP  SHAP_Std
     UNRATE       0.002379  0.003139
     VIXCLS       0.001487  0.001541
     Vol_20       0.001478  0.002289
   CPIAUCSL       0.001243  0.001497
Volume_Lag1       0.001034  0.001624
Return_Lag5       0.000704  0.001440
      SP500       0.000624  0.000778
Return_Lag1       0.000561  0.001437
 Close_Lag5       0.000393  0.000741
   FEDFUNDS       0.000306  0.000431
    DHHNGSP       0.000273  0.000526
  Open_Lag1       0.000243  0.000386
 DCOILWTICO       0.000210  0.000260
     SMA_20       0.000161  0.000345
 Close_Lag1       0.000152  0.000206
     SMA_50       0.000115  0.000227
  High_Lag1       0.000081  0.000083
   Low_Lag1       0.000000  0.000000

Correlación entre valor de Close_Lag1 y su SHAP value: -0.2171
(un valor positivo confirma monotonicidad: a mayor Close_Lag1, mayor contribución SHAP a la predicción)

---

**Status:** Análisis completado con valores calculados directamente sobre el modelo y los datos.
