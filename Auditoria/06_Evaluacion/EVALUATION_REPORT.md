# Evaluación - Análisis Comparativo

**Fecha:** 2026-08-05 21:45:17

## 1. RESULTADOS GLOBALES

```
         Model     MAE    RMSE      R2  Directional_Accuracy
0         LSTM  0.0424  0.0559 -0.0059                0.4892
1  Transformer  0.0431  0.0568 -0.0364                0.4885
2      XGBoost  0.0422  0.0559 -0.0052                0.4953
3     Ensemble  0.0423  0.0560 -0.0072                0.4986
```

## 2. PRUEBAS ESTADÍSTICAS FORMALES (precisión direccional)

```
      Model    N  Directional_Accuracy  CI_95_Low  CI_95_High  Majority_Class_Baseline  p_value_vs_chance_0.5  p_value_vs_majority_baseline  Significant_vs_chance_a0.05  Significant_vs_majority_a0.05
       LSTM 1476                0.4892     0.4637      0.5147                   0.5014                 0.4197                        0.3489                        False                          False
Transformer 1476                0.4885     0.4630      0.5140                   0.5014                 0.3904                        0.3228                        False                          False
    XGBoost 1476                0.4953     0.4698      0.5207                   0.5014                 0.7351                        0.6395                        False                          False
   Ensemble 1476                0.4986     0.4732      0.5241                   0.5014                 0.9378                        0.8351                        False                          False
```

De los 4 modelos evaluados, 0 muestran una precisión
direccional estadísticamente distinguible del azar (p<0.05, prueba binomial de dos colas
contra p=0.5), y 0 son distinguibles de la línea base de clase mayoritaria
observada en el conjunto de prueba.

## 3. ANÁLISIS POR SECTOR (real)

```
    Sector       Model   N    MAE   RMSE      R2  Directional_Accuracy
Tecnología        LSTM 820 0.0475 0.0632  0.0035                0.5122
Tecnología Transformer 820 0.0489 0.0647 -0.0442                0.4951
Tecnología     XGBoost 820 0.0478 0.0637 -0.0106                0.4854
Tecnología    Ensemble 820 0.0476 0.0635 -0.0052                0.5146
   Energía        LSTM 656 0.0359 0.0452 -0.0299                0.4604
   Energía Transformer 656 0.0358 0.0449 -0.0171                0.4802
   Energía     XGBoost 656 0.0352 0.0444  0.0079                0.5076
   Energía    Ensemble 656 0.0357 0.0448 -0.0127                0.4787
```

## 4. CONCLUSIONES

### Mejor R²: XGBoost
- **R² = -0.0052**
- **MAE = 0.0422**
- **Dir Acc = 49.53%**

### Mejor Directional Accuracy: Ensemble
- **Dir Acc = 49.86%**

### Menor R²: Transformer
- R² = -0.0364
- MAE = 0.0431

---

**Status:** Evaluación completada. Todos los valores (globales, por sector, pruebas
estadísticas) se calculan dinámicamente desde metrics.csv/predictions.csv.
**Próximo:** Análisis SHAP en 07_Análisis/
