# Validación del pipeline: recuperación de señal conocida (corregida)

**Fecha:** 2026-08-05 22:23:05

## Corrección respecto a la versión anterior

La versión anterior de esta prueba usaba el NIVEL estandarizado del VIX como fuente de la
señal inyectada. Se detectó que el nivel del VIX tiene una deriva entre train y test en
estos datos sintéticos (media 31.0 en train; rango 11.4-29.9 en test), por lo que
estandarizar el nivel de test con estadísticos de train producía una distribución con
media=-1.81 y desvío=0.77 (no media=0/desvío=1 como se asumía), sesgando artificialmente el
signo del objetivo sintético en el conjunto de prueba y produciendo una precisión
direccional inflada de forma espuria (~97% en gamma=0.05, cuando el techo teórico para un
modelo perfecto era ~81%). Esta versión usa el CAMBIO DIARIO del VIX, que es
aproximadamente estacionario entre train y test (media=-0.013, desvío=1.049 en test al
estandarizar con estadísticos de train), y ejecuta cada nivel de gamma con 10
semillas de ruido independientes en lugar de una sola.

## Resultados (media +/- desvío estándar sobre 10 semillas)

```
 gamma  R2_mean  R2_sd  MAE_mean  MAE_sd  DirAcc_mean  DirAcc_sd  SHAP_VIXCLS_pct_mean  SHAP_VIXCLS_pct_sd  SHAP_VIXCLS_rank_mean  N_SEEDS
 0.000  -0.0026 0.0014    0.0266  0.0004       0.4971     0.0108                0.0397              0.0128                   10.9       10
 0.005   0.0069 0.0067    0.0266  0.0006       0.5332     0.0094                0.2955              0.0494                    1.0       10
 0.010   0.0701 0.0147    0.0267  0.0005       0.5857     0.0066                0.4451              0.0494                    1.0       10
 0.020   0.2597 0.0118    0.0275  0.0004       0.6730     0.0097                0.6252              0.0363                    1.0       10
 0.050   0.7085 0.0074    0.0277  0.0006       0.8187     0.0091                0.7508              0.0455                    1.0       10
 0.100   0.9052 0.0049    0.0279  0.0008       0.9032     0.0050                0.8285              0.0274                    1.0       10
```

## Interpretación

- R² crece monótonamente con gamma (en la media): True
- Importancia SHAP de VIXCLS crece monótonamente con gamma (en la media): True

Esto demuestra que el pipeline (escalado, entrenamiento, evaluación, SHAP) es capaz de
detectar una relación predictiva LINEAL con el cambio diario del VIX en el horizonte t->t+1,
mediante XGBoost, bajo este modelo de ruido gaussiano concreto, en esta partición de datos y
para este conjunto de semillas. La conclusión se limita estrictamente a esta configuración:
no se prueba la recuperación de relaciones no lineales, señales basadas en otras features,
señales localizadas en un régimen específico, ni el desempeño de LSTM/Transformer en esta
misma tarea de recuperación de señal.
