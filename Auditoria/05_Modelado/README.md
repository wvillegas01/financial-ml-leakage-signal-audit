# 05_Modelado

## Contenido

Entrenamiento de 4 arquitecturas de Deep Learning y Machine Learning para predicción de precios.

### Modelos Entrenados

| Modelo | Tipo | Arquitectura | Objetivo |
|--------|------|-------------|----------|
| **LSTM** | Deep Learning | 2 capas LSTM (64→32 units) + Dropout | Capturar dependencias temporales |
| **Transformer** | Deep Learning | Multi-Head Attention + Dense layers | Capturar relaciones a largo plazo |
| **XGBoost** | ML Tradicional | Gradient Boosting (100 trees, depth=5) | Baseline robusto |
| **Ensemble** | Combinación | Voting (0.4 LSTM + 0.4 Transformer + 0.2 XGBoost) | Robustez combinada |

---

## 📊 Archivos Generados

| Archivo | Descripción |
|---------|-------------|
| `metrics.csv` | Comparativa de métricas (MAE, RMSE, R², Dir Acc) |
| `predictions.csv` | Predicciones de todos los modelos en test set |
| `lstm_model.h5` | Modelo LSTM entrenado (Keras) |
| `transformer_model.h5` | Modelo Transformer entrenado (Keras) |
| `xgb_model.json` | Modelo XGBoost entrenado |
| `modeling_log.txt` | Log de entrenamiento con timestamps |

---

## ⚙️ Configuración de Entrenamiento

**Hyperparameters:**
- Sequence Length: 30 días
- Batch Size: 32
- Epochs: 30
- Train/Val/Test: 70/15/15

**Features Utilizados:**
- OHLCV (Open, High, Low, Close, Volume)
- Return (retorno diario)
- SMA_20, SMA_50 (medias móviles)
- Vol_20 (volatilidad anualizada)
- Close_Lag1, Close_Lag5, Return_Lag1 (features rezagados)

---

## 📈 Métricas de Evaluación

- **MAE** (Mean Absolute Error): Error absoluto promedio
- **RMSE** (Root Mean Squared Error): Raíz del error cuadrático promedio
- **R²**: Coeficiente de determinación
- **Directional Accuracy**: % de veces que predice dirección correcta (↑ o ↓)

---

## 🔄 Comparativa Esperada

### Por Arquitectura

| Aspecto | LSTM | Transformer | XGBoost | Ensemble |
|--------|------|-------------|---------|----------|
| **Complejidad** | Media | Alta | Baja | Media |
| **Interpretabilidad** | Baja | Baja | Alta | Media |
| **Velocidad** | Media | Lenta | Rápida | Media |
| **Overfitting** | Posible | Posible | Bajo | Bajo |

### Por Sector (Expectativa)

**Tech (más volátil):**
- Mejor: Transformers (capturan cambios abruptos)
- Peor: XGBoost (asume relaciones lineales)

**Energía (más predecible):**
- Mejor: XGBoost (correlación con petróleo)
- Peor: LSTM (suficiencia de features simples)

---

## 🚀 Próximos Pasos

1. **06_Evaluación/** → Análisis de métricas por sector
2. **07_Análisis/** → SHAP, explicabilidad, factores económicos
3. **08_Paper/** → Redacción manuscrito

---

**Status:** ⏳ Entrenamiento en progreso  
**Timestamp:** 2026-08-04 20:15+
**Duración estimada:** 10-15 minutos
