# Artículo: Análisis Comparativo de Predicción de Mercados con IA
## Economía + IA + Datasets Abiertos | Frontiers in Artificial Intelligence (Q1)

---

## 1. RESEARCH QUESTIONS (RQs)

**RQ1:** ¿Los modelos LSTM/Transformers/Ensemble predicen con igual precisión en mercados de Tech vs Energía?

**RQ2:** ¿Qué factores económicos explican las diferencias en performance entre sectores?

**RQ3:** ¿Cuál arquitectura (LSTM vs Transformer vs Ensemble) es más robusta cross-sector?

**RQ4:** ¿Las predicciones tienen valor económico (Sharpe Ratio, ROI) diferente por sector?

---

## 2. DATASETS ABIERTOS (100% reproducible)

### 2.1 Precios de Acciones (Tech + Energía)

**Opción A: Yahoo Finance API** (Recomendado - simple, gratis)
- URL: https://pypi.org/project/yfinance/
- Tickers Tech: AAPL, MSFT, NVDA, GOOGL, TSLA
- Tickers Energía: XLE (Energy ETF), CVX, COP, MPC
- Período: Últimos 5 años (2021-2026)
- Frecuencia: Diaria (Open, High, Low, Close, Volume)

**Opción B: Kaggle Massive Yahoo Finance Dataset**
- URL: https://www.kaggle.com/datasets/iveeaten3223times/massive-yahoo-finance-dataset
- Contiene: Top 500 empresas + clasificación por sector
- Período: 2015-2023
- Formato: CSV descargable

**Opción C: FirstRate Data** (25 años disponibles)
- URL: https://firstratedata.com/
- Granularidad: Intraday (1-min a 1-hour) + Daily
- 16,272 tickers disponibles

### 2.2 Datos Contextuales Económicos (FRED API)

**Federal Reserve Economic Data (FRED)**
- URL: https://fred.stlouisfed.org/
- Requiere: API key gratis (5 min para registrarse)
- Python Library: `pip install fredapi`

**Series relevantes a descargar:**

**Para Tech:**
- S&P 500 Technology Index (SP500): `SP500`
- Consumer Confidence Index: `UMCSENT`
- Unemployment Rate: `UNRATE`
- Federal Funds Rate: `FEDFUNDS`

**Para Energía:**
- Crude Oil WTI Spot Price: `DCOILWTICO`
- Natural Gas Spot Price: `DHHNGSP`
- Energy Producer Price Index: `PPGIGEE`
- Gasoline Price: `GASDESW`

### 2.3 Datos de Volatilidad (VIX, CBOE)

- VIX Index: https://www.cboe.com/data/
- FRED: Implied Volatility of S&P 500: `VIXCLS`
- Energía: HO (Heating Oil) Volatility

---

## 3. ESTRUCTURA DEL PAPER

### 3.1 Secciones Propuestas

```
1. INTRODUCTION
   - Contexto: IA en predicción de mercados (industria $35B en 2023)
   - Gap: Mayoría de papers ignoran DIFERENCIAS entre sectores
   - Novedad: Análisis comparativo Tech vs Energía + factores económicos
   - Objetivo: Explicar POR QUÉ funcionan diferente

2. LITERATURE REVIEW
   - LSTM/Transformers para predicción financiera (saturado)
   - Modelos ensemble (tendencia 2025-2026)
   - Análisis sector-específico (brecha a llenar)
   - Factores económicos que afectan predictibilidad

3. RESEARCH QUESTIONS & HYPOTHESES
   - RQ1-RQ4 (definidas arriba)
   - Hipótesis: Tech más volátil → modelos adaptativos mejor; Energía más
     predecible por ciclos macro → Transformers mejor

4. METHODOLOGY
   4.1 Datos
       - Fuentes (Yahoo Finance, FRED)
       - Período: 2021-2026 (5 años, post-COVID)
       - Preprocessing
       - Train/Test Split
   
   4.2 Características (Features)
       - OHLCV (Open, High, Low, Close, Volume)
       - Technical indicators (RSI, MACD, SMA, Bollinger Bands)
       - Macroeconomic features (de FRED)
       - Sector volatility (VIX derivatives)
   
   4.3 Modelos
       - LSTM: 2-3 capas, 100-128 units, dropout 0.2
       - Transformer: 4 heads, 2 layers, d_model=128
       - XGBoost: max_depth=5-7, learning_rate=0.1
       - Ensemble: Voting (LSTM + Transformer + XGBoost)
   
   4.4 Evaluación
       - Regresión: MAE, RMSE, R²
       - Directional Accuracy (% predicciones dirección correcta)
       - Sharpe Ratio (valor económico real)
       - Information Coefficient (IC)

5. RESULTS
   5.1 Comparativa de modelos (Tech vs Energía)
       - Tabla: Accuracy, RMSE por modelo/sector
       - Gráfico: Learning curves comparison
       - Gráfico: Predictions vs Actual (examples)
   
   5.2 Factores económicos que explican diferencias
       - Correlación features → performance por sector
       - Feature importance (SHAP values)
       - Análisis de volatilidad sector-específica
   
   5.3 Robustez cross-sector
       - Modelo entrenado en Tech → predice Energía?
       - Análisis de transferability

6. DISCUSSION
   - ¿Por qué Tech y Energía se comportan diferente?
   - Implicaciones para inversores
   - Limitaciones (datos, period, selección de acciones)
   - Investigación futura

7. CONCLUSION
   - Resumen de hallazgos
   - Contribución novedosa
   - Reproducibilidad con datasets abiertos

8. REFERENCES
   - Papers recientes (2025-2026)
   - Documentación APIs

9. SUPPLEMENTARY MATERIAL
   - Código GitHub (reproducible)
   - Hyperparameters tuning details
   - Full results tables
```

### 3.2 Figuras Principales

| Figura | Descripción | Tipo |
|--------|-------------|------|
| Fig 1 | Precios normalizados Tech vs Energía (5 años) | Línea temporal |
| Fig 2 | Comparativa accuracy por modelo y sector | Bar chart |
| Fig 3 | Predictions vs Actual (Tech example) | Línea temporal |
| Fig 4 | Predictions vs Actual (Energía example) | Línea temporal |
| Fig 5 | Feature importance SHAP (Tech vs Energía) | Beeswarm plot |
| Fig 6 | Volatilidad sector-específica en tiempo | Heatmap |
| Fig 7 | Sharpe Ratio por modelo/sector | Box plot |
| Fig 8 | Transfer learning: modelo Tech → Energía | Confusión matrix |

---

## 4. METODOLOGÍA DETALLADA

### 4.1 Preprocesamiento de Datos

```python
# Pseudocódigo workflow

1. DESCARGA
   - Yahoo Finance: 5 años diarios (2021-2026)
   - FRED: Series económicas matching
   - Sincronizar fechas

2. LIMPIEZA
   - Remover missing values (forward fill + interpolation)
   - Remover outliers (±3 std)
   - Verificar duplicados

3. NORMALIZACIÓN
   - MinMaxScaler: [0, 1] para LSTM/Transformer
   - StandardScaler: para features económicas

4. FEATURE ENGINEERING
   - Technical: RSI, MACD, SMA(20), BB
   - Económicas: Merge FRED data
   - Lags: t-1, t-5, t-10, t-20 días

5. TRAIN/TEST SPLIT
   - 70% train (2021-2024)
   - 15% validation (2024-2025)
   - 15% test (2025-2026)
   - NO shuffle (series temporal!)
```

### 4.2 Configuración de Modelos

#### LSTM
```
Input: [batch, timesteps=30, features]
├─ LSTM layer 1: 128 units, return_sequences=True, dropout=0.2
├─ LSTM layer 2: 64 units, dropout=0.2
├─ Dense: 32 units, ReLU
└─ Dense: 1 (output: precio siguiente)
Loss: MSE
Optimizer: Adam (lr=0.001)
Epochs: 50-100
```

#### Transformer
```
Input: [batch, timesteps=30, features]
├─ MultiHeadAttention: 4 heads, d_model=128
├─ FeedForward: 256 units
├─ Dropout: 0.1
├─ 2 Encoder layers (stack)
├─ Dense: 32 units, ReLU
└─ Dense: 1 (output)
Loss: MSE
Optimizer: Adam (lr=0.0005)
Epochs: 50-100
```

#### XGBoost
```
max_depth: 6
learning_rate: 0.1
n_estimators: 500
subsample: 0.8
colsample_bytree: 0.8
```

#### Ensemble
```
Voting regressor:
├─ LSTM (weight: 0.4)
├─ Transformer (weight: 0.4)
└─ XGBoost (weight: 0.2)
```

### 4.3 Métricas de Evaluación

**Regresión:**
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of determination

**Dirección (más importante economía):**
- **Directional Accuracy**: % veces que predice dirección correcta
- **Information Coefficient (IC)**: Rank correlation predicción vs actual

**Valor económico:**
- **Sharpe Ratio**: (Return - Risk-free rate) / Volatility
- **Max Drawdown**: % caída máxima desde pico
- **Win Rate**: % de trades ganadores (si se usa para trading)

---

## 5. TIMELINE & PRÓXIMOS PASOS

### Fase 1: Setup (1-2 semanas)
- [ ] Descargar datasets (Yahoo + FRED)
- [ ] EDA: visualizar Tech vs Energía
- [ ] Feature engineering initial

### Fase 2: Modelado (2-3 semanas)
- [ ] Entrenar LSTM, Transformer, XGBoost
- [ ] Hyperparameter tuning
- [ ] Ensemble model

### Fase 3: Análisis (1-2 semanas)
- [ ] Comparativas sector-específicas
- [ ] SHAP analysis (explicabilidad)
- [ ] Transfer learning test

### Fase 4: Paper (2-3 semanas)
- [ ] Redacción + figuras
- [ ] Revisión metodología
- [ ] Envío a Frontiers

**Total: 6-10 semanas** para manuscript listo

---

## 6. REPRODUCIBILIDAD (Crucial para Q1)

### Código debe incluir:

```
GitHub repo structure:
├── data/
│   ├── raw/
│   │   ├── tech_prices.csv (descargado)
│   │   ├── energy_prices.csv
│   │   └── fred_economic.csv
│   └── processed/
│       ├── tech_features.csv
│       └── energy_features.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_analysis.ipynb
├── src/
│   ├── models.py (LSTM, Transformer, XGBoost)
│   ├── utils.py (preprocessing, metrics)
│   └── config.py (hyperparameters)
├── results/
│   ├── figures/
│   └── metrics.json
├── requirements.txt
└── README.md (replicar en 5 pasos)
```

### requirements.txt
```
yfinance==0.2.38
fredapi==0.5.1
pandas==2.1.3
numpy==1.26.2
scikit-learn==1.3.2
tensorflow==2.14.0
xgboost==2.0.2
matplotlib==3.8.2
seaborn==0.13.0
shap==0.44.1
```

---

## 7. PREGUNTAS CLAVE A RESOLVER

1. **¿Período de análisis?** → 2021-2026 (5 años, post-COVID recovery)
2. **¿Tickers específicos?** → Top 2-3 por sector (AAPL, MSFT, NVDA vs CVX, COP, XLE)
3. **¿Horizonte predicción?** → 1 día (día siguiente)
4. **¿Datos intraday o daily?** → Daily (más simple, suficiente para Q1)
5. **¿Train en ambos sectores juntos o separado?** → Separado primero, luego transfer learning

---

## 8. VENTAJAS DE ESTE ENFOQUE PARA Q1

✅ **Novedoso**: Comparativa sector-específica + factores económicos  
✅ **Reproducible**: Datasets 100% abiertos + código GitHub  
✅ **Rigorous**: RQs claras, metodología sólida, métricas apropiadas  
✅ **Económicamente relevante**: Sharpe Ratio, no solo accuracy  
✅ **Generalizable**: Método aplicable a otros sectores  
✅ **Impacto**: Implicaciones prácticas para trading/inversión  

---

**Siguiente paso:** ¿Empezamos con descarga de datos y EDA?
