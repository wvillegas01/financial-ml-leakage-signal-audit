# Análisis Exploratorio de Datos (EDA)
## Proyecto: Predicción de Mercados con IA

**Fecha:** 2026-08-05 18:50:14

---

## 1. RESUMEN EJECUTIVO

### Datos Descargados
- **Período:** 2021-08-05 a 2026-08-04 (5 años)
- **Sector Tech:** 5 tickers × 1,304 registros = 6,520 observaciones
- **Sector Energía:** 4 tickers × 1,304 registros = 5,216 observaciones
- **Factores Económicos:** 7 series FRED × 1,304 observaciones

### Hallazgos Clave

#### Volatilidad Comparativa
| Métrica | Tech | Energía |
|---------|------|---------|
| **Volatilidad Promedio (Anualizada)** | 0.5625 | 0.5852 |
| **Volatilidad Máxima** | 0.6256 (MSFT) | 0.6963 (CVX) |
| **Volatilidad Mínima** | 0.4993 (GOOGL) | 0.4818 (MPC) |

**Conclusión:** Tech es **-3.9% MÁS VOLÁTIL** que Energía.

---

## 2. ESTADÍSTICAS DESCRIPTIVAS

### Sector Tecnología
```
        count    mean     std     min      max
Ticker                                        
AAPL     1304   78.76   17.33   34.37   156.79
GOOGL    1304  122.15   25.16   62.67   204.42
MSFT     1304  186.24   56.85   79.45   645.40
NVDA     1304  900.86  421.56  251.65  1895.06
TSLA     1304  117.62   40.32   66.72   331.96
```

### Sector Energía
```
        count    mean    std    min     max
Ticker                                     
COP      1304  125.69  31.26  70.60  266.54
CVX      1304   39.50  51.10   7.42  239.37
MPC      1304   95.40  32.53  42.94  236.22
XLE      1304   33.40  21.06  10.40  105.08
```

---

## 3. VOLATILIDAD POR TICKER

### Tech - Volatilidad Anualizada
```
Ticker
MSFT     0.625645
TSLA     0.598939
AAPL     0.576841
NVDA     0.512057
GOOGL    0.499253
```

### Energía - Volatilidad Anualizada
```
Ticker
CVX    0.696286
XLE    0.608942
COP    0.553722
MPC    0.481778
```

**Ranking de Mayor a Menor Volatilidad:**
1. TSLA (Tech): 0.5989
2. NVDA (Tech): 0.5121
3. MPC (Energía): 0.4818
4. COP (Energía): 0.5537

---

## 4. IMPLICACIONES PARA MODELADO

### Observación: Tech más volátil
- **Predictibilidad esperada:** Menor en Tech, mayor en Energía
- **Modelos recomendados:** Transformers (capturan volatilidad) para Tech; LSTM para Energía
- **Métricas críticas:** Sharpe Ratio (no solo accuracy)

### Factores Económicos Relevantes
- **Tech:** VIX (volatilidad del mercado) — correlación esperada positiva
- **Energía:** Precios de petróleo (WTI) — correlación muy alta esperada

---

## 5. VISUALIZACIONES GENERADAS

| # | Archivo | Descripción |
|---|---------|-------------|
| 1 | `01_precios_normalizados.png` | Precios base 100 - Tech vs Energía |
| 2 | `02_volatilidad_comparativa.png` | Volatilidad móvil 30-día |
| 3 | `03_distribucion_retornos.png` | Distribución de retornos diarios |
| 4 | `04_volatilidad_por_ticker.png` | Volatilidad individual por ticker |
| 5 | `05_correlacion_fred.png` | Correlación con factores económicos |

---

## 6. PRÓXIMOS PASOS

1. **Preprocessing (04):** Feature engineering, normalización
2. **Modelado (05):** LSTM, Transformer, XGBoost
3. **Evaluación (06):** Métricas por sector
4. **Análisis (07):** SHAP, explicabilidad

---

**Status:** ✅ EDA Completado
**Duración:** ~18:50:14
