# 03_EDA_Exploratorio

## Contenido

Análisis exploratorio completo comparando **Sector Tecnología vs Sector Energía**.

### Archivos Generados

| Archivo | Descripción |
|---------|-------------|
| `01_precios_normalizados.png` | Precios base 100 - evolución Tech vs Energía |
| `02_volatilidad_comparativa.png` | Volatilidad móvil (30-día) por sector |
| `03_distribucion_retornos.png` | Distribución de retornos diarios |
| `04_volatilidad_por_ticker.png` | Volatilidad anualizada individual |
| `05_correlacion_fred.png` | Correlación con factores económicos (VIX, Oil) |
| `EDA_REPORT.md` | Reporte completo en Markdown |
| `eda_log.txt` | Log de ejecución |

---

## 🔍 Hallazgos Principales

### Volatilidad Comparativa
- **Tech volatilidad promedio:** 44.64% anualizada
- **Energía volatilidad promedio:** 44.61% anualizada
- **Diferencia:** Prácticamente igual (~0.1%)

### Tickers Más Volátiles
1. **TSLA** (Tech): 50.2% anualizada ← MÁS VOLÁTIL
2. **NVDA** (Tech): 48.1% anualizada
3. **MPC** (Energía): 45.3% anualizada
4. **COP** (Energía): 44.8% anualizada

### Implicaciones
- **Tech:** Mayor concentración de volatilidad en 2 tickers (TSLA, NVDA)
- **Energía:** Volatilidad más distribuida entre tickers
- **Modelos:** Tech puede requerir arquitecturas más complejas (Transformers)
- **Predictibilidad:** Energía puede ser más predecible (correlación con petróleo)

---

## 📊 Visualizaciones Clave

### Fig 1: Precios Normalizados
Muestra la evolución relativa de precios desde 2021 en base 100.

### Fig 2: Volatilidad Comparativa
Volatilidad móvil de 30 días - Tech y Energía como agregados.

### Fig 3: Distribución de Retornos
Histogramas de retornos diarios mostrando distribuciones similares (ambas ligeramente sesgadas).

### Fig 4: Volatilidad por Ticker
Ranking de volatilidad individual - muestra TSLA como outlier.

### Fig 5: Correlación con FRED
- Tech correlacionado con VIX (volatilidad del mercado)
- Energía altamente correlacionada con precio de petróleo WTI

---

## 📈 Datos Utilizados

| Métrica | Valor |
|---------|-------|
| Período | 2021-08-05 a 2026-08-04 (5 años) |
| Observaciones Tech | 6,520 (1,304 × 5 tickers) |
| Observaciones Energía | 5,216 (1,304 × 4 tickers) |
| Observaciones FRED | 1,304 (7 series económicas) |

---

## 🚀 Próximos Pasos

→ **04_Preprocessing:** Feature engineering, normalización, creación de features técnicos

---

**Status:** ✅ Completado  
**Timestamp:** 2026-08-04 20:09:33  
**Duración:** ~4 segundos
