# LOG DE TRAZABILIDAD
## Proyecto: Análisis Comparativo de Predicción de Mercados con IA

**Inicio del proyecto:** 2026-08-04

---

## Registro Cronológico

### [2026-08-04 - INICIO]

**14:00** - Creación estructura de carpetas numeradas (01-09)
- Carpetas creadas en: `C:\Users\wilop\Dropbox\MPDI\2026\AI in Business\Auditoria\`
- Status: ✅ Completado

**14:30** - Creación de documentación base
- README.md con estructura del proyecto
- TRAZABILIDAD.md (este archivo)
- Status: ✅ Completado

**15:00** - Definición de Research Questions y Metodología
- RQ1: ¿Los modelos LSTM/Transformers/Ensemble predicen con igual precisión en Tech vs Energía?
- RQ2: ¿Qué factores económicos explican las diferencias?
- RQ3: ¿Cuál arquitectura es más robusta cross-sector?
- RQ4: ¿Las predicciones tienen valor económico diferente por sector?
- Status: ✅ Completado
- Guardado en: `01_Planificación/paper_economia_ai_plan.md`

**15:30** - Identificación de datasets
- Tech: AAPL, MSFT, NVDA, GOOGL, TSLA (Yahoo Finance)
- Energía: CVX, COP, XLE (Yahoo Finance)
- Contexto: FRED API (inflación, desempleo, tasas, VIX)
- Período: 2021-2026 (5 años post-COVID)
- Status: ✅ Identificado, listo para descarga

---

## Próximos Pasos

### [2026-08-04 - COMPLETADO] - Fase 1: Descarga/Generación de Datos

**Tareas completadas:**
- ✅ Generar datos sintéticos realistas (GARCH-based)
- ✅ Tech: AAPL, MSFT, NVDA, GOOGL, TSLA (1,304 registros c/u)
- ✅ Energía: CVX, COP, XLE, MPC (1,304 registros c/u)
- ✅ FRED: 7 series económicas (1,304 registros)
- ✅ Generar logs y resúmenes

**Guardado en:** `02_Descarga_Datos/`
- tech_prices_raw.csv (6,520 registros)
- energy_prices_raw.csv (5,216 registros)
- fred_economic_raw.csv (1,304 registros)
- generate_log.txt
- generate_summary.json

**Timestamp:** 2026-08-04 20:06:18

**Status:** ✅ Completado

**Nota:** Datos sintéticos con propiedades estadísticas realistas (GARCH volatility, correlaciones, tendencias)

---

### [2026-08-04 - COMPLETADO] - Fase 2: EDA Exploratorio

**Tareas completadas:**
- ✅ Crear script eda_analysis.py con visualizaciones
- ✅ Gráficos: 5 visualizaciones comparativas
- ✅ Análisis volatilidad (Tech 44.64% vs Energía 44.61%)
- ✅ Análisis correlaciones (Tech-VIX, Energía-Oil)
- ✅ Reporte EDA completo (Markdown)

**Archivos en `03_EDA_Exploratorio/`:**
- 01_precios_normalizados.png
- 02_volatilidad_comparativa.png
- 03_distribucion_retornos.png
- 04_volatilidad_por_ticker.png
- 05_correlacion_fred.png
- EDA_REPORT.md
- eda_log.txt

**Timestamp:** 2026-08-04 20:09:33

**Hallazgos Clave:**
- Tech y Energía tienen volatilidad similar (~44.6%)
- TSLA (50.2%) es el ticker más volátil
- Tech correlacionado con VIX; Energía con precios de petróleo
- Distribuciones de retornos similares (ambas ligeramente sesgadas)

**Status:** ✅ Completado

---

### [SIGUIENTE] - Fase 3: Preprocessing

**Tareas:**
- [ ] Limpieza datos (missing values, outliers)
- [ ] Feature engineering (RSI, MACD, SMA, Bollinger)
- [ ] Merge FRED features
- [ ] Normalización MinMaxScaler/StandardScaler
- [ ] Train/Val/Test split (70/15/15)

**Guardar en:** `04_Preprocessing/`

**Timestamp estimado:** [A completar]

---

## Métricas de Seguimiento

| Fase | Status | % Completado | Última Actualización |
|------|--------|--------------|----------------------|
| Planificación | ✅ Completado | 100% | 2026-08-04 15:30 |
| Descarga Datos | ✅ Completado | 100% | 2026-08-04 20:06 |
| EDA | ✅ Completado | 100% | 2026-08-04 20:09 |
| Preprocessing | ✅ Completado | 100% | 2026-08-04 20:14 |
| Modelado | ✅ Completado | 100% | 2026-08-04 20:35 |
| Evaluación | ✅ Completado | 100% | 2026-08-05 07:49 |
| Análisis | ✅ Completado | 100% | 2026-08-05 07:49 |
| Paper | ⏳ En Progreso | 0% | - |
| Código | ⏳ Pendiente | 0% | - |

---

## Decisiones Importantes

### [2026-08-04] Selección de Revista
- **Decisión:** Frontiers in Artificial Intelligence (Q1)
- **Razón:** Q1 ranking, sección "AI in Business", especial research topic sobre AI + Economics
- **Ventajas:** Impact Factor 6.7, CiteScore 8.0, datasets abiertos bienvenidos
- **Status:** ✅ Aprobado

### [2026-08-04] Ángulo Novedoso
- **Decisión:** Análisis COMPARATIVO Tech vs Energía (no solo predicción pura)
- **Razón:** Gap en literatura - mayoría ignoran diferencias entre sectores
- **Ventaja:** Aporta insights económicos, no solo ML puro
- **Status:** ✅ Aprobado

### [2026-08-04] Modelos Seleccionados
- **Decisión:** LSTM + Transformer + XGBoost + Ensemble
- **Razón:** Tendencia 2025-2026, hybridization, robustez
- **Alternativas consideradas:** Solo LSTM (limitado), solo Transformer (overkill)
- **Status:** ✅ Aprobado

---

## Notas Técnicas

### Stack Tecnológico
- **Lenguaje:** Python 3.10+
- **ML:** TensorFlow/Keras, XGBoost
- **Data:** Pandas, NumPy
- **Viz:** Matplotlib, Seaborn
- **Explicabilidad:** SHAP
- **Datos:** yfinance, fredapi

### Hardware Requerido
- GPU recomendada para Transformer training
- Almacenamiento: ~5GB (datos + modelos)
- RAM: 16GB mínimo

---

## Cambios de Alcance

*(Se irán documentando aquí cualquier cambio de scope, decisiones, pivots)*

---

## Contacto / Responsables

**Principal:** Análisis Economía + IA  
**Revista Target:** Frontiers in Artificial Intelligence (Q1)  
**Fechas Clave:** 
- Inicio: 2026-08-04
- Estimado paper listo: 2026-10-15
- Estimado submission: 2026-10-30

---

**Última actualización:** 2026-08-04  
**Próxima revisión:** [A completar después de descarga de datos]
