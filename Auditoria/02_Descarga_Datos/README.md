# 02_Descarga_Datos

## Contenido

Esta carpeta contiene todos los **datos crudos descargados** del proyecto.

### Archivos que se generarán

| Archivo | Descripción | Registros |
|---------|-------------|-----------|
| `tech_prices_raw.csv` | Precios diarios Tech (AAPL, MSFT, NVDA, GOOGL, TSLA) | ~1,250 por ticker |
| `energy_prices_raw.csv` | Precios diarios Energía (CVX, COP, XLE, MPC) | ~1,250 por ticker |
| `fred_economic_raw.csv` | Datos económicos (FRED API) | ~260 por serie |
| `download_log.txt` | Log de descarga con timestamps | - |
| `download_summary.json` | Resumen ejecutivo (JSON) | - |

---

## Cómo Descargar los Datos

### Opción A: Automático (Recomendado)

```bash
cd C:\Users\wilop\Dropbox\MPDI\2026\AI in Business\Auditoria\02_Descarga_Datos
python download_data.py
```

**Requisitos previos:**
```bash
pip install yfinance fredapi pandas
```

### Opción B: Con FRED API Key (Opcional)

Para descargar datos económicos de FRED, obtén una API key gratis:
1. Ir a: https://fred.stlouisfed.org/docs/api/api_key.html
2. Registrarse (es gratis)
3. Copiar tu API key
4. Ejecutar:

```bash
set FRED_API_KEY=tu_api_key_aqui
python download_data.py
```

O en PowerShell:
```powershell
$env:FRED_API_KEY="tu_api_key_aqui"
python download_data.py
```

---

## Datos Descargados

### Sector Tecnología (TECH)
- **AAPL** - Apple Inc.
- **MSFT** - Microsoft Corporation
- **NVDA** - NVIDIA Corporation
- **GOOGL** - Alphabet Inc.
- **TSLA** - Tesla Inc.

### Sector Energía (ENERGY)
- **CVX** - Chevron Corporation
- **COP** - ConocoPhillips
- **XLE** - Energy Select Sector SPDR ETF
- **MPC** - Marathon Petroleum

### Datos Económicos (FRED)
- **DCOILWTICO** - Crude Oil WTI Spot Price
- **DHHNGSP** - Natural Gas Spot Price
- **VIXCLS** - VIX (Volatilidad del mercado)
- **UNRATE** - Unemployment Rate
- **CPIAUCSL** - Consumer Price Index
- **FEDFUNDS** - Federal Funds Rate
- **SP500** - S&P 500 Index

---

## Período de Datos

- **Desde:** 5 años atrás (2021-08-04)
- **Hasta:** Hoy (2026-08-04)
- **Frecuencia:** Diaria

---

## ⚠️ IMPORTANTE

1. **Raw Data:** Estos datos NO han sido limpios ni procesados
2. **Inmutable:** Estos archivos son la fuente de verdad (no modificar)
3. **Backup:** Los datos descargados se guardan como copia de seguridad
4. **Siguiente paso:** Ver `03_EDA_Exploratorio/` para análisis inicial

---

## Troubleshooting

**Problema:** `ModuleNotFoundError: No module named 'yfinance'`
```bash
pip install yfinance fredapi pandas
```

**Problema:** FRED API Key error
```
Solución: Saltará FRED. Puedes obtener la key más tarde y re-ejecutar.
```

**Problema:** Sin conexión a internet
```
Solución: El script intentará descargar pero fallará. Verifica tu conexión.
```

---

## Log de Ejecución

Ver `download_log.txt` para un registro detallado de:
- Qué se descargó
- Cuándo se descargó (timestamps)
- Cuántos registros por ticker/serie
- Errores o advertencias

---

**Status:** ⏳ Pendiente de ejecución  
**Última actualización:** 2026-08-04
