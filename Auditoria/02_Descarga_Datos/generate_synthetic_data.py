"""
Script de Generación de Datos Sintéticos Realistas
Proyecto: Análisis Comparativo de Predicción de Mercados con IA

Genera datos que simulan patrones históricos reales de:
- Acciones Tech (AAPL, MSFT, NVDA, GOOGL, TSLA)
- Acciones Energía (CVX, COP, XLE, MPC)
- Factores económicos (FRED series)

Nota: Estos datos sintéticos mantienen propiedades estadísticas realistas
incluyendo volatilidad sector-específica, correlaciones, tendencias.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(OUTPUT_DIR, "generate_log.txt")
BASE_SEED = 42  # Semilla base; cada serie usa BASE_SEED + offset para ser independiente y reproducible

# Período
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=5*365)  # 5 años
DATES = pd.date_range(start=START_DATE, end=END_DATE, freq='B')  # Business days

# Características por sector
TECH_PARAMS = {
    "tickers": {
        "AAPL": {"base_price": 150, "volatility": 0.022, "trend": 0.0003},
        "MSFT": {"base_price": 320, "volatility": 0.018, "trend": 0.0004},
        "NVDA": {"base_price": 450, "volatility": 0.035, "trend": 0.0005},
        "GOOGL": {"base_price": 140, "volatility": 0.019, "trend": 0.0003},
        "TSLA": {"base_price": 200, "volatility": 0.040, "trend": 0.0002},
    },
    "base_volatility": 0.025,
    "description": "Sector Tecnología"
}

ENERGY_PARAMS = {
    "tickers": {
        "CVX": {"base_price": 145, "volatility": 0.028, "trend": 0.0001},
        "COP": {"base_price": 110, "volatility": 0.032, "trend": 0.00015},
        "XLE": {"base_price": 85, "volatility": 0.025, "trend": 0.0},
        "MPC": {"base_price": 120, "volatility": 0.030, "trend": 0.0002},
    },
    "base_volatility": 0.028,
    "description": "Sector Energía"
}

FRED_PARAMS = {
    "DCOILWTICO": {"base": 90, "volatility": 0.035, "trend": 0.00001},
    "DHHNGSP": {"base": 3.5, "volatility": 0.040, "trend": 0.00002},
    "VIXCLS": {"base": 18, "volatility": 0.08, "trend": -0.0001},
    "UNRATE": {"base": 3.8, "volatility": 0.0002, "trend": 0.00001},
    "CPIAUCSL": {"base": 310, "volatility": 0.003, "trend": 0.0004},
    "FEDFUNDS": {"base": 4.5, "volatility": 0.015, "trend": 0.00001},
    "SP500": {"base": 5200, "volatility": 0.015, "trend": 0.0003},
}

# ============================================================================
# FUNCIONES
# ============================================================================

def log_message(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")

def generate_garch_series(dates, base_price, volatility, trend, mean_reversion=0.05, rng=None,
                           factor_shock=None, factor_beta=0.0):
    """
    Genera serie de precios con volatilidad realista (GARCH-like)
    Parámetros:
    - base_price: precio inicial
    - volatility: volatilidad diaria promedio
    - trend: tendencia diaria
    - mean_reversion: velocidad de reversión a la media
    - rng: generador de números aleatorios independiente por serie (np.random.Generator)
    - factor_shock: serie externa ya estandarizada (media 0, var 1) de un factor común
      (p.ej. shock diario del VIX o del petróleo) que se mezcla en el shock propio del ticker.
      Si es None, el shock es puramente idiosincrático (comportamiento original).
    - factor_beta: peso del factor común en el shock combinado, en [-1, 1]. El shock final es
      beta*factor_shock + sqrt(1-beta^2)*shock_idiosincrático, preservando varianza unitaria,
      de modo que corr(retorno, factor) ≈ beta y la volatilidad total del ticker no cambia.
    """
    n = len(dates)
    returns = np.zeros(n)
    prices = np.zeros(n)
    prices[0] = base_price

    # Volatilidad estocástica
    vol_series = np.zeros(n)
    vol_series[0] = volatility

    if rng is None:
        rng = np.random.default_rng()

    idio_weight = np.sqrt(max(0.0, 1.0 - factor_beta**2))

    for t in range(1, n):
        # GARCH: volatilidad depende de retorno previo y volatilidad previa
        idio_shock = rng.normal(0, 1)
        if factor_shock is not None:
            shock = factor_beta * factor_shock[t] + idio_weight * idio_shock
        else:
            shock = idio_shock
        vol_series[t] = np.sqrt(
            0.00001 + 0.05 * (returns[t-1]**2) + 0.94 * (vol_series[t-1]**2)
        )

        # Retorno con tendencia y volatilidad estocástica
        # Clip a +/-20% diario: el GARCH(1,1) con alpha+beta=0.99 es casi-integrado y puede
        # retroalimentarse hacia una trayectoria explosiva para ciertas semillas; 20% ya excede
        # el límite de circuit-breaker de una acción individual en un día real.
        raw_return = trend + vol_series[t] * shock
        returns[t] = np.clip(raw_return, -0.20, 0.20)
        prices[t] = prices[t-1] * (1 + returns[t])

    return prices

def generate_price_data(sector_params, sector_name, seed_offset, factor_shock=None, factor_beta=0.0):
    """Genera datos de precios para un sector"""
    log_message(f"Generando datos {sector_name}...")

    all_data = []

    for i, (ticker, params) in enumerate(sector_params["tickers"].items()):
        rng = np.random.default_rng(BASE_SEED + seed_offset + i)
        prices = generate_garch_series(
            DATES,
            base_price=params["base_price"],
            volatility=params["volatility"],
            trend=params["trend"],
            rng=rng,
            factor_shock=factor_shock,
            factor_beta=factor_beta
        )

        # Calcular OHLC sintético
        data = pd.DataFrame({
            "Date": DATES,
            "Ticker": ticker,
            "Close": prices,
            "Open": prices * rng.uniform(0.99, 1.01, len(prices)),
            "High": prices * rng.uniform(1.00, 1.02, len(prices)),
            "Low": prices * rng.uniform(0.98, 1.00, len(prices)),
            "Volume": rng.integers(50_000_000, 200_000_000, len(prices)),
        })

        all_data.append(data)
        log_message(f"  ✓ {ticker}: {len(data)} registros")

    combined = pd.concat(all_data, ignore_index=True)
    return combined

def generate_fred_data(seed_offset):
    """Genera datos económicos FRED sintéticos"""
    log_message("Generando datos económicos FRED...")

    data = pd.DataFrame({"Date": DATES})

    for i, (series_id, params) in enumerate(FRED_PARAMS.items()):
        rng = np.random.default_rng(BASE_SEED + seed_offset + i)
        values = generate_garch_series(
            DATES,
            base_price=params["base"],
            volatility=params["volatility"],
            trend=params["trend"],
            rng=rng
        )
        # Asegurar valores positivos
        values = np.maximum(values, params["base"] * 0.5)
        data[series_id] = values
        log_message(f"  ✓ {series_id}")

    return data

def save_data(data, filename):
    """Guarda datos en CSV"""
    filepath = os.path.join(OUTPUT_DIR, filename)
    data.to_csv(filepath, index=False)
    log_message(f"✓ Guardado: {filename} ({len(data)} registros)")
    return filepath

def create_summary(tech_data, energy_data, fred_data):
    """Crea resumen de generación"""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "type": "SYNTHETIC DATA (Generated for research)",
        "period": {
            "start": START_DATE.strftime("%Y-%m-%d"),
            "end": END_DATE.strftime("%Y-%m-%d"),
            "days": (END_DATE - START_DATE).days,
            "business_days": len(DATES)
        },
        "yahoo_finance": {
            "tech": {
                "tickers": list(TECH_PARAMS["tickers"].keys()),
                "total_records": len(tech_data),
                "volatility": TECH_PARAMS["base_volatility"]
            },
            "energy": {
                "tickers": list(ENERGY_PARAMS["tickers"].keys()),
                "total_records": len(energy_data),
                "volatility": ENERGY_PARAMS["base_volatility"]
            }
        },
        "fred": {
            "series": list(FRED_PARAMS.keys()),
            "total_records": len(fred_data)
        },
        "notes": [
            "Datos generados sintéticamente con propiedades estadísticas realistas",
            "Utilizan modelos GARCH para volatilidad estocástica",
            "Tech sector: volatilidad más alta (2.5%) que Energía (2.8%)",
            "Mantienen correlaciones y tendencias realistas"
        ]
    }

    filepath = os.path.join(OUTPUT_DIR, "generate_summary.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log_message(f"✓ Resumen guardado: generate_summary.json")
    return summary

# ============================================================================
# MAIN
# ============================================================================

def main():
    log_message("="*70)
    log_message("GENERACIÓN DE DATOS SINTÉTICOS REALISTAS")
    log_message(f"Fecha/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"Período: {START_DATE.strftime('%Y-%m-%d')} a {END_DATE.strftime('%Y-%m-%d')}")
    log_message("="*70)

    # 1. Generar datos FRED primero (offsets 200-206): Tech y Energía necesitan el
    #    shock diario de VIX/petróleo ya generado para poder exponerse a él.
    fred_data = generate_fred_data(seed_offset=200)
    fred_file = save_data(fred_data, "fred_economic_raw.csv")

    # Shocks estandarizados (media 0, var 1) de VIX y petróleo, para inyectar como
    # factor de riesgo común en los retornos de Tech y Energía respectivamente.
    vix_pct_change = fred_data["VIXCLS"].pct_change().fillna(0.0).values
    vix_shock = (vix_pct_change - vix_pct_change.mean()) / vix_pct_change.std()

    oil_pct_change = fred_data["DCOILWTICO"].pct_change().fillna(0.0).values
    oil_shock = (oil_pct_change - oil_pct_change.mean()) / oil_pct_change.std()

    # 2. Generar datos Tech (offsets 0-4): retorno negativamente expuesto al VIX
    #    (beta=-0.55 -> cuando el VIX sube, tech tiende a caer; sentimiento de riesgo).
    tech_data = generate_price_data(
        TECH_PARAMS, "SECTOR TECNOLOGÍA", seed_offset=0,
        factor_shock=vix_shock, factor_beta=-0.55
    )
    tech_file = save_data(tech_data, "tech_prices_raw.csv")

    # 3. Generar datos Energía (offsets 100-103): retorno positivamente expuesto al
    #    precio del petróleo (beta=+0.65 -> acoplamiento con el commodity subyacente).
    energy_data = generate_price_data(
        ENERGY_PARAMS, "SECTOR ENERGÍA", seed_offset=100,
        factor_shock=oil_shock, factor_beta=0.65
    )
    energy_file = save_data(energy_data, "energy_prices_raw.csv")

    # 4. Crear resumen
    log_message("\n" + "="*70)
    create_summary(tech_data, energy_data, fred_data)

    log_message("="*70)
    log_message("✓ GENERACIÓN COMPLETADA")
    log_message("="*70)
    log_message("\nArchivos generados:")
    log_message(f"  1. tech_prices_raw.csv ({len(tech_data)} registros, 5 tickers)")
    log_message(f"  2. energy_prices_raw.csv ({len(energy_data)} registros, 4 tickers)")
    log_message(f"  3. fred_economic_raw.csv ({len(fred_data)} registros, 7 series)")
    log_message(f"  4. generate_log.txt (este log)")
    log_message(f"  5. generate_summary.json (resumen ejecutivo)")
    log_message("\n⚠ NOTA: Estos datos son SINTÉTICOS con propiedades estadísticas realistas")
    log_message("   Se generaron para investigación/desarrollo de metodología")
    log_message("\nPróximo paso: EDA en 03_EDA_Exploratorio/")

if __name__ == "__main__":
    main()
