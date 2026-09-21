"""
Script de Descarga de Datos
Proyecto: Análisis Comparativo de Predicción de Mercados con IA
Descarga: Yahoo Finance (precios) + FRED (datos económicos)
"""

import pandas as pd
import yfinance as yf
import fredapi
from datetime import datetime, timedelta
import os
import json

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Períodos
END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=5*365)  # 5 años

# Tickers
TECH_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"]
ENERGY_TICKERS = ["CVX", "COP", "XLE", "MPC"]  # XLE es Energy ETF

# FRED Series IDs
FRED_SERIES = {
    "DCOILWTICO": "Crude Oil WTI Spot Price",
    "DHHNGSP": "Natural Gas Spot Price",
    "VIXCLS": "VIX Index",
    "UNRATE": "Unemployment Rate",
    "CPIAUCSL": "Consumer Price Index",
    "FEDFUNDS": "Federal Funds Rate",
    "SP500": "S&P 500",
}

# Rutas
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(OUTPUT_DIR, "download_log.txt")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "download_summary.json")

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def log_message(msg):
    """Log a mensaje a consola y archivo"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted_msg + "\n")

def download_yahoo_data(tickers, start, end, sector_name):
    """Descarga datos de Yahoo Finance"""
    log_message(f"\n=== Descargando {sector_name} de Yahoo Finance ===")

    all_data = {}

    for ticker in tickers:
        try:
            log_message(f"  Descargando {ticker}...", )
            data = yf.download(ticker, start=start, end=end, progress=False)
            all_data[ticker] = data
            log_message(f"  ✓ {ticker}: {len(data)} registros")
        except Exception as e:
            log_message(f"  ✗ Error descargando {ticker}: {str(e)}")

    return all_data

def save_yahoo_data(data_dict, filename):
    """Guarda datos de Yahoo Finance en CSV consolidado"""
    combined_data = pd.DataFrame()

    for ticker, data in data_dict.items():
        data['Ticker'] = ticker
        combined_data = pd.concat([combined_data, data], ignore_index=False)

    combined_data.reset_index(inplace=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    combined_data.to_csv(filepath, index=False)
    log_message(f"✓ Guardado: {filename} ({len(combined_data)} registros)")
    return filepath

def download_fred_data(fred_api_key, series_dict, start, end):
    """Descarga datos de FRED"""
    log_message(f"\n=== Descargando datos económicos de FRED ===")
    log_message(f"API Key: {'***' + fred_api_key[-4:] if fred_api_key else 'NO CONFIGURADA'}")

    if not fred_api_key:
        log_message("⚠ FRED API Key no encontrada. Saltando FRED.")
        log_message("  Para descargar, obtén key gratis en: https://fred.stlouisfed.org/docs/api/api_key.html")
        return None

    fred = fredapi.Fred(api_key=fred_api_key)
    fred_data = pd.DataFrame()

    for series_id, description in series_dict.items():
        try:
            log_message(f"  Descargando {series_id} ({description})...")
            data = fred.get_series(series_id, observations_start_date=start, observations_end_date=end)
            fred_data[series_id] = data
            log_message(f"  ✓ {series_id}: {len(data)} registros")
        except Exception as e:
            log_message(f"  ✗ Error descargando {series_id}: {str(e)}")

    return fred_data

def save_fred_data(data, filename):
    """Guarda datos de FRED en CSV"""
    if data is None or data.empty:
        log_message("⚠ No hay datos FRED para guardar")
        return None

    filepath = os.path.join(OUTPUT_DIR, filename)
    data.to_csv(filepath)
    log_message(f"✓ Guardado: {filename} ({len(data)} registros)")
    return filepath

def create_summary(tech_data, energy_data, fred_data):
    """Crea resumen de descarga"""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "period": {
            "start": START_DATE.strftime("%Y-%m-%d"),
            "end": END_DATE.strftime("%Y-%m-%d"),
            "days": (END_DATE - START_DATE).days
        },
        "yahoo_finance": {
            "tech": {
                "tickers": TECH_TICKERS,
                "total_records": sum(len(df) for df in tech_data.values()) if tech_data else 0
            },
            "energy": {
                "tickers": ENERGY_TICKERS,
                "total_records": sum(len(df) for df in energy_data.values()) if energy_data else 0
            }
        },
        "fred": {
            "series_downloaded": list(FRED_SERIES.keys()),
            "total_records": len(fred_data) if fred_data is not None else 0
        },
        "files_created": [
            "tech_prices_raw.csv",
            "energy_prices_raw.csv",
            "fred_economic_raw.csv (si FRED disponible)"
        ]
    }

    filepath = SUMMARY_FILE
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log_message(f"\n✓ Resumen guardado: download_summary.json")
    return summary

# ============================================================================
# MAIN
# ============================================================================

def main():
    log_message("="*70)
    log_message("INICIO DE DESCARGA DE DATOS")
    log_message(f"Fecha/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"Período: {START_DATE.strftime('%Y-%m-%d')} a {END_DATE.strftime('%Y-%m-%d')}")
    log_message("="*70)

    # 1. Descargar Yahoo Finance - Tech
    tech_data = download_yahoo_data(TECH_TICKERS, START_DATE, END_DATE, "SECTOR TECNOLOGÍA")
    tech_file = save_yahoo_data(tech_data, "tech_prices_raw.csv")

    # 2. Descargar Yahoo Finance - Energía
    energy_data = download_yahoo_data(ENERGY_TICKERS, START_DATE, END_DATE, "SECTOR ENERGÍA")
    energy_file = save_yahoo_data(energy_data, "energy_prices_raw.csv")

    # 3. Descargar FRED
    fred_api_key = os.environ.get("FRED_API_KEY")
    fred_data = download_fred_data(fred_api_key, FRED_SERIES, START_DATE, END_DATE)
    fred_file = save_fred_data(fred_data, "fred_economic_raw.csv")

    # 4. Crear resumen
    log_message("\n" + "="*70)
    create_summary(tech_data, energy_data, fred_data)

    log_message("="*70)
    log_message("✓ DESCARGA COMPLETADA")
    log_message("="*70)
    log_message("\nArchivos generados:")
    log_message(f"  1. tech_prices_raw.csv ({len(tech_data)} tickers)")
    log_message(f"  2. energy_prices_raw.csv ({len(energy_data)} tickers)")
    if fred_file:
        log_message(f"  3. fred_economic_raw.csv ({len(fred_data)} series)")
    log_message(f"  4. download_log.txt (este log)")
    log_message(f"  5. download_summary.json (resumen ejecutivo)")
    log_message("\nPróximo paso: EDA en 03_EDA_Exploratorio/")

if __name__ == "__main__":
    main()
