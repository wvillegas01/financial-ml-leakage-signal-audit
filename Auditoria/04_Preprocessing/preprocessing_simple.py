"""
Script de Preprocessing (Versión Simplificada)
Proyecto: Análisis Comparativo de Predicción de Mercados con IA

Realiza:
- Limpieza de datos
- Feature engineering básico
- Normalización
- Train/Val/Test split
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), "02_Descarga_Datos")
LOG_FILE = os.path.join(OUTPUT_DIR, "preprocessing_log.txt")

# ============================================================================
# FUNCIONES
# ============================================================================

def log_msg(msg):
    """Log a consola y archivo"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")

def load_data():
    """Carga datos raw"""
    log_msg("Cargando datos raw...")
    tech = pd.read_csv(os.path.join(DATA_DIR, "tech_prices_raw.csv"))
    energy = pd.read_csv(os.path.join(DATA_DIR, "energy_prices_raw.csv"))
    fred = pd.read_csv(os.path.join(DATA_DIR, "fred_economic_raw.csv"))

    for df in [tech, energy, fred]:
        df['Date'] = pd.to_datetime(df['Date'])

    log_msg(f"✓ Tech: {len(tech)} registros")
    log_msg(f"✓ Energy: {len(energy)} registros")
    log_msg(f"✓ FRED: {len(fred)} registros")

    return tech, energy, fred

def clean_and_process(data, sector_name):
    """Limpia y procesa datos por sector"""
    log_msg(f"\nProcesando {sector_name}...")

    data = data.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    # Calcular retornos diarios
    data['Daily_Return'] = data.groupby('Ticker')['Close'].pct_change()
    data['Log_Return'] = data.groupby('Ticker')['Close'].transform(
        lambda x: np.log(x / x.shift(1))
    )

    # Indicadores técnicos simples
    data['SMA_20'] = data.groupby('Ticker')['Close'].transform(
        lambda x: x.rolling(window=20).mean()
    )
    data['SMA_50'] = data.groupby('Ticker')['Close'].transform(
        lambda x: x.rolling(window=50).mean()
    )
    data['Volatility_20'] = data.groupby('Ticker')['Daily_Return'].transform(
        lambda x: x.rolling(window=20).std() * np.sqrt(252)
    )

    # Lags
    for lag in [1, 5, 10]:
        data[f'Close_Lag_{lag}'] = data.groupby('Ticker')['Close'].shift(lag)
        data[f'Return_Lag_{lag}'] = data.groupby('Ticker')['Daily_Return'].shift(lag)

    log_msg(f"✓ Features calculados para {len(data.groupby('Ticker'))} tickers")

    return data

def merge_fred(data, fred):
    """Merge con FRED"""
    log_msg("Mergeando con FRED...")

    data = data.merge(fred, on='Date', how='left')

    # Fill missing
    fred_cols = ['DCOILWTICO', 'DHHNGSP', 'VIXCLS', 'UNRATE', 'CPIAUCSL', 'FEDFUNDS', 'SP500']
    for col in fred_cols:
        if col in data.columns:
            data[col] = data[col].fillna(method='ffill')

    log_msg(f"✓ FRED merged")

    return data

def normalize(data):
    """Normaliza features"""
    log_msg("Normalizando features...")

    # MinMax para precios
    scaler_mm = MinMaxScaler()
    price_cols = ['Close', 'Open', 'High', 'Low']

    for col in price_cols:
        valid = ~data[col].isna()
        if valid.sum() > 0:
            data.loc[valid, f'{col}_Norm'] = scaler_mm.fit_transform(
                data.loc[valid, [col]]
            )

    # Standard para económicos
    scaler_st = StandardScaler()
    fred_cols = ['DCOILWTICO', 'DHHNGSP', 'VIXCLS']

    for col in fred_cols:
        if col in data.columns:
            valid = ~data[col].isna()
            if valid.sum() > 0:
                data.loc[valid, f'{col}_Std'] = scaler_st.fit_transform(
                    data.loc[valid, [[col]]]
                ).flatten()

    log_msg(f"✓ Normalización completada")

    return data

def train_val_test_split(data):
    """Divide train/val/test 70/15/15"""
    log_msg("\nTrain/Val/Test split...")

    dates = sorted(data['Date'].unique())
    n = len(dates)

    train_cutoff = dates[int(n * 0.7)]
    val_cutoff = dates[int(n * 0.85)]

    train = data[data['Date'] <= train_cutoff].copy()
    val = data[(data['Date'] > train_cutoff) & (data['Date'] <= val_cutoff)].copy()
    test = data[data['Date'] > val_cutoff].copy()

    log_msg(f"✓ Train: {len(train)} ({len(train)/len(data)*100:.1f}%)")
    log_msg(f"✓ Val: {len(val)} ({len(val)/len(data)*100:.1f}%)")
    log_msg(f"✓ Test: {len(test)} ({len(test)/len(data)*100:.1f}%)")

    return train, val, test

def save_data(tech, energy, train, val, test):
    """Guarda datos procesados"""
    log_msg("\nGuardando datos procesados...")

    tech.to_csv(os.path.join(OUTPUT_DIR, "tech_features.csv"), index=False)
    energy.to_csv(os.path.join(OUTPUT_DIR, "energy_features.csv"), index=False)
    train.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    test.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    log_msg(f"✓ tech_features.csv")
    log_msg(f"✓ energy_features.csv")
    log_msg(f"✓ train.csv")
    log_msg(f"✓ val.csv")
    log_msg(f"✓ test.csv")

# ============================================================================
# MAIN
# ============================================================================

def main():
    log_msg("="*70)
    log_msg("PREPROCESSING - FEATURE ENGINEERING & NORMALIZATION")
    log_msg(f"Fecha/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_msg("="*70)

    # Cargar
    tech, energy, fred = load_data()

    # Procesar
    tech = clean_and_process(tech, "TECH")
    energy = clean_and_process(energy, "ENERGY")

    # Merge FRED
    tech = merge_fred(tech, fred)
    energy = merge_fred(energy, fred)

    # Normalizar
    tech = normalize(tech)
    energy = normalize(energy)

    # Split
    combined = pd.concat([tech, energy], ignore_index=True)
    train, val, test = train_val_test_split(combined)

    # Guardar
    save_data(tech, energy, train, val, test)

    log_msg("\n" + "="*70)
    log_msg("✓ PREPROCESSING COMPLETADO")
    log_msg("="*70)
    log_msg("\nArchivos generados:")
    log_msg("  1. tech_features.csv")
    log_msg("  2. energy_features.csv")
    log_msg("  3. train.csv (70%)")
    log_msg("  4. val.csv (15%)")
    log_msg("  5. test.csv (15%)")
    log_msg("\nPróximo paso: Modelado en 05_Modelado/")

if __name__ == "__main__":
    main()
