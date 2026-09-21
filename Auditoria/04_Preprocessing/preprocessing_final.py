"""
Preprocessing Final - Versión Estable
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), "02_Descarga_Datos")
LOG_FILE = os.path.join(OUTPUT_DIR, "preprocessing_log.txt")

def log_msg(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def main():
    log_msg("="*70)
    log_msg("PREPROCESSING - FEATURE ENGINEERING")
    log_msg(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_msg("="*70)

    # 1. CARGAR
    log_msg("\n1. CARGANDO DATOS...")
    tech = pd.read_csv(os.path.join(DATA_DIR, "tech_prices_raw.csv"))
    energy = pd.read_csv(os.path.join(DATA_DIR, "energy_prices_raw.csv"))
    fred = pd.read_csv(os.path.join(DATA_DIR, "fred_economic_raw.csv"))

    for df in [tech, energy, fred]:
        df['Date'] = pd.to_datetime(df['Date'])

    log_msg(f"✓ Tech: {len(tech)} registros, 5 tickers")
    log_msg(f"✓ Energy: {len(energy)} registros, 4 tickers")
    log_msg(f"✓ FRED: {len(fred)} registros, 7 series")

    # 2. PROCESAR CADA SECTOR
    log_msg("\n2. CALCULANDO FEATURES...")

    for df, name in [(tech, "Tech"), (energy, "Energy")]:
        df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

        # Retornos
        df['Return'] = df.groupby('Ticker')['Close'].pct_change()

        # Moving averages
        df['SMA_20'] = df.groupby('Ticker')['Close'].transform(
            lambda x: x.rolling(20).mean()
        )
        df['SMA_50'] = df.groupby('Ticker')['Close'].transform(
            lambda x: x.rolling(50).mean()
        )

        # Volatilidad
        df['Vol_20'] = df.groupby('Ticker')['Return'].transform(
            lambda x: x.rolling(20).std() * np.sqrt(252)
        )

        # Lags
        df['Close_Lag1'] = df.groupby('Ticker')['Close'].shift(1)
        df['Close_Lag5'] = df.groupby('Ticker')['Close'].shift(5)
        df['Return_Lag1'] = df.groupby('Ticker')['Return'].shift(1)
        df['Return_Lag5'] = df.groupby('Ticker')['Return'].shift(5)

        # Intradía rezagado 1 día (información ya cerrada, sin fuga)
        df['Open_Lag1'] = df.groupby('Ticker')['Open'].shift(1)
        df['High_Lag1'] = df.groupby('Ticker')['High'].shift(1)
        df['Low_Lag1'] = df.groupby('Ticker')['Low'].shift(1)
        df['Volume_Lag1'] = df.groupby('Ticker')['Volume'].shift(1)

        # Target: retorno logarítmico del día SIGUIENTE (r_{t+1} = ln(Close_{t+1}/Close_t)).
        # Se predice con información disponible al cierre del día t; no incluye Close_t
        # como feature+target simultáneo (sin fuga de información).
        df['Target_Return'] = df.groupby('Ticker')['Close'].transform(
            lambda x: np.log(x.shift(-1) / x)
        )

        if name == "Tech":
            tech = df
        else:
            energy = df

    log_msg(f"✓ Tech features: {len(tech)} registros")
    log_msg(f"✓ Energy features: {len(energy)} registros")

    # 3. MERGE FRED (las 7 series completas, no solo VIX/petróleo)
    log_msg("\n3. MERGING FRED DATA...")
    fred_cols = ['DCOILWTICO', 'VIXCLS', 'DHHNGSP', 'UNRATE', 'CPIAUCSL', 'FEDFUNDS', 'SP500']
    tech = tech.merge(fred[['Date'] + fred_cols], on='Date', how='left')
    energy = energy.merge(fred[['Date'] + fred_cols], on='Date', how='left')

    # Fill NaN
    for col in fred_cols:
        tech[col] = tech[col].fillna(method='ffill')
        energy[col] = energy[col].fillna(method='ffill')

    log_msg("✓ FRED merged (7 series)")

    # 4. TRAIN/VAL/TEST SPLIT
    log_msg("\n4. TRAIN/VAL/TEST SPLIT...")

    combined = pd.concat([tech, energy], ignore_index=True)
    dates = sorted(combined['Date'].unique())
    n = len(dates)

    train_end = dates[int(n * 0.70)]
    val_end = dates[int(n * 0.85)]

    train = combined[combined['Date'] <= train_end]
    val = combined[(combined['Date'] > train_end) & (combined['Date'] <= val_end)]
    test = combined[combined['Date'] > val_end]

    log_msg(f"✓ Train: {len(train)} ({len(train)/len(combined)*100:.1f}%)")
    log_msg(f"✓ Val: {len(val)} ({len(val)/len(combined)*100:.1f}%)")
    log_msg(f"✓ Test: {len(test)} ({len(test)/len(combined)*100:.1f}%)")

    # 5. GUARDAR
    log_msg("\n5. GUARDANDO DATOS...")

    tech.to_csv(os.path.join(OUTPUT_DIR, "tech_processed.csv"), index=False)
    energy.to_csv(os.path.join(OUTPUT_DIR, "energy_processed.csv"), index=False)
    train.to_csv(os.path.join(OUTPUT_DIR, "train_set.csv"), index=False)
    val.to_csv(os.path.join(OUTPUT_DIR, "val_set.csv"), index=False)
    test.to_csv(os.path.join(OUTPUT_DIR, "test_set.csv"), index=False)

    log_msg(f"✓ tech_processed.csv")
    log_msg(f"✓ energy_processed.csv")
    log_msg(f"✓ train_set.csv")
    log_msg(f"✓ val_set.csv")
    log_msg(f"✓ test_set.csv")

    # REPORTE
    log_msg("\n" + "="*70)
    log_msg("✓ PREPROCESSING COMPLETADO")
    log_msg("="*70)
    log_msg("\nFEATURES DISPONIBLES (todas conocidas al cierre del día t, sin fuga):")
    log_msg("  - Open, High, Low, Close, Volume, Return (día t, ya cerrado)")
    log_msg("  - SMA_20, SMA_50, Vol_20 (hasta día t inclusive)")
    log_msg("  - Close_Lag1, Close_Lag5, Return_Lag1, Return_Lag5")
    log_msg("  - Open_Lag1, High_Lag1, Low_Lag1, Volume_Lag1")
    log_msg("  - DCOILWTICO, VIXCLS, DHHNGSP, UNRATE, CPIAUCSL, FEDFUNDS, SP500")
    log_msg("\nTARGET: Target_Return = ln(Close_{t+1}/Close_t) (retorno del día siguiente)")
    log_msg("\nDatos listos para modelado en 05_Modelado/")

if __name__ == "__main__":
    main()
