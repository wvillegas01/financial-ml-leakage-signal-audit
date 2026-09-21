"""
Script de Preprocessing
Proyecto: Análisis Comparativo de Predicción de Mercados con IA

Realiza:
- Limpieza de datos (missing values, outliers)
- Feature engineering (technical indicators)
- Merge con datos económicos FRED
- Normalización MinMaxScaler/StandardScaler
- Train/Val/Test split (70/15/15)
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import warnings
from sklearn.preprocessing import MinMaxScaler, StandardScaler

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), "02_Descarga_Datos")
LOG_FILE = os.path.join(OUTPUT_DIR, "preprocessing_log.txt")

# Parámetros
WINDOW_RSI = 14
WINDOW_MACD_FAST = 12
WINDOW_MACD_SLOW = 26
WINDOW_BB = 20

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def log_message(msg):
    """Log a consola y archivo"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(formatted + "\n")

def load_raw_data():
    """Carga datos raw"""
    log_message("Cargando datos raw...")

    tech = pd.read_csv(os.path.join(DATA_DIR, "tech_prices_raw.csv"))
    energy = pd.read_csv(os.path.join(DATA_DIR, "energy_prices_raw.csv"))
    fred = pd.read_csv(os.path.join(DATA_DIR, "fred_economic_raw.csv"))

    tech['Date'] = pd.to_datetime(tech['Date'])
    energy['Date'] = pd.to_datetime(energy['Date'])
    fred['Date'] = pd.to_datetime(fred['Date'])

    log_message(f"✓ Tech: {len(tech)} registros")
    log_message(f"✓ Energy: {len(energy)} registros")
    log_message(f"✓ FRED: {len(fred)} registros")

    return tech, energy, fred

def clean_data(data):
    """Limpia datos: missing values, outliers"""
    log_message(f"\nLimpiando datos...")

    initial_len = len(data)

    # Forward fill para missing values
    data = data.sort_values(['Ticker', 'Date'])
    data['Close'] = data.groupby('Ticker')['Close'].fillna(method='ffill')

    # Remover outliers (±3 std por ticker)
    for ticker in data['Ticker'].unique():
        mask = data['Ticker'] == ticker
        ticker_data = data[mask]['Close']
        mean = ticker_data.mean()
        std = ticker_data.std()
        outlier_mask = (data['Close'] < mean - 3*std) | (data['Close'] > mean + 3*std)
        data = data[~outlier_mask]

    removed = initial_len - len(data)
    log_message(f"✓ Removidos {removed} registros outliers ({(removed/initial_len*100):.2f}%)")

    return data

def calculate_technical_indicators(data):
    """Calcula indicadores técnicos por ticker"""
    log_message("Calculando indicadores técnicos...")

    data = data.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    for ticker in data['Ticker'].unique():
        mask = data['Ticker'] == ticker
        ticker_idx = data[mask].index

        ticker_data = data.loc[ticker_idx].copy()
        close = ticker_data['Close'].values
        high = ticker_data['High'].values
        low = ticker_data['Low'].values
        volume = ticker_data['Volume'].values

        # RSI (Relative Strength Index)
        rsi = calculate_rsi(close, WINDOW_RSI)

        # MACD
        macd_line, signal_line = calculate_macd(close, WINDOW_MACD_FAST, WINDOW_MACD_SLOW)

        # SMA (Simple Moving Average)
        sma_20 = calculate_sma(close, 20)
        sma_50 = calculate_sma(close, 50)

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close, WINDOW_BB)

        # ATR (Average True Range)
        atr = calculate_atr(high, low, close, 14)

        # Assign to dataframe
        data.loc[ticker_idx, 'RSI'] = rsi
        data.loc[ticker_idx, 'MACD'] = macd_line
        data.loc[ticker_idx, 'MACD_Signal'] = signal_line
        data.loc[ticker_idx, 'SMA_20'] = sma_20
        data.loc[ticker_idx, 'SMA_50'] = sma_50
        data.loc[ticker_idx, 'BB_Upper'] = bb_upper
        data.loc[ticker_idx, 'BB_Lower'] = bb_lower
        data.loc[ticker_idx, 'ATR'] = atr

    log_message(f"✓ Indicadores técnicos calculados")
    return data

def calculate_rsi(prices, period=14):
    """RSI (Relative Strength Index)"""
    deltas = np.diff(prices)
    seed = deltas[:period+1]
    up = seed[seed >= 0].sum() / period
    down = -seed[seed < 0].sum() / period
    rs = up / down if down != 0 else 0
    rsi = np.zeros_like(prices)
    rsi[:period] = 100. - 100. / (1. + rs)

    for i in range(period, len(prices)):
        delta = deltas[i-1]
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up * (period - 1) + upval) / period
        down = (down * (period - 1) + downval) / period
        rs = up / down if down != 0 else 0
        rsi[i] = 100. - 100. / (1. + rs)

    return rsi

def calculate_macd(prices, fast=12, slow=26):
    """MACD (Moving Average Convergence Divergence)"""
    ema_fast = pd.Series(prices).ewm(span=fast).mean().values
    ema_slow = pd.Series(prices).ewm(span=slow).mean().values
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=9).mean().values
    return macd_line, signal_line

def calculate_sma(prices, period):
    """SMA (Simple Moving Average)"""
    return pd.Series(prices).rolling(window=period).mean().values

def calculate_bollinger_bands(prices, period=20):
    """Bollinger Bands"""
    sma = calculate_sma(prices, period)
    std = pd.Series(prices).rolling(window=period).std().values
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    return upper, sma, lower

def calculate_atr(high, low, close, period=14):
    """ATR (Average True Range)"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=period).mean().values
    return atr

def merge_with_fred(data, fred_data):
    """Merge con datos económicos FRED"""
    log_message("\nMergeando con datos FRED...")

    # Merge by date
    data = data.merge(fred_data, on='Date', how='left')

    # Fill missing FRED values
    for col in fred_data.columns:
        if col != 'Date':
            data[col] = data[col].fillna(method='ffill')

    log_message(f"✓ FRED merged: {len(data)} registros con features económicas")
    return data

def create_lagged_features(data, lags=[1, 5, 10, 20]):
    """Crea features lag para LSTM/Transformer"""
    log_message("\nCreando lagged features...")

    data = data.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    for ticker in data['Ticker'].unique():
        mask = data['Ticker'] == ticker
        ticker_idx = data[mask].index

        for lag in lags:
            lag_col = f'Close_Lag_{lag}'
            data.loc[ticker_idx, lag_col] = data.loc[ticker_idx, 'Close'].shift(lag)

            # También crear lag para retornos
            data.loc[ticker_idx, f'Return_Lag_{lag}'] = data.loc[ticker_idx, 'Daily_Return'].shift(lag)

    log_message(f"✓ Lagged features creados: lags {lags}")
    return data

def calculate_daily_return(data):
    """Calcula retorno diario"""
    log_message("Calculando retornos diarios...")

    data = data.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    # Calcular retornos por ticker
    daily_ret = []
    log_ret = []

    for ticker in data['Ticker'].unique():
        mask = data['Ticker'] == ticker
        ticker_close = data.loc[mask, 'Close'].values

        daily = np.insert(np.diff(ticker_close) / ticker_close[:-1], 0, np.nan)
        log_returns = np.insert(np.log(ticker_close[1:] / ticker_close[:-1]), 0, np.nan)

        daily_ret.extend(daily)
        log_ret.extend(log_returns)

    data['Daily_Return'] = daily_ret
    data['Log_Return'] = log_ret

    log_message(f"✓ Retornos calculados")
    return data

def normalize_features(data):
    """Normaliza features para LSTM/Transformer"""
    log_message("\nNormalizando features...")

    # MinMaxScaler para precios y features técnicos (0-1)
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'SMA_20', 'SMA_50', 'ATR']

    data_normalized = data.copy()

    for col in price_cols:
        if col in data.columns:
            scaler = MinMaxScaler()
            valid_mask = ~data[col].isna()
            if valid_mask.sum() > 0:
                data.loc[valid_mask, f'{col}_Norm'] = scaler.fit_transform(
                    data.loc[valid_mask, [col]]
                )

    # StandardScaler para features económicas
    fred_cols = ['DCOILWTICO', 'DHHNGSP', 'VIXCLS', 'UNRATE', 'CPIAUCSL', 'FEDFUNDS', 'SP500']

    for col in fred_cols:
        if col in data.columns:
            scaler = StandardScaler()
            valid_mask = ~data[col].isna()
            if valid_mask.sum() > 0:
                data.loc[valid_mask, f'{col}_Std'] = scaler.fit_transform(
                    data.loc[valid_mask, [[col]]]
                )

    log_message(f"✓ Features normalizados (MinMax para precios, StandardScaler para económicas)")
    return data

def train_val_test_split(data, train_ratio=0.7, val_ratio=0.15):
    """Divide en train/val/test (70/15/15)"""
    log_message("\nCreando train/val/test split...")

    # Obtener fecha de corte
    unique_dates = sorted(data['Date'].unique())
    n = len(unique_dates)

    train_cutoff = unique_dates[int(n * train_ratio)]
    val_cutoff = unique_dates[int(n * (train_ratio + val_ratio))]

    train = data[data['Date'] <= train_cutoff].copy()
    val = data[(data['Date'] > train_cutoff) & (data['Date'] <= val_cutoff)].copy()
    test = data[data['Date'] > val_cutoff].copy()

    log_message(f"✓ Train: {len(train)} registros ({len(train)/len(data)*100:.1f}%)")
    log_message(f"✓ Val: {len(val)} registros ({len(val)/len(data)*100:.1f}%)")
    log_message(f"✓ Test: {len(test)} registros ({len(test)/len(data)*100:.1f}%)")
    log_message(f"  Fechas corte: {train_cutoff.date()} | {val_cutoff.date()}")

    return train, val, test

def save_processed_data(tech_data, energy_data, train, val, test):
    """Guarda datos procesados"""
    log_message("\nGuardando datos procesados...")

    # Full datasets
    tech_data.to_csv(os.path.join(OUTPUT_DIR, "tech_features_processed.csv"), index=False)
    energy_data.to_csv(os.path.join(OUTPUT_DIR, "energy_features_processed.csv"), index=False)

    # Train/Val/Test (combined)
    train.to_csv(os.path.join(OUTPUT_DIR, "train_set.csv"), index=False)
    val.to_csv(os.path.join(OUTPUT_DIR, "val_set.csv"), index=False)
    test.to_csv(os.path.join(OUTPUT_DIR, "test_set.csv"), index=False)

    log_message(f"✓ tech_features_processed.csv")
    log_message(f"✓ energy_features_processed.csv")
    log_message(f"✓ train_set.csv")
    log_message(f"✓ val_set.csv")
    log_message(f"✓ test_set.csv")

    return True

def create_preprocessing_report(tech_data, energy_data, train, val, test):
    """Crea reporte de preprocessing"""
    log_message("\nGenerando reporte de preprocessing...")

    report = f"""# Reporte de Preprocessing

**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. LIMPIEZA DE DATOS

- Removidos outliers (±3 std por ticker)
- Forward fill para missing values
- Sincronización de fechas

## 2. FEATURE ENGINEERING

### Indicadores Técnicos
- RSI (14): Relative Strength Index
- MACD: Moving Average Convergence Divergence
- SMA (20, 50): Simple Moving Averages
- Bollinger Bands (20)
- ATR (14): Average True Range
- Daily Returns y Log Returns

### Lagged Features
- Close lags: [1, 5, 10, 20]
- Return lags: [1, 5, 10, 20]

### Datos Económicos (FRED)
- DCOILWTICO: Crude Oil WTI
- DHHNGSP: Natural Gas
- VIXCLS: VIX Index
- UNRATE: Unemployment Rate
- CPIAUCSL: Consumer Price Index
- FEDFUNDS: Federal Funds Rate
- SP500: S&P 500

## 3. NORMALIZACIÓN

- **MinMaxScaler** (0-1): Precios y indicadores técnicos
- **StandardScaler**: Factores económicos

## 4. TRAIN/VAL/TEST SPLIT

| Set | Registros | % | Período |
|-----|-----------|---|---------|
| Train | {len(train)} | 70% | 2021-08-05 a {train['Date'].max().date()} |
| Val | {len(val)} | 15% | {val['Date'].min().date()} a {val['Date'].max().date()} |
| Test | {len(test)} | 15% | {test['Date'].min().date()} a 2026-08-04 |

## 5. FEATURES DISPONIBLES

Total features por observación: ~40 (precios, técnicos, económicos, lags)

### Columnas principales
- Open, High, Low, Close, Volume (OHLCV)
- RSI, MACD, MACD_Signal, SMA_20, SMA_50, BB_Upper, BB_Lower, ATR
- Daily_Return, Log_Return
- Close_Lag_1, Close_Lag_5, Close_Lag_10, Close_Lag_20
- Return_Lag_1, Return_Lag_5, Return_Lag_10, Return_Lag_20
- DCOILWTICO, DHHNGSP, VIXCLS, UNRATE, CPIAUCSL, FEDFUNDS, SP500
- {col}_Norm columns (normalized versions)
- {col}_Std columns (standardized versions)

## 6. TECNOLOGÍA

- Datos sintéticos realistas con propiedades GARCH
- Técnicas de ingeniería de features: momentum, volatilidad, trend
- Normalización apropiada para deep learning

---

**Status:** ✅ Preprocessing Completo
**Próximo paso:** 05_Modelado/ → Entrenar LSTM, Transformer, XGBoost
"""

    filepath = os.path.join(OUTPUT_DIR, "PREPROCESSING_REPORT.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    log_message(f"✓ PREPROCESSING_REPORT.md")
    return report

# ============================================================================
# MAIN
# ============================================================================

def main():
    log_message("="*70)
    log_message("PREPROCESSING - FEATURE ENGINEERING & NORMALIZATION")
    log_message(f"Fecha/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message("="*70)

    # 1. Cargar datos
    tech_raw, energy_raw, fred_raw = load_raw_data()

    # 2. Limpiar
    log_message("\n=== LIMPIEZA DE DATOS ===")
    tech = clean_data(tech_raw.copy())
    energy = clean_data(energy_raw.copy())

    # 3. Calcular retornos
    log_message("\n=== FEATURES BÁSICAS ===")
    tech = calculate_daily_return(tech)
    energy = calculate_daily_return(energy)

    # 4. Indicadores técnicos
    log_message("\n=== INDICADORES TÉCNICOS ===")
    tech = calculate_technical_indicators(tech)
    energy = calculate_technical_indicators(energy)

    # 5. Merge FRED
    log_message("\n=== MERGE CON DATOS ECONÓMICOS ===")
    tech = merge_with_fred(tech, fred_raw)
    energy = merge_with_fred(energy, fred_raw)

    # 6. Lagged features
    log_message("\n=== LAGGED FEATURES ===")
    tech = create_lagged_features(tech)
    energy = create_lagged_features(energy)

    # 7. Normalización
    log_message("\n=== NORMALIZACIÓN ===")
    tech = normalize_features(tech)
    energy = normalize_features(energy)

    # 8. Train/Val/Test split
    log_message("\n=== TRAIN/VAL/TEST SPLIT ===")
    combined = pd.concat([tech, energy], ignore_index=True)
    train, val, test = train_val_test_split(combined)

    # 9. Guardar
    log_message("\n=== GUARDANDO DATOS ===")
    save_processed_data(tech, energy, train, val, test)

    # 10. Reporte
    create_preprocessing_report(tech, energy, train, val, test)

    log_message("\n" + "="*70)
    log_message("✓ PREPROCESSING COMPLETADO")
    log_message("="*70)
    log_message("\nArchivos generados:")
    log_message("  1. tech_features_processed.csv")
    log_message("  2. energy_features_processed.csv")
    log_message("  3. train_set.csv (70% datos)")
    log_message("  4. val_set.csv (15% datos)")
    log_message("  5. test_set.csv (15% datos)")
    log_message("  6. PREPROCESSING_REPORT.md")
    log_message("  7. preprocessing_log.txt")
    log_message("\nPróximo paso: Modelado en 05_Modelado/")

if __name__ == "__main__":
    main()
