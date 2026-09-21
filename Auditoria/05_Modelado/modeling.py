"""
Modelado - LSTM, Transformer, XGBoost, Ensemble
Proyecto: Análisis Comparativo de Predicción de Mercados con IA
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import json
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), "04_Preprocessing")
LOG_FILE = os.path.join(OUTPUT_DIR, "modeling_log.txt")

SEQUENCE_LENGTH = 30
BATCH_SIZE = 32
EPOCHS = 30

SECTOR_MAP = {
    "AAPL": "Tecnología", "MSFT": "Tecnología", "NVDA": "Tecnología",
    "GOOGL": "Tecnología", "TSLA": "Tecnología",
    "CVX": "Energía", "COP": "Energía", "XLE": "Energía", "MPC": "Energía",
}

# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def log_msg(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def load_data():
    """Carga datos procesados"""
    log_msg("Cargando datos...")

    train = pd.read_csv(os.path.join(DATA_DIR, "train_set.csv"))
    val = pd.read_csv(os.path.join(DATA_DIR, "val_set.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test_set.csv"))

    for df in [train, val, test]:
        df['Date'] = pd.to_datetime(df['Date'])

    log_msg(f"✓ Train: {len(train)} registros")
    log_msg(f"✓ Val: {len(val)} registros")
    log_msg(f"✓ Test: {len(test)} registros")

    return train, val, test

FEATURE_COLS = [
    # Agregados hasta el día t inclusive (SMA/Vol incorporan Close_t, pero eso no es
    # fuga: predicen el retorno de t+1, no de t)
    'SMA_20', 'SMA_50', 'Vol_20',
    # Rezagos de precio y retorno
    'Close_Lag1', 'Close_Lag5', 'Return_Lag1', 'Return_Lag5',
    # Intradía rezagado 1 día (día t-1, ya cerrado)
    'Open_Lag1', 'High_Lag1', 'Low_Lag1', 'Volume_Lag1',
    # Factores económicos del día t (conocidos antes de predecir el retorno de t+1)
    'DCOILWTICO', 'VIXCLS', 'DHHNGSP', 'UNRATE', 'CPIAUCSL', 'FEDFUNDS', 'SP500',
]

def prepare_data(data, target_col='Target_Return'):
    """Prepara datos para modelado. Target = retorno del día siguiente
    (Target_Return, calculado en preprocessing); ninguna feature usa información
    del mismo día o posterior al target, evitando la fuga de Close_t->Close_t."""

    feature_cols = [c for c in FEATURE_COLS if c in data.columns]

    # Ordenar por Ticker y Fecha: create_sequences_by_ticker asume bloques
    # contiguos por ticker en orden cronológico.
    data = data.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    # Remover filas con NaN en cualquier feature o en el target (SMA_20/SMA_50/rezagos
    # tienen NaN en los primeros días de cada ventana por ticker; el target tiene NaN
    # en el último día de cada ticker, al no existir un "día siguiente").
    data = data.dropna(subset=feature_cols + [target_col])

    X = data[feature_cols].values
    y = data[target_col].values
    tickers = data['Ticker'].values

    # Normalizar
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, tickers, feature_cols

def create_sequences_by_ticker(X, y, tickers, seq_len=30):
    """Crea secuencias para LSTM/Transformer sin cruzar los límites de cada ticker.
    Devuelve también el índice (en el arreglo plano X/y) del target de cada
    secuencia, para poder alinear correctamente las predicciones de XGBoost
    (que no usa secuencias) con las de LSTM/Transformer más adelante."""
    Xs, ys, idxs = [], [], []
    n = len(tickers)
    i = 0
    while i < n:
        j = i
        while j < n and tickers[j] == tickers[i]:
            j += 1
        # Filas [i, j) pertenecen al mismo ticker, en orden cronológico
        for k in range(i, j - seq_len):
            Xs.append(X[k:k + seq_len])
            ys.append(y[k + seq_len])
            idxs.append(k + seq_len)
        i = j
    return np.array(Xs), np.array(ys), np.array(idxs)

def build_lstm(input_shape):
    """Construye modelo LSTM"""
    model = Sequential([
        LSTM(64, activation='tanh', return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, activation='tanh'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def build_transformer(input_shape):
    """Construye modelo Transformer simple"""
    inputs = keras.Input(shape=input_shape)

    # Attention
    x = MultiHeadAttention(num_heads=4, key_dim=16)(inputs, inputs)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)

    # Reducir la dimensión de secuencia antes de las capas densas: sin este pooling,
    # el modelo emite una predicción por cada paso de tiempo (batch, 30, 1) en vez de
    # una por secuencia (batch, 1), lo que rompe la comparación con y_test.
    x = keras.layers.GlobalAveragePooling1D()(x)

    # Dense layers
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation='relu')(x)

    # Output
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def train_xgboost(X_train, y_train, X_val, y_val):
    """Entrena XGBoost"""
    log_msg("\nEntrenando XGBoost...")

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        early_stopping_rounds=10
    )

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)

    log_msg("✓ XGBoost entrenado")
    return model

def train_lstm(X_train, y_train, X_val, y_val):
    """Entrena LSTM"""
    log_msg("\nEntrenando LSTM...")

    model = build_lstm((X_train.shape[1], X_train.shape[2]))

    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              verbose=0)

    log_msg(f"✓ LSTM entrenado ({EPOCHS} epochs)")
    return model

def train_transformer(X_train, y_train, X_val, y_val):
    """Entrena Transformer"""
    log_msg("\nEntrenando Transformer...")

    model = build_transformer((X_train.shape[1], X_train.shape[2]))

    model.fit(X_train, y_train,
              validation_data=(X_val, y_val),
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              verbose=0)

    log_msg(f"✓ Transformer entrenado ({EPOCHS} epochs)")
    return model

def evaluate_model(y_true, y_pred, model_name):
    """Calcula métricas"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Directional accuracy: el target ya es un retorno, por lo que la dirección
    # correcta es sign(y_pred) == sign(y_true) observación por observación (no la
    # diferencia entre observaciones consecutivas, que no tiene sentido para retornos).
    dir_acc = np.mean(np.sign(y_pred) == np.sign(y_true))

    metrics = {
        'Model': model_name,
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'R2': round(r2, 4),
        'Directional_Accuracy': round(dir_acc, 4)
    }

    log_msg(f"\n{model_name}:")
    log_msg(f"  MAE: {mae:.4f}")
    log_msg(f"  RMSE: {rmse:.4f}")
    log_msg(f"  R2: {r2:.4f}")
    log_msg(f"  Dir Acc: {dir_acc:.4f}")

    return metrics

# ============================================================================
# MAIN
# ============================================================================

def main():
    log_msg("="*70)
    log_msg("MODELADO - LSTM, TRANSFORMER, XGBOOST, ENSEMBLE")
    log_msg(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_msg("="*70)

    # 1. Cargar datos
    train, val, test = load_data()

    # 2. Preparar datos
    log_msg("\n=== PREPARANDO DATOS ===")
    X_train, y_train, tickers_train, features = prepare_data(train)
    X_val, y_val, tickers_val, _ = prepare_data(val)
    X_test, y_test, tickers_test, _ = prepare_data(test)

    log_msg(f"✓ Features: {len(features)}")
    log_msg(f"  {', '.join(features)}")

    # 3. Crear secuencias para LSTM/Transformer (sin cruzar límites de ticker)
    log_msg(f"\nCreando secuencias (length={SEQUENCE_LENGTH})...")
    X_train_seq, y_train_seq, _ = create_sequences_by_ticker(X_train, y_train, tickers_train, SEQUENCE_LENGTH)
    X_val_seq, y_val_seq, _ = create_sequences_by_ticker(X_val, y_val, tickers_val, SEQUENCE_LENGTH)
    X_test_seq, y_test_seq, test_idx = create_sequences_by_ticker(X_test, y_test, tickers_test, SEQUENCE_LENGTH)

    log_msg(f"✓ Train seq: {X_train_seq.shape}")
    log_msg(f"✓ Val seq: {X_val_seq.shape}")
    log_msg(f"✓ Test seq: {X_test_seq.shape}")

    # 4. Entrenar modelos
    log_msg("\n=== ENTRENANDO MODELOS ===")

    # LSTM
    lstm_model = train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
    y_pred_lstm = lstm_model.predict(X_test_seq, verbose=0).flatten()
    metrics_lstm = evaluate_model(y_test_seq, y_pred_lstm, "LSTM")

    # Transformer
    transformer_model = train_transformer(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
    y_pred_transformer = transformer_model.predict(X_test_seq, verbose=0).flatten()
    metrics_transformer = evaluate_model(y_test_seq, y_pred_transformer, "Transformer")

    # XGBoost
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    y_pred_xgb = xgb_model.predict(X_test)

    # XGBoost predice sobre X_test (todas las filas, 1.755). Las secuencias
    # LSTM/Transformer se construyen por ticker y pierden los primeros 30 días de
    # cada ticker (necesarios como ventana de entrada), por lo que su conjunto de
    # evaluación es más pequeño (1.476 filas). test_idx guarda, para cada secuencia,
    # la posición exacta (en el arreglo plano X_test/y_test) de su target: se usa
    # para restringir XGBoost al MISMO subconjunto de 1.476 observaciones que los
    # demás modelos, de modo que las cuatro filas de la Tabla 1 sean comparables
    # (mismo objetivo, mismo tamaño de muestra). Evaluar XGBoost sobre las 1.755
    # filas completas —como se hacía antes— y a la vez reportar sus predicciones
    # alineadas (1.476 filas) en predictions.csv producía dos muestras distintas
    # bajo una sola fila de tabla, lo que rompía la identidad R²=1-SS_res/SS_tot
    # entre modelos y desalineaba el MAE de la Tabla 1 con el "error medio" de la
    # Tabla 2.
    y_pred_xgb_aligned = y_pred_xgb[test_idx]
    assert np.allclose(y_test[test_idx], y_test_seq), "Desalineación entre XGBoost y secuencias LSTM/Transformer"

    metrics_xgb = evaluate_model(y_test_seq, y_pred_xgb_aligned, "XGBoost")

    # 5. Ensemble
    log_msg("\n=== ENSEMBLE ===")

    # Predicts con pesos
    y_pred_ensemble = (0.4 * y_pred_lstm +
                       0.4 * y_pred_transformer +
                       0.2 * y_pred_xgb_aligned)

    metrics_ensemble = evaluate_model(y_test_seq, y_pred_ensemble, "Ensemble")

    # 6. Guardar resultados
    log_msg("\n=== GUARDANDO RESULTADOS ===")

    # Métricas
    all_metrics = pd.DataFrame([
        metrics_lstm,
        metrics_transformer,
        metrics_xgb,
        metrics_ensemble
    ])

    all_metrics.to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)
    log_msg("✓ metrics.csv")

    # Predicciones (con Ticker/Sector alineados vía test_idx, para permitir
    # desglose sectorial real en 06_Evaluacion, en vez del promedio global
    # etiquetado incorrectamente como "por sector" de versiones anteriores)
    tickers_seq = tickers_test[test_idx]
    sectors_seq = np.array([SECTOR_MAP.get(t, "Desconocido") for t in tickers_seq])

    results = pd.DataFrame({
        'Ticker': tickers_seq,
        'Sector': sectors_seq,
        'y_test': y_test_seq,
        'LSTM': y_pred_lstm,
        'Transformer': y_pred_transformer,
        'XGBoost': y_pred_xgb_aligned,
        'Ensemble': y_pred_ensemble
    })

    results.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
    log_msg("✓ predictions.csv")

    # Guardar modelos
    lstm_model.save(os.path.join(OUTPUT_DIR, "lstm_model.h5"))
    transformer_model.save(os.path.join(OUTPUT_DIR, "transformer_model.h5"))
    xgb_model.save_model(os.path.join(OUTPUT_DIR, "xgb_model.json"))

    log_msg("✓ Modelos guardados (.h5, .json)")

    # Reporte
    log_msg("\n" + "="*70)
    log_msg("✓ MODELADO COMPLETADO")
    log_msg("="*70)
    log_msg("\nRESULTADOS COMPARATIVOS:")
    log_msg(f"\n{all_metrics.to_string()}")
    log_msg("\nPróximo paso: Evaluación en 06_Evaluación/")

if __name__ == "__main__":
    main()
