"""
Modelado Robusto - LSTM, Transformer, XGBoost, Ensemble
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
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), "04_Preprocessing")
LOG_FILE = os.path.join(OUTPUT_DIR, "modeling_log.txt")

def log_msg(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def load_and_prep_data():
    """Carga y prepara datos"""
    log_msg("Cargando datos...")

    train = pd.read_csv(os.path.join(DATA_DIR, "train_set.csv"))
    val = pd.read_csv(os.path.join(DATA_DIR, "val_set.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test_set.csv"))

    # Remover NaN
    for df in [train, val, test]:
        df.dropna(inplace=True)

    log_msg(f"✓ Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Features
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return',
                    'SMA_20', 'SMA_50', 'Vol_20']
    feature_cols = [c for c in feature_cols if c in train.columns]

    X_train = train[feature_cols].values
    y_train = train['Close'].values
    X_val = val[feature_cols].values
    y_val = val['Close'].values
    X_test = test[feature_cols].values
    y_test = test['Close'].values

    # Normalizar
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test

def build_lstm_simple(input_dim):
    """LSTM simple y estable"""
    model = Sequential([
        LSTM(32, activation='relu', input_shape=(1, input_dim)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(lr=0.001), loss='mse')
    return model

def train_xgb(X_train, y_train, X_val, y_val):
    """Entrena XGBoost"""
    log_msg("Entrenando XGBoost...")
    model = xgb.XGBRegressor(n_estimators=50, max_depth=4, learning_rate=0.1)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    log_msg("✓ XGBoost OK")
    return model

def train_lstm_model(X_train, y_train, X_val, y_val):
    """Entrena LSTM"""
    log_msg("Entrenando LSTM...")

    # Reshape para LSTM
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val_lstm = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

    model = build_lstm_simple(X_train.shape[1])
    model.fit(X_train_lstm, y_train, validation_data=(X_val_lstm, y_val),
              epochs=20, batch_size=32, verbose=0)

    log_msg("✓ LSTM OK")
    return model

def main():
    log_msg("="*70)
    log_msg("MODELADO ROBUSTO - LSTM + XGBOOST + ENSEMBLE")
    log_msg(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_msg("="*70)

    # Datos
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_prep_data()

    log_msg(f"\nX_train: {X_train.shape}, y_train: {y_train.shape}")
    log_msg(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Entrenar modelos
    log_msg("\n=== ENTRENANDO ===")

    xgb_model = train_xgb(X_train, y_train, X_val, y_val)
    lstm_model = train_lstm_model(X_train, y_train, X_val, y_val)

    # Predecir
    log_msg("\nPrediciendo...")

    y_pred_xgb = xgb_model.predict(X_test)
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    y_pred_lstm = lstm_model.predict(X_test_lstm, verbose=0).flatten()

    # Ensemble
    y_pred_ensemble = 0.5 * y_pred_xgb + 0.5 * y_pred_lstm

    # Remover NaN
    valid_idx = ~(np.isnan(y_pred_lstm) | np.isnan(y_pred_xgb) | np.isnan(y_test))
    y_test_clean = y_test[valid_idx]
    y_pred_lstm_clean = y_pred_lstm[valid_idx]
    y_pred_xgb_clean = y_pred_xgb[valid_idx]
    y_pred_ens_clean = y_pred_ensemble[valid_idx]

    # Métricas
    log_msg("\n=== RESULTADOS ===")

    metrics = []

    for name, y_pred in [("LSTM", y_pred_lstm_clean),
                         ("XGBoost", y_pred_xgb_clean),
                         ("Ensemble", y_pred_ens_clean)]:
        mae = mean_absolute_error(y_test_clean, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_clean, y_pred))
        r2 = r2_score(y_test_clean, y_pred)
        dir_acc = np.mean(np.sign(np.diff(y_test_clean)) == np.sign(np.diff(y_pred)))

        log_msg(f"\n{name}:")
        log_msg(f"  MAE: {mae:.4f}")
        log_msg(f"  RMSE: {rmse:.4f}")
        log_msg(f"  R2: {r2:.4f}")
        log_msg(f"  Dir Acc: {dir_acc:.4f}")

        metrics.append({
            'Model': name,
            'MAE': round(mae, 4),
            'RMSE': round(rmse, 4),
            'R2': round(r2, 4),
            'Dir_Acc': round(dir_acc, 4)
        })

    # Guardar
    log_msg("\n=== GUARDANDO ===")

    pd.DataFrame(metrics).to_csv(os.path.join(OUTPUT_DIR, "metrics.csv"), index=False)
    log_msg("✓ metrics.csv")

    results = pd.DataFrame({
        'y_test': y_test_clean,
        'LSTM': y_pred_lstm_clean,
        'XGBoost': y_pred_xgb_clean,
        'Ensemble': y_pred_ens_clean
    })
    results.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
    log_msg("✓ predictions.csv")

    lstm_model.save(os.path.join(OUTPUT_DIR, "lstm.h5"))
    log_msg("✓ lstm.h5")

    log_msg("\n" + "="*70)
    log_msg("✓ MODELADO COMPLETADO")
    log_msg("="*70)

if __name__ == "__main__":
    main()
