"""
Minimum robustness analyses for reviewer response.

This script does not modify the original manuscript tables. It adds three
reviewer-facing diagnostics:
1. Multi-seed neural sensitivity for the leakage-free null-signal task.
2. Simple baselines on the same common sequential test subset.
3. Date-level cluster bootstrap for directional accuracy.
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, LSTM, LayerNormalization, MultiHeadAttention
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "04_Preprocessing"
MODEL_DIR = ROOT / "05_Modelado"
OUTPUT_DIR = Path(__file__).resolve().parent

SEQUENCE_LENGTH = 30
BATCH_SIZE = 32
EPOCHS = 30
BOOTSTRAP_REPS = 5000

FEATURE_COLS = [
    "SMA_20", "SMA_50", "Vol_20",
    "Close_Lag1", "Close_Lag5", "Return_Lag1", "Return_Lag5",
    "Open_Lag1", "High_Lag1", "Low_Lag1", "Volume_Lag1",
    "DCOILWTICO", "VIXCLS", "DHHNGSP", "UNRATE", "CPIAUCSL", "FEDFUNDS", "SP500",
]
TARGET_COL = "Target_Return"

SECTOR_MAP = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Technology", "TSLA": "Technology",
    "CVX": "Energy", "COP": "Energy", "XLE": "Energy", "MPC": "Energy",
}


def log(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_partition(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{name}_set.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def prepare_data(data: pd.DataFrame):
    data = data.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    data = data.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)

    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(data[FEATURE_COLS].values)
    y = data[TARGET_COL].values
    tickers = data["Ticker"].values
    dates = data["Date"].values
    return x_scaled, y, tickers, dates, data


def create_sequences_by_ticker(x, y, tickers, dates, seq_len=SEQUENCE_LENGTH):
    xs, ys, idxs, seq_dates, seq_tickers = [], [], [], [], []
    n = len(tickers)
    i = 0
    while i < n:
        j = i
        while j < n and tickers[j] == tickers[i]:
            j += 1
        for k in range(i, j - seq_len):
            target_idx = k + seq_len
            xs.append(x[k:k + seq_len])
            ys.append(y[target_idx])
            idxs.append(target_idx)
            seq_dates.append(dates[target_idx])
            seq_tickers.append(tickers[target_idx])
        i = j
    return (
        np.array(xs),
        np.array(ys),
        np.array(idxs),
        np.array(seq_dates),
        np.array(seq_tickers),
    )


def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, activation="tanh", return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, activation="tanh"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def build_attention(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = MultiHeadAttention(num_heads=4, key_dim=16)(inputs, inputs)
    x = LayerNormalization(epsilon=1e-6)(x + inputs)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)
    x = Dense(16, activation="relu")(x)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


def metric_row(y_true, y_pred, model, seed=None):
    y_pred = np.asarray(y_pred).reshape(-1)
    if np.allclose(y_pred, 0.0):
        directional_accuracy = np.nan
    else:
        directional_accuracy = np.mean(np.sign(y_pred) == np.sign(y_true))
    return {
        "Model": model,
        "Seed": seed,
        "N": len(y_true),
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "Directional_Accuracy": directional_accuracy,
    }


def reconstruct_aligned_test():
    train = load_partition("train")
    val = load_partition("val")
    test = load_partition("test")

    x_train, y_train, tickers_train, dates_train, train_clean = prepare_data(train)
    x_val, y_val, tickers_val, dates_val, val_clean = prepare_data(val)
    x_test, y_test, tickers_test, dates_test, test_clean = prepare_data(test)

    x_train_seq, y_train_seq, _, _, _ = create_sequences_by_ticker(x_train, y_train, tickers_train, dates_train)
    x_val_seq, y_val_seq, _, _, _ = create_sequences_by_ticker(x_val, y_val, tickers_val, dates_val)
    x_test_seq, y_test_seq, test_idx, seq_dates, seq_tickers = create_sequences_by_ticker(
        x_test, y_test, tickers_test, dates_test
    )

    predictions = pd.read_csv(MODEL_DIR / "predictions.csv")
    aligned = pd.DataFrame({
        "Date": pd.to_datetime(seq_dates),
        "Ticker": seq_tickers,
        "Sector": [SECTOR_MAP[t] for t in seq_tickers],
        "y_test": y_test_seq,
        "test_idx": test_idx,
    })
    for col in ["LSTM", "Transformer", "XGBoost", "Ensemble"]:
        aligned[col] = predictions[col].values

    if not np.allclose(aligned["y_test"].values, predictions["y_test"].values):
        raise ValueError("Aligned target does not match 05_Modelado/predictions.csv")
    if not np.array_equal(aligned["Ticker"].values, predictions["Ticker"].values):
        raise ValueError("Aligned ticker order does not match 05_Modelado/predictions.csv")

    return {
        "train": train_clean,
        "val": val_clean,
        "test": test_clean,
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "x_train_seq": x_train_seq,
        "y_train_seq": y_train_seq,
        "x_val_seq": x_val_seq,
        "y_val_seq": y_val_seq,
        "x_test_seq": x_test_seq,
        "y_test_seq": y_test_seq,
        "test_idx": test_idx,
        "aligned": aligned,
    }


def run_multi_seed_neural(data, seeds=range(10)):
    rows = []
    prediction_rows = []
    input_shape = (data["x_train_seq"].shape[1], data["x_train_seq"].shape[2])

    for seed in seeds:
        log(f"Training neural models with fixed seed={seed}")
        set_all_seeds(seed)
        tf.keras.backend.clear_session()
        lstm = build_lstm(input_shape)
        lstm.fit(
            data["x_train_seq"], data["y_train_seq"],
            validation_data=(data["x_val_seq"], data["y_val_seq"]),
            epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
        )
        pred_lstm = lstm.predict(data["x_test_seq"], verbose=0).flatten()
        rows.append(metric_row(data["y_test_seq"], pred_lstm, "LSTM", seed))

        set_all_seeds(seed)
        tf.keras.backend.clear_session()
        att = build_attention(input_shape)
        att.fit(
            data["x_train_seq"], data["y_train_seq"],
            validation_data=(data["x_val_seq"], data["y_val_seq"]),
            epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
        )
        pred_att = att.predict(data["x_test_seq"], verbose=0).flatten()
        rows.append(metric_row(data["y_test_seq"], pred_att, "Simplified attention", seed))

        prediction_rows.append(pd.DataFrame({
            "Seed": seed,
            "Date": data["aligned"]["Date"],
            "Ticker": data["aligned"]["Ticker"],
            "y_test": data["y_test_seq"],
            "LSTM": pred_lstm,
            "Simplified_attention": pred_att,
        }))

    metrics = pd.DataFrame(rows)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    return metrics, predictions


def run_baselines(data):
    y_train = data["y_train"]
    y_test_seq = data["y_test_seq"]
    test_idx = data["test_idx"]

    zero_pred = np.zeros_like(y_test_seq)
    train_mean_pred = np.full_like(y_test_seq, float(np.mean(y_train)))

    lag_cols = ["Return_Lag1", "Return_Lag5"]
    lag_idx = [FEATURE_COLS.index(c) for c in lag_cols]
    reg = LinearRegression()
    reg.fit(data["x_train"][:, lag_idx], data["y_train"])
    lag_pred = reg.predict(data["x_test"][test_idx][:, lag_idx])

    baseline_rows = [
        metric_row(y_test_seq, zero_pred, "Zero-return baseline"),
        metric_row(y_test_seq, train_mean_pred, "Train-mean baseline"),
        metric_row(y_test_seq, lag_pred, "Lagged-return OLS"),
    ]
    baseline_predictions = data["aligned"][["Date", "Ticker", "Sector", "y_test"]].copy()
    baseline_predictions["Zero_return"] = zero_pred
    baseline_predictions["Train_mean"] = train_mean_pred
    baseline_predictions["Lagged_return_OLS"] = lag_pred
    return pd.DataFrame(baseline_rows), baseline_predictions


def date_cluster_bootstrap(aligned, model_columns, reps=BOOTSTRAP_REPS, seed=20260917):
    rng = np.random.default_rng(seed)
    dates = np.array(sorted(aligned["Date"].unique()))
    y_sign = np.sign(aligned["y_test"].values)
    majority_baseline = max(float(np.mean(aligned["y_test"].values > 0)), float(np.mean(aligned["y_test"].values <= 0)))
    rows = []

    grouped_indices = {
        date: aligned.index[aligned["Date"] == date].to_numpy()
        for date in dates
    }

    for model in model_columns:
        correct = (np.sign(aligned[model].values) == y_sign).astype(float)
        observed = float(correct.mean())
        boot_stats = []
        for _ in range(reps):
            sampled_dates = rng.choice(dates, size=len(dates), replace=True)
            idx = np.concatenate([grouped_indices[d] for d in sampled_dates])
            boot_stats.append(float(correct[idx].mean()))
        boot_stats = np.array(boot_stats)

        # Two-sided bootstrap diagnostic around reference rates, preserving dates as clusters.
        p_chance = min(1.0, 2 * min(np.mean(boot_stats <= 0.5), np.mean(boot_stats >= 0.5)))
        p_majority = min(1.0, 2 * min(np.mean(boot_stats <= majority_baseline), np.mean(boot_stats >= majority_baseline)))

        rows.append({
            "Model": model,
            "N_asset_date": len(aligned),
            "N_dates": len(dates),
            "Observed_Directional_Accuracy": observed,
            "Bootstrap_CI95_Low": float(np.quantile(boot_stats, 0.025)),
            "Bootstrap_CI95_High": float(np.quantile(boot_stats, 0.975)),
            "Majority_Class_Baseline": majority_baseline,
            "Bootstrap_p_vs_0.5": p_chance,
            "Bootstrap_p_vs_majority": p_majority,
        })
    return pd.DataFrame(rows)


def summarize_multi_seed(metrics):
    return (
        metrics
        .groupby("Model")
        .agg(
            Seeds=("Seed", "nunique"),
            N=("N", "first"),
            MAE_mean=("MAE", "mean"),
            MAE_sd=("MAE", "std"),
            RMSE_mean=("RMSE", "mean"),
            RMSE_sd=("RMSE", "std"),
            R2_mean=("R2", "mean"),
            R2_sd=("R2", "std"),
            Directional_Accuracy_mean=("Directional_Accuracy", "mean"),
            Directional_Accuracy_sd=("Directional_Accuracy", "std"),
        )
        .reset_index()
    )


def write_report(summary, baseline_metrics, bootstrap, metadata):
    report = [
        "# Minimum robustness analyses for reviewer response",
        "",
        f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}",
        "",
        "## Execution metadata",
        "",
        "```json",
        json.dumps(metadata, indent=2),
        "```",
        "",
        "## Multi-seed neural sensitivity",
        "",
        "```",
        summary.to_string(index=False),
        "```",
        "",
        "## Simple baselines on the common sequential subset",
        "",
        "```",
        baseline_metrics.to_string(index=False),
        "```",
        "",
        "## Date-cluster bootstrap for directional accuracy",
        "",
        "```",
        bootstrap.to_string(index=False),
        "```",
        "",
    ]
    (OUTPUT_DIR / "minimum_revision_report.md").write_text("\n".join(report), encoding="utf-8")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log("Reconstructing aligned test subset")
    data = reconstruct_aligned_test()

    log("Running simple baselines")
    baseline_metrics, baseline_predictions = run_baselines(data)

    log("Running date-level cluster bootstrap on reported and baseline predictions")
    bootstrap_input = data["aligned"].copy()
    bootstrap_input["Zero_return"] = baseline_predictions["Zero_return"].values
    bootstrap_input["Train_mean"] = baseline_predictions["Train_mean"].values
    bootstrap_input["Lagged_return_OLS"] = baseline_predictions["Lagged_return_OLS"].values
    bootstrap = date_cluster_bootstrap(
        bootstrap_input,
        ["LSTM", "Transformer", "XGBoost", "Ensemble", "Train_mean", "Lagged_return_OLS"],
    )

    log("Running multi-seed neural sensitivity")
    neural_metrics, neural_predictions = run_multi_seed_neural(data, seeds=range(10))
    neural_summary = summarize_multi_seed(neural_metrics)

    metadata = {
        "feature_count": len(FEATURE_COLS),
        "features": FEATURE_COLS,
        "train_complete_case_rows": int(len(data["train"])),
        "validation_complete_case_rows": int(len(data["val"])),
        "test_complete_case_rows": int(len(data["test"])),
        "common_sequential_test_rows": int(len(data["aligned"])),
        "common_sequential_test_dates": int(data["aligned"]["Date"].nunique()),
        "sequence_length": SEQUENCE_LENGTH,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "learning_rate": 0.001,
        "loss": "mse",
        "neural_seeds": list(range(10)),
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "bootstrap_cluster": "Date",
    }

    data["aligned"].to_csv(OUTPUT_DIR / "aligned_common_test_subset.csv", index=False)
    baseline_metrics.to_csv(OUTPUT_DIR / "baseline_metrics.csv", index=False)
    baseline_predictions.to_csv(OUTPUT_DIR / "baseline_predictions.csv", index=False)
    bootstrap.to_csv(OUTPUT_DIR / "date_cluster_bootstrap_directional_accuracy.csv", index=False)
    neural_metrics.to_csv(OUTPUT_DIR / "neural_multiseed_metrics.csv", index=False)
    neural_summary.to_csv(OUTPUT_DIR / "neural_multiseed_summary.csv", index=False)
    neural_predictions.to_csv(OUTPUT_DIR / "neural_multiseed_predictions.csv", index=False)
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(neural_summary, baseline_metrics, bootstrap, metadata)

    log("Done")
    log(f"Outputs written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
