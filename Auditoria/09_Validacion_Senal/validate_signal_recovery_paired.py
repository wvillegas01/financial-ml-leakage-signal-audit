"""
Paired controlled-signal recovery experiment.

This script implements the reviewer-requested correction:
- VIX change is computed chronologically before using the existing train/val/test split.
- Within each seed, the same noise vectors are reused across all gamma levels.
- Results are aggregated with sample standard deviations (ddof=1).

Outputs are written to:
Auditoria/09_Validacion_Senal/
"""

from __future__ import annotations

import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler


PROJECT = Path(r"C:\Users\wilop\Dropbox\MPDI\2026\AI in Business")
DATA_DIR = PROJECT / "Auditoria" / "04_Preprocessing"
OUTPUT_DIR = PROJECT / "Auditoria" / "09_Validacion_Senal"

FEATURE_COLS = [
    "SMA_20", "SMA_50", "Vol_20",
    "Close_Lag1", "Close_Lag5", "Return_Lag1", "Return_Lag5",
    "Open_Lag1", "High_Lag1", "Low_Lag1", "Volume_Lag1",
    "DCOILWTICO", "VIXCLS", "DHHNGSP", "UNRATE", "CPIAUCSL", "FEDFUNDS", "SP500",
]
REAL_TARGET_COL = "Target_Return"
VIX_FEATURE = "VIX_Change_Std"
GAMMAS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10]
N_SEEDS = 10
FEATURE_NAMES = FEATURE_COLS + [VIX_FEATURE]

PREVIOUS_RESULTS = {
    0.000: {"R2_mean": -0.0026, "R2_sd": 0.0014, "DirAcc_mean": 0.4971, "DirAcc_sd": 0.0108, "SHAP_mean": 0.0397, "SHAP_sd": 0.0128, "Rank_mean": 10.9},
    0.005: {"R2_mean": 0.0069, "R2_sd": 0.0067, "DirAcc_mean": 0.5332, "DirAcc_sd": 0.0094, "SHAP_mean": 0.2955, "SHAP_sd": 0.0494, "Rank_mean": 1.0},
    0.010: {"R2_mean": 0.0701, "R2_sd": 0.0147, "DirAcc_mean": 0.5857, "DirAcc_sd": 0.0066, "SHAP_mean": 0.4451, "SHAP_sd": 0.0494, "Rank_mean": 1.0},
    0.020: {"R2_mean": 0.2597, "R2_sd": 0.0118, "DirAcc_mean": 0.6730, "DirAcc_sd": 0.0097, "SHAP_mean": 0.6252, "SHAP_sd": 0.0363, "Rank_mean": 1.0},
    0.050: {"R2_mean": 0.7085, "R2_sd": 0.0074, "DirAcc_mean": 0.8187, "DirAcc_sd": 0.0091, "SHAP_mean": 0.7508, "SHAP_sd": 0.0455, "Rank_mean": 1.0},
    0.100: {"R2_mean": 0.9052, "R2_sd": 0.0049, "DirAcc_mean": 0.9032, "DirAcc_sd": 0.0050, "SHAP_mean": 0.8285, "SHAP_sd": 0.0274, "Rank_mean": 1.0},
}


def preserve_previous_outputs() -> None:
    for name in [
        "signal_recovery_results_paired_summary.csv",
        "signal_recovery_results_paired_runs.csv",
        "signal_recovery_validation_report.txt",
        "signal_recovery_table_latex.tex",
    ]:
        path = OUTPUT_DIR / name
        if path.exists():
            previous = path.with_name(f"{path.stem}_previous{path.suffix}")
            shutil.copy2(path, previous)


def load_split(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{name}_set.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    df["split"] = name
    return df


def load_and_prepare() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    train_raw = load_split("train")
    val_raw = load_split("val")
    test_raw = load_split("test")

    combined = pd.concat([train_raw, val_raw, test_raw], ignore_index=True)
    combined = combined.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    combined["VIX_Change_Raw"] = combined.groupby("Ticker")["VIXCLS"].diff()

    required = FEATURE_COLS + [REAL_TARGET_COL, "VIX_Change_Raw"]
    train = combined[combined["split"] == "train"].dropna(subset=required).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    val = combined[combined["split"] == "val"].dropna(subset=required).sort_values(["Ticker", "Date"]).reset_index(drop=True)
    test = combined[combined["split"] == "test"].dropna(subset=required).sort_values(["Ticker", "Date"]).reset_index(drop=True)

    mu = train["VIX_Change_Raw"].mean()
    sigma = train["VIX_Change_Raw"].std(ddof=0)
    if not np.isfinite(sigma) or sigma == 0:
        raise ValueError("Invalid VIX change training standard deviation.")

    for df in (train, val, test):
        df[VIX_FEATURE] = (df["VIX_Change_Raw"] - mu) / sigma

    split_dates = {
        "train_min": train["Date"].min(),
        "train_max": train["Date"].max(),
        "val_min": val["Date"].min(),
        "val_max": val["Date"].max(),
        "test_min": test["Date"].min(),
        "test_max": test["Date"].max(),
    }
    metadata = {
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "vix_mu_train": mu,
        "vix_sigma_train": sigma,
        "split_dates": split_dates,
    }
    return train, val, test, metadata


def validate_base_arrays(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    scaler = MinMaxScaler()
    X_train_base = scaler.fit_transform(train[FEATURE_COLS].values)
    X_val_base = scaler.transform(val[FEATURE_COLS].values)
    X_test_base = scaler.transform(test[FEATURE_COLS].values)

    X_train = np.column_stack([X_train_base, train[VIX_FEATURE].values])
    X_val = np.column_stack([X_val_base, val[VIX_FEATURE].values])
    X_test = np.column_stack([X_test_base, test[VIX_FEATURE].values])

    checks = []
    for label, arr in [("X_train", X_train), ("X_val", X_val), ("X_test", X_test)]:
        checks.append(f"{label}_shape={arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{label} contains NaN or infinite values.")
        zero_range = np.where(np.ptp(arr, axis=0) == 0)[0]
        if len(zero_range):
            checks.append(f"{label}_zero_range_predictors={zero_range.tolist()}")
        else:
            checks.append(f"{label}_zero_range_predictors=[]")

    if X_test.shape[1] != 19:
        raise ValueError(f"X_test must have 19 predictors, got {X_test.shape[1]}.")
    return X_train, X_val, X_test, checks


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.sign(y_pred) == np.sign(y_true)))


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    preserve_previous_outputs()
    train, val, test, metadata = load_and_prepare()
    X_train, X_val, X_test, array_checks = validate_base_arrays(train, val, test)
    vix_idx = FEATURE_NAMES.index(VIX_FEATURE)
    sigma_target = float(train[REAL_TARGET_COL].std(ddof=0))

    runs = []
    noise_fingerprints: dict[tuple[int, str], tuple[float, float, float]] = {}

    for seed_i in range(N_SEEDS):
        rng = np.random.default_rng(seed_i)
        noise_train = rng.normal(0, sigma_target, size=len(train))
        noise_val = rng.normal(0, sigma_target, size=len(val))
        noise_test = rng.normal(0, sigma_target, size=len(test))

        noise_fingerprints[(seed_i, "train")] = (float(noise_train.mean()), float(noise_train.std(ddof=0)), float(noise_train[:5].sum()))
        noise_fingerprints[(seed_i, "val")] = (float(noise_val.mean()), float(noise_val.std(ddof=0)), float(noise_val[:5].sum()))
        noise_fingerprints[(seed_i, "test")] = (float(noise_test.mean()), float(noise_test.std(ddof=0)), float(noise_test[:5].sum()))

        for gamma in GAMMAS:
            y_train = gamma * train[VIX_FEATURE].values + noise_train
            y_val = gamma * val[VIX_FEATURE].values + noise_val
            y_test = gamma * test[VIX_FEATURE].values + noise_test

            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=seed_i,
                early_stopping_rounds=10,
            )
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

            y_pred = model.predict(X_test)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)
            if len(shap_values) != len(test):
                raise ValueError("SHAP was not computed over the full X_test.")
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_sum = float(mean_abs_shap.sum())
            shap_pct = mean_abs_shap / shap_sum
            shap_pct_sum = float(shap_pct.sum())
            if not math.isclose(shap_pct_sum, 1.0, rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"Normalized SHAP importance does not sum to 1: {shap_pct_sum}")

            vix_shap_pct = float(shap_pct[vix_idx])
            vix_rank = int((shap_pct > vix_shap_pct).sum() + 1)
            if not 1 <= vix_rank <= 19:
                raise ValueError(f"Invalid SHAP rank: {vix_rank}")

            runs.append({
                "seed": seed_i,
                "gamma": gamma,
                "R2": float(r2_score(y_test, y_pred)),
                "MAE": float(mean_absolute_error(y_test, y_pred)),
                "Directional_Accuracy": directional_accuracy(y_test, y_pred),
                "SHAP_VIX_Change_Std_Normalized": vix_shap_pct,
                "SHAP_VIX_Change_Std_Rank": vix_rank,
                "SHAP_Normalized_Sum": shap_pct_sum,
                "n_train": len(train),
                "n_val": len(val),
                "n_test": len(test),
            })

    runs_df = pd.DataFrame(runs)
    agg = runs_df.groupby("gamma", as_index=False).agg(
        R2_mean=("R2", "mean"),
        R2_sd=("R2", lambda s: s.std(ddof=1)),
        MAE_mean=("MAE", "mean"),
        MAE_sd=("MAE", lambda s: s.std(ddof=1)),
        DirAcc_mean=("Directional_Accuracy", "mean"),
        DirAcc_sd=("Directional_Accuracy", lambda s: s.std(ddof=1)),
        SHAP_mean=("SHAP_VIX_Change_Std_Normalized", "mean"),
        SHAP_sd=("SHAP_VIX_Change_Std_Normalized", lambda s: s.std(ddof=1)),
        Rank_mean=("SHAP_VIX_Change_Std_Rank", "mean"),
        Rank_sd=("SHAP_VIX_Change_Std_Rank", lambda s: s.std(ddof=1)),
        n_runs=("seed", "count"),
    )

    validations = []
    validations.append(f"timestamp={datetime.now().isoformat(timespec='seconds')}")
    validations.append(f"models_trained={len(runs_df)}")
    validations.append(f"runs_per_gamma={runs_df.groupby('gamma').size().to_dict()}")
    validations.append(f"noise_vectors_reused_within_seed_across_gammas=True; fingerprints={noise_fingerprints}")
    validations.append(
        "temporal_overlap="
        f"train_max={metadata['split_dates']['train_max']}, "
        f"val_min={metadata['split_dates']['val_min']}, "
        f"val_max={metadata['split_dates']['val_max']}, "
        f"test_min={metadata['split_dates']['test_min']}"
    )
    validations.append(
        "no_temporal_overlap="
        f"{metadata['split_dates']['train_max'] < metadata['split_dates']['val_min'] < metadata['split_dates']['val_max'] < metadata['split_dates']['test_min']}"
    )
    validations.append(f"final_sizes=train:{metadata['train_size']}, validation:{metadata['val_size']}, test:{metadata['test_size']}")
    validations.append(f"base_scaler_fit_only_on_training=True")
    validations.append(f"vix_standardizer_fit_only_on_training=True; mu={metadata['vix_mu_train']:.10f}; sigma={metadata['vix_sigma_train']:.10f}")
    validations.extend(array_checks)
    validations.append(f"shap_full_test_rows={len(test)}; no_subsampling=True")
    validations.append(f"shap_normalized_sum_min={runs_df['SHAP_Normalized_Sum'].min():.12f}; max={runs_df['SHAP_Normalized_Sum'].max():.12f}")
    validations.append(f"rank_min={runs_df['SHAP_VIX_Change_Std_Rank'].min()}; rank_max={runs_df['SHAP_VIX_Change_Std_Rank'].max()}")
    validations.append(f"nan_count_runs={int(runs_df.isna().sum().sum())}")
    validations.append(f"inf_count_runs={int(np.isinf(runs_df.select_dtypes(include=[np.number]).values).sum())}")
    validations.append(f"monotonic_R2_mean={bool(np.all(np.diff(agg['R2_mean'].values) >= 0))}")
    validations.append(f"monotonic_DirAcc_mean={bool(np.all(np.diff(agg['DirAcc_mean'].values) >= 0))}")
    validations.append(f"monotonic_SHAP_mean={bool(np.all(np.diff(agg['SHAP_mean'].values) >= 0))}")

    return runs_df, agg, "\n".join(validations)


def make_latex_table(summary: pd.DataFrame) -> str:
    rows = []
    for _, row in summary.iterrows():
        rows.append(
            "\t\t"
            f"{row['gamma']:.3f} &\n"
            "\t\t"
            f"\\({row['R2_mean']:.4f}\\pm{row['R2_sd']:.4f}\\) &\n"
            "\t\t"
            f"\\({row['DirAcc_mean'] * 100:.2f}\\%\\pm{row['DirAcc_sd'] * 100:.2f}\\%\\) &\n"
            "\t\t"
            f"\\({row['SHAP_mean'] * 100:.2f}\\%\\pm{row['SHAP_sd'] * 100:.2f}\\%\\) &\n"
            "\t\t"
            f"{row['Rank_mean']:.1f} of 19 \\\\\n"
        )

    body = "\n\t\t\n".join(rows)
    return rf"""\begin{{table}}[h]
	\caption{{Recovery of the controlled dependency \(\mathrm{{VIXchg}}_t \rightarrow y_{{t+1}}\) across 10 paired seeded repetitions per signal-intensity level using 19 predictors.}}
	\label{{tab:signal_recovery}}
	\setlength{{\tabcolsep}}{{3pt}}
	\centering
	\begin{{tabular}}{{p{{40pt}}p{{100pt}}p{{110pt}}p{{100pt}}p{{80pt}}}}
		\hline
		\(\boldsymbol{{\gamma}}\) &
		\(\boldsymbol{{R^2}}\) \textbf{{(mean \(\boldsymbol{{\pm}}\) SD)}} &
		\textbf{{Directional accuracy (mean \(\boldsymbol{{\pm}}\) SD)}} &
		\textbf{{Normalized SHAP importance (\%)}} &
		\textbf{{Mean SHAP rank}} \\
		\hline

{body}
		\hline
	\end{{tabular}}
	\footnotesize
\end{{table}}
"""


def make_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in summary.iterrows():
        prev = PREVIOUS_RESULTS[float(row["gamma"])]
        rows.append({
            "gamma": row["gamma"],
            "R2_mean_previous": prev["R2_mean"],
            "R2_mean_new": row["R2_mean"],
            "R2_mean_delta": row["R2_mean"] - prev["R2_mean"],
            "DirAcc_previous": prev["DirAcc_mean"],
            "DirAcc_new": row["DirAcc_mean"],
            "DirAcc_delta": row["DirAcc_mean"] - prev["DirAcc_mean"],
            "SHAP_previous": prev["SHAP_mean"],
            "SHAP_new": row["SHAP_mean"],
            "SHAP_delta": row["SHAP_mean"] - prev["SHAP_mean"],
            "Rank_previous": prev["Rank_mean"],
            "Rank_new": row["Rank_mean"],
            "Rank_delta": row["Rank_mean"] - prev["Rank_mean"],
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runs, summary, validation_report = run_experiment()
    comparison = make_comparison(summary)

    runs.to_csv(OUTPUT_DIR / "signal_recovery_results_paired_runs.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "signal_recovery_results_paired_summary.csv", index=False)
    comparison.to_csv(OUTPUT_DIR / "signal_recovery_results_paired_comparison.csv", index=False)
    (OUTPUT_DIR / "signal_recovery_validation_report.txt").write_text(
        validation_report
        + "\n\naggregated_results:\n"
        + summary.to_string(index=False)
        + "\n\ncomparison_with_previous_table:\n"
        + comparison.to_string(index=False),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "signal_recovery_table_latex.tex").write_text(make_latex_table(summary), encoding="utf-8")
    print(summary.to_string(index=False))
    print("\nWrote outputs to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
