from pathlib import Path

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler


PROJECT = Path(r"C:\Users\wilop\Dropbox\MPDI\2026\AI in Business")
DATA_DIR = PROJECT / "Auditoria" / "04_Preprocessing"
MODEL_DIR = PROJECT / "Auditoria" / "05_Modelado"
OUT_DIR = PROJECT / "Auditoria" / "07_Analisis"

FEATURE_COLS = [
    "SMA_20", "SMA_50", "Vol_20",
    "Close_Lag1", "Close_Lag5", "Return_Lag1", "Return_Lag5",
    "Open_Lag1", "High_Lag1", "Low_Lag1", "Volume_Lag1",
    "DCOILWTICO", "VIXCLS", "DHHNGSP", "UNRATE", "CPIAUCSL", "FEDFUNDS", "SP500",
]
TARGET_COL = "Target_Return"

CSV_OUT = OUT_DIR / "main_xgb_shap_18_predictor_ranking.csv"
REPORT_OUT = OUT_DIR / "main_xgb_shap_validation_report.txt"
LATEX_OUT = OUT_DIR / "main_xgb_shap_top10_table.tex"


def load_complete_case():
    train = pd.read_csv(DATA_DIR / "train_set.csv")
    test = pd.read_csv(DATA_DIR / "test_set.csv")
    for df in (train, test):
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values(["Ticker", "Date"], inplace=True)
        df.dropna(subset=FEATURE_COLS + [TARGET_COL], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return train, test


def latex_escape(text):
    return (
        text.replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
    )


def make_latex_table(ranking):
    top10 = ranking.head(10)
    remaining = ranking.iloc[10:]
    remaining_row = {
        "Rank": "11--18",
        "Predictor": "Remaining eight predictors",
        "Mean_Abs_SHAP": remaining["Mean_Abs_SHAP"].sum(),
        "Normalized_SHAP_Importance_Percent": remaining["Normalized_SHAP_Importance_Percent"].sum(),
    }

    rows = []
    for _, row in top10.iterrows():
        rows.append(
            "\t\t"
            f"{int(row['Rank'])} & "
            f"{latex_escape(row['Predictor'])} & "
            f"{row['Mean_Abs_SHAP']:.6f} & "
            f"{row['Normalized_SHAP_Importance_Percent']:.2f}\\% \\\\"
        )
    rows.append(
        "\t\t"
        f"{remaining_row['Rank']} & "
        f"{remaining_row['Predictor']} & "
        f"{remaining_row['Mean_Abs_SHAP']:.6f} & "
        f"{remaining_row['Normalized_SHAP_Importance_Percent']:.2f}\\% \\\\"
    )

    return "\n".join([
        r"\begin{table}[h]",
        r"	\caption{Main leakage-free XGBoost SHAP ranking computed on the complete-case tabular test sample (\(N=1{,}746\)) using the original 18 predictors.}",
        r"	\label{tab:main_xgb_shap}",
        r"	\setlength{\tabcolsep}{4pt}",
        r"	\centering",
        r"	\begin{tabular}{p{35pt}p{125pt}p{95pt}p{105pt}}",
        r"		\hline",
        r"		\textbf{Rank} & \textbf{Predictor} & \textbf{Mean \(|\mathrm{SHAP}|\)} & \textbf{Normalized SHAP importance} \\",
        r"		\hline",
        *rows,
        r"		\hline",
        r"	\end{tabular}",
        r"	\footnotesize",
        r"\end{table}",
        "",
    ])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, test = load_complete_case()

    if len(FEATURE_COLS) != 18:
        raise ValueError(f"Expected 18 predictors, got {len(FEATURE_COLS)}")
    if len(test) != 1746:
        raise ValueError(f"Expected complete-case test N=1746, got {len(test)}")

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train[FEATURE_COLS].values)
    X_test = scaler.transform(test[FEATURE_COLS].values)

    if X_test.shape[1] != 18:
        raise ValueError(f"Expected X_test with 18 predictors, got {X_test.shape[1]}")
    if not np.isfinite(X_test).all():
        raise ValueError("X_test contains NaN or infinite values.")

    model = xgb.XGBRegressor()
    model.load_model(MODEL_DIR / "xgb_model.json")

    # SHAP-only pathway: do not read or use xgboost feature_importances_.
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap_values = np.asarray(shap_values)
    if shap_values.shape != X_test.shape:
        raise ValueError(f"Unexpected SHAP shape {shap_values.shape}, expected {X_test.shape}")

    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_sum = mean_abs.sum()
    if shap_sum <= 0 or not np.isfinite(shap_sum):
        raise ValueError("Invalid SHAP denominator.")

    pct = 100 * mean_abs / shap_sum
    ranking = pd.DataFrame({
        "Rank": np.arange(1, len(FEATURE_COLS) + 1),
        "Predictor": FEATURE_COLS,
        "Mean_Abs_SHAP": mean_abs,
        "Normalized_SHAP_Importance_Percent": pct,
    }).sort_values("Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    ranking["Rank"] = np.arange(1, len(ranking) + 1)

    pct_sum = ranking["Normalized_SHAP_Importance_Percent"].sum()
    if not np.isclose(pct_sum, 100.0, atol=1e-8):
        raise ValueError(f"Normalized SHAP percentages do not sum to 100: {pct_sum}")

    ranking.to_csv(CSV_OUT, index=False)
    LATEX_OUT.write_text(make_latex_table(ranking), encoding="utf-8")

    declared_model_config = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "reg:squarederror",
        "random_state": 42,
        "early_stopping_rounds": 10,
    }
    report = "\n".join([
        "Main leakage-free XGBoost SHAP validation report",
        f"train_complete_case_N={len(train)}",
        f"test_complete_case_N={len(test)}",
        f"predictor_count={len(FEATURE_COLS)}",
        f"predictors={FEATURE_COLS}",
        f"X_test_shape={X_test.shape}",
        f"shap_values_shape={shap_values.shape}",
        f"explainer_type={type(explainer).__name__}",
        "model_source=05_Modelado/xgb_model.json",
        f"declared_model_configuration={declared_model_config}",
        "target=next-day logarithmic return (Target_Return)",
        "information_set=predictors available through day t",
        "predictor_space=original 18 predictors",
        "signal_injection_predictor_included=False",
        "VIX_Change_Std_included=False",
        "xgboost_feature_importances_used=False",
        "shap_subsampling_used=False",
        f"mean_abs_shap_sum={shap_sum:.12f}",
        f"normalized_shap_percent_sum={pct_sum:.12f}",
        f"top_predictor={ranking.iloc[0]['Predictor']}",
        f"top_predictor_percent={ranking.iloc[0]['Normalized_SHAP_Importance_Percent']:.6f}",
    ])
    REPORT_OUT.write_text(report, encoding="utf-8")

    print(ranking.to_string(index=False))
    print(CSV_OUT)
    print(REPORT_OUT)
    print(LATEX_OUT)


if __name__ == "__main__":
    main()
