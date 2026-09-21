"""
Análisis - SHAP, Factores Económicos, Explicabilidad

Todas las tablas/valores de esta sección se calculan directamente sobre los datos
y el modelo XGBoost entrenado (05_Modelado/xgb_model.json). No contiene valores
fabricados ni simulados.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import warnings

import xgboost as xgb
import shap
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), "05_Modelado")
DATA_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), "04_Preprocessing")
RAW_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), "02_Descarga_Datos")
LOG_FILE = os.path.join(OUTPUT_DIR, "analysis_log.txt")

FEATURE_COLS = [
    'SMA_20', 'SMA_50', 'Vol_20',
    'Close_Lag1', 'Close_Lag5', 'Return_Lag1', 'Return_Lag5',
    'Open_Lag1', 'High_Lag1', 'Low_Lag1', 'Volume_Lag1',
    'DCOILWTICO', 'VIXCLS', 'DHHNGSP', 'UNRATE', 'CPIAUCSL', 'FEDFUNDS', 'SP500',
]
TARGET_COL = 'Target_Return'

TECH_TICKERS = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'TSLA']
ENERGY_TICKERS = ['CVX', 'COP', 'XLE', 'MPC']


def log_msg(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    log_msg("=" * 70)
    log_msg("ANÁLISIS - SHAP, FACTORES ECONÓMICOS, EXPLICABILIDAD")
    log_msg(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_msg("=" * 70)

    # 1. CARGAR DATOS
    log_msg("\n1. CARGANDO DATOS...")

    train = pd.read_csv(os.path.join(DATA_DIR, "train_set.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test_set.csv"))
    predictions = pd.read_csv(os.path.join(MODEL_DIR, "predictions.csv"))

    log_msg(f"✓ Train: {len(train)} registros")
    log_msg(f"✓ Test: {len(test)} registros")
    log_msg(f"✓ Predicciones: {len(predictions)} registros")

    # 2. FEATURE IMPORTANCE REAL (desde el modelo XGBoost entrenado)
    log_msg("\n2. FEATURE IMPORTANCE (XGBoost real)...")

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(os.path.join(MODEL_DIR, "xgb_model.json"))

    raw_importances = xgb_model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': FEATURE_COLS,
        'Importance': raw_importances / raw_importances.sum()
    }).sort_values('Importance', ascending=False).reset_index(drop=True)

    log_msg("\nTop 5 Features (XGBoost feature_importances_, normalizado a 100%):")
    for _, row in feature_importance.head(5).iterrows():
        log_msg(f"  {row['Feature']}: {row['Importance']:.2%}")

    feature_importance.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
    log_msg("\n✓ feature_importance.csv")

    # 3. ANÁLISIS DE ERRORES (ya calculado sobre predicciones reales, sin cambios)
    log_msg("\n3. ANÁLISIS DE ERRORES...")

    lstm_errors = np.abs(predictions['y_test'] - predictions['LSTM'])
    transformer_errors = np.abs(predictions['y_test'] - predictions['Transformer'])
    xgb_errors = np.abs(predictions['y_test'] - predictions['XGBoost'])
    ens_errors = np.abs(predictions['y_test'] - predictions['Ensemble'])

    error_analysis = pd.DataFrame({
        'Metric': ['Mean Error', 'Std Error', 'Max Error', 'Min Error', 'Q1', 'Median', 'Q3'],
        'LSTM': [lstm_errors.mean(), lstm_errors.std(), lstm_errors.max(), lstm_errors.min(),
                 lstm_errors.quantile(0.25), lstm_errors.median(), lstm_errors.quantile(0.75)],
        'Transformer': [transformer_errors.mean(), transformer_errors.std(), transformer_errors.max(), transformer_errors.min(),
                         transformer_errors.quantile(0.25), transformer_errors.median(), transformer_errors.quantile(0.75)],
        'XGBoost': [xgb_errors.mean(), xgb_errors.std(), xgb_errors.max(), xgb_errors.min(),
                    xgb_errors.quantile(0.25), xgb_errors.median(), xgb_errors.quantile(0.75)],
        'Ensemble': [ens_errors.mean(), ens_errors.std(), ens_errors.max(), ens_errors.min(),
                     ens_errors.quantile(0.25), ens_errors.median(), ens_errors.quantile(0.75)]
    })

    log_msg("\n" + error_analysis.to_string())
    error_analysis.to_csv(os.path.join(OUTPUT_DIR, "error_analysis.csv"), index=False)
    log_msg("\n✓ error_analysis.csv")

    # 4. FACTORES ECONÓMICOS (correlación real retorno sectorial vs. factor FRED)
    log_msg("\n4. FACTORES ECONÓMICOS (correlaciones reales)...")

    tech_raw = pd.read_csv(os.path.join(RAW_DIR, "tech_prices_raw.csv"))
    energy_raw = pd.read_csv(os.path.join(RAW_DIR, "energy_prices_raw.csv"))
    fred_raw = pd.read_csv(os.path.join(RAW_DIR, "fred_economic_raw.csv"))
    for df in (tech_raw, energy_raw, fred_raw):
        df['Date'] = pd.to_datetime(df['Date'])

    tech_raw['Return'] = tech_raw.groupby('Ticker')['Close'].pct_change()
    energy_raw['Return'] = energy_raw.groupby('Ticker')['Close'].pct_change()

    tech_sector_ret = tech_raw.groupby('Date')['Return'].mean()
    energy_sector_ret = energy_raw.groupby('Date')['Return'].mean()

    fred_idx = fred_raw.set_index('Date')
    factor_changes = {
        'Oil Price (DCOILWTICO)': fred_idx['DCOILWTICO'].pct_change(),
        'VIX Index': fred_idx['VIXCLS'].pct_change(),
        'Unemployment': fred_idx['UNRATE'].pct_change(),
        'Fed Rate': fred_idx['FEDFUNDS'].pct_change(),
        'S&P 500': fred_idx['SP500'].pct_change(),
    }

    rows = []
    for factor_name, series in factor_changes.items():
        corr_tech = tech_sector_ret.corr(series)
        corr_energy = energy_sector_ret.corr(series)
        rows.append({
            'Factor': factor_name,
            'Impact_on_Tech': round(corr_tech, 3),
            'Impact_on_Energy': round(corr_energy, 3),
        })

    economic_impact = pd.DataFrame(rows)
    log_msg("\nCorrelaciones reales (retorno diario promedio del sector vs. cambio diario del factor):")
    log_msg("\n" + economic_impact.to_string())

    economic_impact.to_csv(os.path.join(OUTPUT_DIR, "economic_impact.csv"), index=False)
    log_msg("\n✓ economic_impact.csv")

    # 5. SHAP REAL (TreeExplainer sobre el XGBoost entrenado, muestra de test)
    log_msg("\n5. EXPLICABILIDAD (SHAP real, TreeExplainer sobre XGBoost)...")

    train_feat = train.sort_values(['Ticker', 'Date']).dropna(subset=FEATURE_COLS + [TARGET_COL])
    test_feat = test.sort_values(['Ticker', 'Date']).dropna(subset=FEATURE_COLS + [TARGET_COL])

    scaler = MinMaxScaler()
    scaler.fit(train_feat[FEATURE_COLS].values)
    X_test_scaled = scaler.transform(test_feat[FEATURE_COLS].values)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test_scaled)

    shap_analysis = pd.DataFrame({
        'Feature': FEATURE_COLS,
        'Mean_Abs_SHAP': np.abs(shap_values).mean(axis=0),
        'SHAP_Std': shap_values.std(axis=0),
    }).sort_values('Mean_Abs_SHAP', ascending=False).reset_index(drop=True)

    log_msg("\nSHAP Values reales (Top 5, |SHAP| medio sobre el test set):")
    for _, row in shap_analysis.head(5).iterrows():
        log_msg(f"  {row['Feature']}: {row['Mean_Abs_SHAP']:.4f} (std={row['SHAP_Std']:.4f})")

    shap_analysis.to_csv(os.path.join(OUTPUT_DIR, "shap_analysis.csv"), index=False)
    log_msg("\n✓ shap_analysis.csv")

    # Monotonicidad real: correlación entre el valor de Close_Lag1 y su SHAP value
    close_lag1_idx = FEATURE_COLS.index('Close_Lag1')
    close_lag1_vals = X_test_scaled[:, close_lag1_idx]
    close_lag1_shap = shap_values[:, close_lag1_idx]
    monotonicity_corr = np.corrcoef(close_lag1_vals, close_lag1_shap)[0, 1]
    log_msg(f"\n✓ Correlación Close_Lag1 (valor de feature) vs. su SHAP value: {monotonicity_corr:.4f}")

    # 6. REPORTE
    log_msg("\n6. GENERANDO REPORTE...")

    top_econ_tech = economic_impact.reindex(economic_impact['Impact_on_Tech'].abs().sort_values(ascending=False).index).iloc[0]
    top_econ_energy = economic_impact.reindex(economic_impact['Impact_on_Energy'].abs().sort_values(ascending=False).index).iloc[0]

    insights = f"""# Análisis Detallado - Explicabilidad e Impacto (valores reales, no simulados)

**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. FEATURE IMPORTANCE (XGBoost `.feature_importances_`, real)

Top 5 features:
{chr(10).join(f"{i+1}. **{r.Feature}** ({r.Importance:.1%})" for i, r in feature_importance.head(5).iterrows())}

## 2. ANÁLISIS DE ERRORES (sobre predicciones reales del test set)

- LSTM: media={lstm_errors.mean():.2f}, std={lstm_errors.std():.2f}, max={lstm_errors.max():.2f}
- Transformer: media={transformer_errors.mean():.2f}, std={transformer_errors.std():.2f}, max={transformer_errors.max():.2f}
- XGBoost: media={xgb_errors.mean():.2f}, std={xgb_errors.std():.2f}, max={xgb_errors.max():.2f}
- Ensemble: media={ens_errors.mean():.2f}, std={ens_errors.std():.2f}, max={ens_errors.max():.2f}

## 3. FACTORES ECONÓMICOS (correlación real retorno sectorial vs. cambio diario del factor)

{economic_impact.to_string(index=False)}

Factor con mayor |correlación| para Tech: **{top_econ_tech.Factor}** ({top_econ_tech.Impact_on_Tech:+.3f})
Factor con mayor |correlación| para Energía: **{top_econ_energy.Factor}** ({top_econ_energy.Impact_on_Energy:+.3f})

## 4. EXPLICABILIDAD (SHAP real vía TreeExplainer sobre XGBoost)

{shap_analysis.to_string(index=False)}

Correlación entre valor de Close_Lag1 y su SHAP value: {monotonicity_corr:.4f}
(un valor positivo confirma monotonicidad: a mayor Close_Lag1, mayor contribución SHAP a la predicción)

---

**Status:** Análisis completado con valores calculados directamente sobre el modelo y los datos.
"""

    with open(os.path.join(OUTPUT_DIR, "ANALYSIS_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(insights)

    log_msg("\n✓ ANALYSIS_REPORT.md")

    log_msg("\n" + "=" * 70)
    log_msg("✓ ANÁLISIS COMPLETADO")
    log_msg("=" * 70)


if __name__ == "__main__":
    main()
