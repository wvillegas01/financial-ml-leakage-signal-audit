"""
Evaluación - Análisis por sector y pruebas estadísticas formales

Todos los valores de este script se calculan directamente sobre metrics.csv y
predictions.csv (05_Modelado/). No contiene valores fabricados, simulados ni
hardcodeados.
"""

import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime
import os

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), "05_Modelado")
LOG_FILE = os.path.join(OUTPUT_DIR, "evaluation_log.txt")

MODELS = ['LSTM', 'Transformer', 'XGBoost', 'Ensemble']


def log_msg(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def wilson_ci(k, n, confidence=0.95):
    """Intervalo de confianza de Wilson para una proporción (más estable que el
    intervalo normal cuando la proporción está cerca de 0.5 y n es moderado)."""
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return center - margin, center + margin


def directional_stats(y_true, y_pred, model_name):
    """Precisión direccional + prueba binomial formal contra dos referencias:
    (a) azar puro (p=0.5) y (b) la clase mayoritaria observada en y_true (predecir
    siempre el signo más frecuente). Esta segunda referencia es la que de verdad
    hay que superar: si el 51% de los días son positivos, un modelo que siempre
    predice "positivo" ya acierta 51% sin usar ninguna información."""
    n = len(y_true)
    correct = int(np.sum(np.sign(y_pred) == np.sign(y_true)))
    dir_acc = correct / n

    pos_rate = float(np.mean(y_true > 0))
    majority_baseline = max(pos_rate, 1 - pos_rate)

    # Prueba binomial: ¿dir_acc es distinguible de 0.5?
    test_vs_chance = stats.binomtest(correct, n, p=0.5, alternative='two-sided')
    # ¿dir_acc es distinguible de la línea base de clase mayoritaria?
    test_vs_majority = stats.binomtest(correct, n, p=majority_baseline, alternative='two-sided')

    ci_low, ci_high = wilson_ci(correct, n)

    return {
        'Model': model_name,
        'N': n,
        'Directional_Accuracy': round(dir_acc, 4),
        'CI_95_Low': round(ci_low, 4),
        'CI_95_High': round(ci_high, 4),
        'Majority_Class_Baseline': round(majority_baseline, 4),
        'p_value_vs_chance_0.5': round(test_vs_chance.pvalue, 4),
        'p_value_vs_majority_baseline': round(test_vs_majority.pvalue, 4),
        'Significant_vs_chance_a0.05': bool(test_vs_chance.pvalue < 0.05),
        'Significant_vs_majority_a0.05': bool(test_vs_majority.pvalue < 0.05),
    }


def sector_metrics(df, sector, model_name):
    sub = df[df['Sector'] == sector]
    y_true = sub['y_test'].values
    y_pred = sub[model_name].values
    n = len(sub)
    if n == 0:
        return None
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    dir_acc = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    return {
        'Sector': sector, 'Model': model_name, 'N': n,
        'MAE': round(mae, 4), 'RMSE': round(rmse, 4), 'R2': round(r2, 4),
        'Directional_Accuracy': round(dir_acc, 4),
    }


def main():
    log_msg("=" * 70)
    log_msg("EVALUACIÓN - ANÁLISIS POR SECTOR Y PRUEBAS ESTADÍSTICAS")
    log_msg(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_msg("=" * 70)

    # 1. CARGAR RESULTADOS
    log_msg("\n1. CARGANDO RESULTADOS...")

    metrics = pd.read_csv(os.path.join(MODEL_DIR, "metrics.csv"))
    predictions = pd.read_csv(os.path.join(MODEL_DIR, "predictions.csv"))

    log_msg(f"✓ Métricas: {len(metrics)} modelos")
    log_msg(f"✓ Predicciones: {len(predictions)} registros")
    log_msg(f"✓ Sectores presentes: {predictions['Sector'].unique().tolist()}")

    # 2. PRUEBAS ESTADÍSTICAS FORMALES SOBRE PRECISIÓN DIRECCIONAL
    log_msg("\n2. PRUEBA BINOMIAL: ¿la precisión direccional es distinguible del azar "
            "y de la línea base de clase mayoritaria?")

    y_true_all = predictions['y_test'].values
    stat_rows = []
    for model in MODELS:
        if model in predictions.columns:
            row = directional_stats(y_true_all, predictions[model].values, model)
            stat_rows.append(row)
            log_msg(f"  {model}: DirAcc={row['Directional_Accuracy']:.4f} "
                     f"(IC95%=[{row['CI_95_Low']:.4f}, {row['CI_95_High']:.4f}]), "
                     f"baseline mayoritaria={row['Majority_Class_Baseline']:.4f}, "
                     f"p(vs azar)={row['p_value_vs_chance_0.5']:.4f}, "
                     f"p(vs mayoría)={row['p_value_vs_majority_baseline']:.4f}")

    directional_tests = pd.DataFrame(stat_rows)
    directional_tests.to_csv(os.path.join(OUTPUT_DIR, "directional_significance_tests.csv"), index=False)
    log_msg("\n✓ directional_significance_tests.csv")

    # 3. ANÁLISIS POR SECTOR (real, usando el Ticker/Sector alineado en predictions.csv)
    log_msg("\n3. ANÁLISIS POR SECTOR (métricas reales, no promedios repetidos)...")

    sector_rows = []
    for sector in ['Tecnología', 'Energía']:
        for model in MODELS:
            if model in predictions.columns:
                r = sector_metrics(predictions, sector, model)
                if r is not None:
                    sector_rows.append(r)
                    log_msg(f"  {sector} / {model}: N={r['N']}, MAE={r['MAE']:.4f}, "
                             f"RMSE={r['RMSE']:.4f}, R2={r['R2']:.4f}, DirAcc={r['Directional_Accuracy']:.4f}")

    sector_comparison = pd.DataFrame(sector_rows)
    sector_comparison.to_csv(os.path.join(OUTPUT_DIR, "sector_comparison.csv"), index=False)
    log_msg("\n✓ sector_comparison.csv")

    # 4. REPORTE
    log_msg("\n4. GENERANDO REPORTE...")

    metrics_by_r2 = metrics.sort_values('R2', ascending=False).reset_index(drop=True)
    best_r2 = metrics_by_r2.iloc[0]
    metrics_by_dir = metrics.sort_values('Directional_Accuracy', ascending=False).reset_index(drop=True)
    best_dir = metrics_by_dir.iloc[0]
    worst_r2 = metrics_by_r2.iloc[-1]

    n_sig_chance = int(directional_tests['Significant_vs_chance_a0.05'].sum())
    n_sig_majority = int(directional_tests['Significant_vs_majority_a0.05'].sum())

    reporte = f"""# Evaluación - Análisis Comparativo

**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. RESULTADOS GLOBALES

```
{metrics.to_string()}
```

## 2. PRUEBAS ESTADÍSTICAS FORMALES (precisión direccional)

```
{directional_tests.to_string(index=False)}
```

De los {len(directional_tests)} modelos evaluados, {n_sig_chance} muestran una precisión
direccional estadísticamente distinguible del azar (p<0.05, prueba binomial de dos colas
contra p=0.5), y {n_sig_majority} son distinguibles de la línea base de clase mayoritaria
observada en el conjunto de prueba.

## 3. ANÁLISIS POR SECTOR (real)

```
{sector_comparison.to_string(index=False)}
```

## 4. CONCLUSIONES

### Mejor R²: {best_r2['Model']}
- **R² = {best_r2['R2']:.4f}**
- **MAE = {best_r2['MAE']:.4f}**
- **Dir Acc = {best_r2['Directional_Accuracy']*100:.2f}%**

### Mejor Directional Accuracy: {best_dir['Model']}
- **Dir Acc = {best_dir['Directional_Accuracy']*100:.2f}%**

### Menor R²: {worst_r2['Model']}
- R² = {worst_r2['R2']:.4f}
- MAE = {worst_r2['MAE']:.4f}

---

**Status:** Evaluación completada. Todos los valores (globales, por sector, pruebas
estadísticas) se calculan dinámicamente desde metrics.csv/predictions.csv.
**Próximo:** Análisis SHAP en 07_Análisis/
"""

    with open(os.path.join(OUTPUT_DIR, "EVALUATION_REPORT.md"), "w", encoding="utf-8") as f:
        f.write(reporte)

    log_msg("✓ EVALUATION_REPORT.md")

    log_msg("\n" + "=" * 70)
    log_msg("✓ EVALUACIÓN COMPLETADA")
    log_msg("=" * 70)


if __name__ == "__main__":
    main()
