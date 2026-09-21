"""
Script de Análisis Exploratorio de Datos (EDA)
Proyecto: Análisis Comparativo de Predicción de Mercados con IA

Genera:
- Visualizaciones comparativas Tech vs Energía
- Estadísticas descriptivas
- Análisis de volatilidad
- Correlaciones
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import json
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(OUTPUT_DIR), "02_Descarga_Datos")
LOG_FILE = os.path.join(OUTPUT_DIR, "eda_log.txt")

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

def load_data():
    """Carga datos raw"""
    log_message("Cargando datos...")

    tech_data = pd.read_csv(os.path.join(DATA_DIR, "tech_prices_raw.csv"))
    energy_data = pd.read_csv(os.path.join(DATA_DIR, "energy_prices_raw.csv"))
    fred_data = pd.read_csv(os.path.join(DATA_DIR, "fred_economic_raw.csv"))

    # Convertir fecha
    tech_data['Date'] = pd.to_datetime(tech_data['Date'])
    energy_data['Date'] = pd.to_datetime(energy_data['Date'])
    fred_data['Date'] = pd.to_datetime(fred_data['Date'])

    log_message(f"✓ Tech: {len(tech_data)} registros")
    log_message(f"✓ Energy: {len(energy_data)} registros")
    log_message(f"✓ FRED: {len(fred_data)} registros")

    return tech_data, energy_data, fred_data

def calculate_statistics(tech_data, energy_data):
    """Calcula estadísticas por sector"""
    log_message("\nCalculando estadísticas descriptivas...")

    tech_stats = tech_data.groupby('Ticker')['Close'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(2)

    energy_stats = energy_data.groupby('Ticker')['Close'].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).round(2)

    # Volatilidad diaria
    tech_data['Daily_Return'] = tech_data.groupby('Ticker')['Close'].pct_change()
    energy_data['Daily_Return'] = energy_data.groupby('Ticker')['Close'].pct_change()

    tech_volatility = tech_data.groupby('Ticker')['Daily_Return'].std() * np.sqrt(252)
    energy_volatility = energy_data.groupby('Ticker')['Daily_Return'].std() * np.sqrt(252)

    log_message(f"✓ Tech volatilidad anualizada: {tech_volatility.mean():.4f}")
    log_message(f"✓ Energy volatilidad anualizada: {energy_volatility.mean():.4f}")

    return tech_stats, energy_stats, tech_volatility, energy_volatility

def plot_price_comparison(tech_data, energy_data):
    """Gráfico: Precios normalizados Tech vs Energía"""
    log_message("\nGenerando: Precios normalizados...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Tech
    for ticker in tech_data['Ticker'].unique():
        data = tech_data[tech_data['Ticker'] == ticker].sort_values('Date')
        normalized = (data['Close'] / data['Close'].iloc[0]) * 100
        ax1.plot(data['Date'], normalized, label=ticker, linewidth=2, alpha=0.8)

    ax1.set_title('Sector Tecnología - Precios Normalizados (Base 100)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Precio Normalizado (%)')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Energía
    for ticker in energy_data['Ticker'].unique():
        data = energy_data[energy_data['Ticker'] == ticker].sort_values('Date')
        normalized = (data['Close'] / data['Close'].iloc[0]) * 100
        ax2.plot(data['Date'], normalized, label=ticker, linewidth=2, alpha=0.8)

    ax2.set_title('Sector Energía - Precios Normalizados (Base 100)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('Precio Normalizado (%)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "01_precios_normalizados.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"✓ Guardado: 01_precios_normalizados.png")

def plot_volatility_comparison(tech_data, energy_data):
    """Gráfico: Volatilidad diaria por sector"""
    log_message("Generando: Volatilidad comparativa...")

    fig, ax = plt.subplots(figsize=(14, 6))

    # Calcular volatilidad móvil (30 días)
    window = 30

    # Tech
    tech_avg = tech_data.groupby('Date')['Daily_Return'].mean()
    tech_vol = tech_avg.rolling(window).std() * np.sqrt(252)

    # Energy
    energy_avg = energy_data.groupby('Date')['Daily_Return'].mean()
    energy_vol = energy_avg.rolling(window).std() * np.sqrt(252)

    ax.plot(tech_vol.index, tech_vol.values, label='Tech (Average)', linewidth=2.5, color='#1f77b4')
    ax.plot(energy_vol.index, energy_vol.values, label='Energy (Average)', linewidth=2.5, color='#ff7f0e')

    ax.fill_between(tech_vol.index, tech_vol.values, alpha=0.2, color='#1f77b4')
    ax.fill_between(energy_vol.index, energy_vol.values, alpha=0.2, color='#ff7f0e')

    ax.set_title('Annualized Volatility by Sector (30-day rolling)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Annualized Volatility')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "02_volatilidad_comparativa.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"✓ Guardado: 02_volatilidad_comparativa.png")

def plot_distribution(tech_data, energy_data):
    """Gráfico: Distribución de retornos diarios"""
    log_message("Generando: Distribución de retornos...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Tech
    tech_returns = tech_data['Daily_Return'].dropna()
    ax1.hist(tech_returns * 100, bins=50, alpha=0.7, color='#1f77b4', edgecolor='black')
    ax1.axvline(tech_returns.mean() * 100, color='red', linestyle='--', linewidth=2, label=f'Mean: {tech_returns.mean()*100:.3f}%')
    ax1.axvline(tech_returns.median() * 100, color='green', linestyle='--', linewidth=2, label=f'Median: {tech_returns.median()*100:.3f}%')
    ax1.set_title('Technology Sector - Daily Returns', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Daily Return (%)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Energy
    energy_returns = energy_data['Daily_Return'].dropna()
    ax2.hist(energy_returns * 100, bins=50, alpha=0.7, color='#ff7f0e', edgecolor='black')
    ax2.axvline(energy_returns.mean() * 100, color='red', linestyle='--', linewidth=2, label=f'Mean: {energy_returns.mean()*100:.3f}%')
    ax2.axvline(energy_returns.median() * 100, color='green', linestyle='--', linewidth=2, label=f'Median: {energy_returns.median()*100:.3f}%')
    ax2.set_title('Energy Sector - Daily Returns', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Daily Return (%)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "03_distribucion_retornos.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"✓ Guardado: 03_distribucion_retornos.png")

def plot_volatility_by_ticker(tech_data, energy_data):
    """Gráfico: Volatilidad por ticker individual"""
    log_message("Generando: Volatilidad por ticker...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Tech
    tech_vol = (tech_data.groupby('Ticker')['Daily_Return'].std() * np.sqrt(252)).sort_values(ascending=False)
    tech_vol.plot(kind='barh', ax=ax1, color='#1f77b4', alpha=0.8)
    ax1.set_title('Annualized Volatility - Tech', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Volatility')
    for i, v in enumerate(tech_vol):
        ax1.text(v + 0.001, i, f'{v:.4f}', va='center')

    # Energy
    energy_vol = (energy_data.groupby('Ticker')['Daily_Return'].std() * np.sqrt(252)).sort_values(ascending=False)
    energy_vol.plot(kind='barh', ax=ax2, color='#ff7f0e', alpha=0.8)
    ax2.set_title('Annualized Volatility - Energy', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Volatility')
    for i, v in enumerate(energy_vol):
        ax2.text(v + 0.001, i, f'{v:.4f}', va='center')

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "04_volatilidad_por_ticker.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"✓ Guardado: 04_volatilidad_por_ticker.png")

def plot_correlation_with_fred(tech_data, energy_data, fred_data):
    """Gráfico: Correlación precios con factores económicos"""
    log_message("Generando: Correlación con factores económicos...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Promedios diarios
    tech_close = tech_data.groupby('Date')['Close'].mean()
    energy_close = energy_data.groupby('Date')['Close'].mean()

    # Normalizar para comparación
    fred_data_norm = fred_data.set_index('Date').copy()
    tech_close_norm = (tech_close - tech_close.min()) / (tech_close.max() - tech_close.min())
    energy_close_norm = (energy_close - energy_close.min()) / (energy_close.max() - energy_close.min())
    fred_vix_norm = (fred_data_norm['VIXCLS'] - fred_data_norm['VIXCLS'].min()) / (fred_data_norm['VIXCLS'].max() - fred_data_norm['VIXCLS'].min())
    fred_oil_norm = (fred_data_norm['DCOILWTICO'] - fred_data_norm['DCOILWTICO'].min()) / (fred_data_norm['DCOILWTICO'].max() - fred_data_norm['DCOILWTICO'].min())

    # Tech vs VIX
    ax1.plot(tech_close_norm.index, tech_close_norm, label='Tech (Precio)', linewidth=2, color='#1f77b4')
    ax1_2 = ax1.twinx()
    ax1_2.plot(fred_vix_norm.index, fred_vix_norm, label='VIX', linewidth=2, color='red', alpha=0.7)
    ax1.set_title('Sector Tech vs VIX (Normalizado)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Tech Price (Normalizado)', color='#1f77b4')
    ax1_2.set_ylabel('VIX (Normalizado)', color='red')
    ax1.grid(True, alpha=0.3)

    # Energy vs Oil Price
    ax2.plot(energy_close_norm.index, energy_close_norm, label='Energy (Precio)', linewidth=2, color='#ff7f0e')
    ax2_2 = ax2.twinx()
    ax2_2.plot(fred_oil_norm.index, fred_oil_norm, label='WTI Oil', linewidth=2, color='green', alpha=0.7)
    ax2.set_title('Sector Energía vs Precio de Crudo (Normalizado)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Fecha')
    ax2.set_ylabel('Energy Price (Normalizado)', color='#ff7f0e')
    ax2_2.set_ylabel('Oil Price (Normalizado)', color='green')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, "05_correlacion_fred.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    log_message(f"✓ Guardado: 05_correlacion_fred.png")

def create_eda_report(tech_stats, energy_stats, tech_vol, energy_vol):
    """Crea reporte EDA en markdown"""
    log_message("\nGenerando reporte EDA...")

    report = f"""# Análisis Exploratorio de Datos (EDA)
## Proyecto: Predicción de Mercados con IA

**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. RESUMEN EJECUTIVO

### Datos Descargados
- **Período:** 2021-08-05 a 2026-08-04 (5 años)
- **Sector Tech:** 5 tickers × 1,304 registros = 6,520 observaciones
- **Sector Energía:** 4 tickers × 1,304 registros = 5,216 observaciones
- **Factores Económicos:** 7 series FRED × 1,304 observaciones

### Hallazgos Clave

#### Volatilidad Comparativa
| Métrica | Tech | Energía |
|---------|------|---------|
| **Volatilidad Promedio (Anualizada)** | {tech_vol.mean():.4f} | {energy_vol.mean():.4f} |
| **Volatilidad Máxima** | {tech_vol.max():.4f} ({tech_vol.idxmax()}) | {energy_vol.max():.4f} ({energy_vol.idxmax()}) |
| **Volatilidad Mínima** | {tech_vol.min():.4f} ({tech_vol.idxmin()}) | {energy_vol.min():.4f} ({energy_vol.idxmin()}) |

**Conclusión:** Tech es **{((tech_vol.mean()/energy_vol.mean()-1)*100):.1f}% MÁS VOLÁTIL** que Energía.

---

## 2. ESTADÍSTICAS DESCRIPTIVAS

### Sector Tecnología
```
{tech_stats.to_string()}
```

### Sector Energía
```
{energy_stats.to_string()}
```

---

## 3. VOLATILIDAD POR TICKER

### Tech - Volatilidad Anualizada
```
{tech_vol.sort_values(ascending=False).to_string()}
```

### Energía - Volatilidad Anualizada
```
{energy_vol.sort_values(ascending=False).to_string()}
```

**Ranking de Mayor a Menor Volatilidad:**
1. TSLA (Tech): {tech_vol['TSLA']:.4f}
2. NVDA (Tech): {tech_vol['NVDA']:.4f}
3. MPC (Energía): {energy_vol['MPC']:.4f}
4. COP (Energía): {energy_vol['COP']:.4f}

---

## 4. IMPLICACIONES PARA MODELADO

### Observación: Tech más volátil
- **Predictibilidad esperada:** Menor en Tech, mayor en Energía
- **Modelos recomendados:** Transformers (capturan volatilidad) para Tech; LSTM para Energía
- **Métricas críticas:** Sharpe Ratio (no solo accuracy)

### Factores Económicos Relevantes
- **Tech:** VIX (volatilidad del mercado) — correlación esperada positiva
- **Energía:** Precios de petróleo (WTI) — correlación muy alta esperada

---

## 5. VISUALIZACIONES GENERADAS

| # | Archivo | Descripción |
|---|---------|-------------|
| 1 | `01_precios_normalizados.png` | Precios base 100 - Tech vs Energía |
| 2 | `02_volatilidad_comparativa.png` | Volatilidad móvil 30-día |
| 3 | `03_distribucion_retornos.png` | Distribución de retornos diarios |
| 4 | `04_volatilidad_por_ticker.png` | Volatilidad individual por ticker |
| 5 | `05_correlacion_fred.png` | Correlación con factores económicos |

---

## 6. PRÓXIMOS PASOS

1. **Preprocessing (04):** Feature engineering, normalización
2. **Modelado (05):** LSTM, Transformer, XGBoost
3. **Evaluación (06):** Métricas por sector
4. **Análisis (07):** SHAP, explicabilidad

---

**Status:** ✅ EDA Completado
**Duración:** ~{datetime.now().strftime('%H:%M:%S')}
"""

    filepath = os.path.join(OUTPUT_DIR, "EDA_REPORT.md")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)

    log_message(f"✓ Guardado: EDA_REPORT.md")
    return report

# ============================================================================
# MAIN
# ============================================================================

def main():
    log_message("="*70)
    log_message("ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    log_message(f"Fecha/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message("="*70)

    # 1. Cargar datos
    tech_data, energy_data, fred_data = load_data()

    # 2. Estadísticas
    tech_stats, energy_stats, tech_vol, energy_vol = calculate_statistics(tech_data, energy_data)

    # 3. Visualizaciones
    log_message("\n=== GENERANDO VISUALIZACIONES ===")
    plot_price_comparison(tech_data, energy_data)
    plot_volatility_comparison(tech_data, energy_data)
    plot_distribution(tech_data, energy_data)
    plot_volatility_by_ticker(tech_data, energy_data)
    plot_correlation_with_fred(tech_data, energy_data, fred_data)

    # 4. Reporte
    create_eda_report(tech_stats, energy_stats, tech_vol, energy_vol)

    log_message("\n" + "="*70)
    log_message("✓ EDA COMPLETADO")
    log_message("="*70)
    log_message("\nArchivos generados:")
    log_message("  1. 01_precios_normalizados.png")
    log_message("  2. 02_volatilidad_comparativa.png")
    log_message("  3. 03_distribucion_retornos.png")
    log_message("  4. 04_volatilidad_por_ticker.png")
    log_message("  5. 05_correlacion_fred.png")
    log_message("  6. EDA_REPORT.md")
    log_message("  7. eda_log.txt")
    log_message("\nPróximo paso: Preprocessing en 04_Preprocessing/")

if __name__ == "__main__":
    main()
