"""
Regenera la Figura 5 (Correlación con factores económicos) como scatter de
retornos diarios estandarizados, que es lo que el caption del paper describe
-- la versión anterior graficaba niveles de precio normalizados en el tiempo,
lo cual produce correlación espuria por tendencia compartida, no por retornos.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "02_Descarga_Datos")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

plt.style.use('seaborn-v0_8-darkgrid')

tech = pd.read_csv(os.path.join(RAW_DIR, "tech_prices_raw.csv"))
energy = pd.read_csv(os.path.join(RAW_DIR, "energy_prices_raw.csv"))
fred = pd.read_csv(os.path.join(RAW_DIR, "fred_economic_raw.csv"))
for df in (tech, energy, fred):
    df['Date'] = pd.to_datetime(df['Date'])

tech['Return'] = tech.groupby('Ticker')['Close'].pct_change()
energy['Return'] = energy.groupby('Ticker')['Close'].pct_change()

tech_ret = tech.groupby('Date')['Return'].mean()
energy_ret = energy.groupby('Date')['Return'].mean()

fred_idx = fred.set_index('Date')
vix_chg = fred_idx['VIXCLS'].pct_change()
oil_chg = fred_idx['DCOILWTICO'].pct_change()

df_tech = pd.DataFrame({'sector_ret': tech_ret, 'vix_chg': vix_chg}).dropna()
df_energy = pd.DataFrame({'sector_ret': energy_ret, 'oil_chg': oil_chg}).dropna()

z_tech_ret = stats.zscore(df_tech['sector_ret'])
z_vix = stats.zscore(df_tech['vix_chg'])
z_energy_ret = stats.zscore(df_energy['sector_ret'])
z_oil = stats.zscore(df_energy['oil_chg'])

r_tech = np.corrcoef(z_vix, z_tech_ret)[0, 1]
r_energy = np.corrcoef(z_oil, z_energy_ret)[0, 1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.scatter(z_vix, z_tech_ret, alpha=0.4, s=15, color='#1f77b4')
b, a = np.polyfit(z_vix, z_tech_ret, 1)
xs = np.linspace(z_vix.min(), z_vix.max(), 100)
ax1.plot(xs, a + b * xs, color='red', linewidth=2)
ax1.set_title(f'Technology Daily Return vs. VIX Change\n(standardized, r={r_tech:.2f})', fontsize=12, fontweight='bold')
ax1.set_xlabel('VIX Daily % Change (standardized)')
ax1.set_ylabel('Tech Sector Daily Return (standardized)')

ax2.scatter(z_oil, z_energy_ret, alpha=0.4, s=15, color='#ff7f0e')
b2, a2 = np.polyfit(z_oil, z_energy_ret, 1)
xs2 = np.linspace(z_oil.min(), z_oil.max(), 100)
ax2.plot(xs2, a2 + b2 * xs2, color='red', linewidth=2)
ax2.set_title(f'Energy Daily Return vs. WTI Oil Change\n(standardized, r={r_energy:.2f})', fontsize=12, fontweight='bold')
ax2.set_xlabel('WTI Daily % Change (standardized)')
ax2.set_ylabel('Energy Sector Daily Return (standardized)')

plt.tight_layout()
filepath = os.path.join(OUTPUT_DIR, "figure5_economic_correlation.png")
plt.savefig(filepath, dpi=300, bbox_inches='tight')
plt.close()

print(f"r_tech (VIX) = {r_tech:.4f}")
print(f"r_energy (Oil) = {r_energy:.4f}")
print(f"Saved: {filepath}")
