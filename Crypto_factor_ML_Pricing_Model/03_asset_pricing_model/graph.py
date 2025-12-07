import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

REGRESSION_RESULTS_PATH = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/result/crypto_4_daily_factors_stats.csv'

# Output Directory for Charts
CHART_OUTPUT_DIR = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/results/charts/'

df = pd.read_csv(REGRESSION_RESULTS_PATH)

# Optional: clean rounding just for nicer labels (does not affect accuracy)
df_plot = df.copy()
df_plot['Mean Return (Daily)'] = df_plot['Mean Return (Daily)'].round(6)
df_plot['Volatility (Daily σ)'] = df_plot['Volatility (Daily σ)'].round(6)
df_plot['Sharpe Ratio (Annualized)'] = df_plot['Sharpe Ratio (Annualized)'].round(4)

# =============================================================================
# 1. Three-panel bar chart (most common in research)
# =============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(1, 3, figsize=(16, 6))

# Mean Daily Return
bars1 = axes[0].bar(df_plot['Factor'],
                    df_plot['Mean Return (Daily)'],
                    color=['#d62728' if x < 0 else '#2ca02c' for x in df_plot['Mean Return (Daily)']])
axes[0].set_title('Mean Daily Return', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Daily Return')
axes[0].bar_label(bars1, fmt='%.5f', fontsize=9)

# Daily Volatility
bars2 = axes[1].bar(df_plot['Factor'], df_plot['Volatility (Daily σ)'], color='#1f77b4')
axes[1].set_title('Daily Volatility (σ)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Volatility')
axes[1].bar_label(bars2, fmt='%.5f', fontsize=9)

# Annualized Sharpe Ratio
colors_sr = ['#d62728' if x < 0 else '#2ca02c' for x in df_plot['Sharpe Ratio (Annualized)']]
bars3 = axes[2].bar(df_plot['Factor'], df_plot['Sharpe Ratio (Annualized)'], color=colors_sr)
axes[2].set_title('Annualized Sharpe Ratio', fontsize=14, fontweight='bold')
axes[2].set_ylabel('Sharpe Ratio')
axes[2].axhline(0, color='black', linewidth=0.8)
axes[2].bar_label(bars3, fmt='%.3f', fontsize=10)

plt.suptitle('Factor Performance Summary', fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout()

# Save instead of showing
plt.savefig(os.path.join(CHART_OUTPUT_DIR, 'Factor_Performance_3Panel.png'), dpi=300, bbox_inches='tight')
plt.close(fig)   # close to free memory

# =============================================================================
# 2. Risk–Return scatter plot (very popular in papers & presentations)
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 7))

scatter = ax.scatter(df['Volatility (Daily σ)'],
                     df['Mean Return (Daily)'],
                     s=1000,
                     c=df['Sharpe Ratio (Annualized)'],
                     cmap='RdYlGn',
                     alpha=0.9,
                     edgecolors='black', linewidth=1)

for i, row in df.iterrows():
    ax.annotate(row['Factor'],
                (row['Volatility (Daily σ)'], row['Mean Return (Daily)']),
                xytext=(6, 6), textcoords='offset points',
                fontsize=12, fontweight='bold', ha='left')

plt.colorbar(scatter, label='Annualized Sharpe Ratio')
ax.set_xlabel('Daily Volatility (σ)', fontsize=12)
ax.set_ylabel('Mean Daily Return', fontsize=12)
ax.set_title('Risk–Return Profile of Factors', fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
ax.axvline(df['Volatility (Daily σ)'].mean(), color='gray', linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(CHART_OUTPUT_DIR, 'Factor_Risk_Return_Scatter.png'), dpi=300, bbox_inches='tight')
plt.close(fig)

print("Two PNG files have been saved:")
print("   → Factor_Performance_3Panel.png")
print("   → Factor_Risk_Return_Scatter.png")