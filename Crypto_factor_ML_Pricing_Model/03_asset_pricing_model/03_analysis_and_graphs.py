# code to perform analysis (e.g. calculate sharpe ratios, alpha t-sats) and generate all graphs

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os

# ==================== CONFIGURATION ====================

# Input Files
FACTORS_PATH = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/data/crypto_4_daily_factors_FINAL.csv'
REGRESSION_RESULTS_PATH = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/results/regression_results.csv'

# Output Directory for Charts
CHART_OUTPUT_DIR = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/results/charts/'
# ==================== ANALYSIS FUNCTIONS ====================

def generate_comparison_charts(df_results: pd.DataFrame, df_factors: pd.DataFrame, chart_dir: str):
    """Generates charts for model comparison and factor performance."""
    
    # 1. R-Squared Comparison Chart (CAPM vs. 4-Factor)
    
    print("\nGenerating R-Squared Comparison Chart...")
    # Pivot the R-Squared results for easy comparison across coins
    df_r_squared = df_results.pivot_table(
        index='Coin_ID', 
        columns='Model', 
        values='Adj_R_Squared'
    ).sort_values(by='CAPM', ascending=False)
    
    plt.figure(figsize=(14, 7))
    df_r_squared.plot(kind='bar', figsize=(14, 7))

    plt.title(r'Comparison of Model Explanatory Power ($\mathbf{Adj. R^2}$) Across Coins')
    plt.xlabel('Coin ID')
    plt.ylabel(r'$\mathbf{Adj. R^2}$ Value (Proportion of Variance Explained)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model')
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, '01_R_squared_comparison.png'))
    plt.close()
    print("  -> Chart 1 saved: 01_R_squared_comparison.png")
    
    # 2. Factor Sharpe Ratios Chart
    
    print("\nGenerating Factor Sharpe Ratios Chart...")
    # Calculate Sharpe Ratio for each factor
    # Annualized Sharpe = (Mean Daily Return / Standard Deviation of Daily Return) * sqrt(252 trading days)
    factor_stats = df_factors.drop(columns=['R_MKT']).agg(['mean', 'std']) # Exclude R_MKT for this comparison
    factor_stats.loc['Sharpe_Ratio'] = (factor_stats.loc['mean'] / factor_stats.loc['std']) * np.sqrt(252)
    
    df_sharpe = factor_stats.loc['Sharpe_Ratio'].to_frame(name='Sharpe Ratio')
    df_sharpe = df_sharpe.sort_values(by='Sharpe Ratio', ascending=False)

    plt.figure(figsize=(8, 6))
    # Use different colors for different factors for clarity
    df_sharpe['Sharpe Ratio'].plot(
        kind='bar', 
        color=['darkblue', 'green', 'orange'] 
    )

    plt.title('Annualized Sharpe Ratio for New Factors (CSMB, CMH, CIHML)')
    plt.xlabel('Factor')
    plt.ylabel('Annualized Sharpe Ratio')
    plt.xticks(rotation=0)
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, '02_factor_sharpe_ratios.png'))
    plt.close()
    print("  -> Chart 2 saved: 02_factor_sharpe_ratios.png")
    
    # 3. Alpha (Intercept) Comparison Chart
    
    print("\nGenerating Alpha Comparison Chart...")
    # Pivot the Alpha t-stats for comparison
    df_alpha_t = df_results.pivot_table(
        index='Coin_ID', 
        columns='Model', 
        values='Alpha_t_stat'
    ).sort_values(by='CAPM', ascending=False)
    
    plt.figure(figsize=(14, 7))
    df_alpha_t.plot(kind='bar', figsize=(14, 7))
    
    # Add lines for statistical significance (t-stat > |2| is often considered significant)
    plt.axhline(2.0, color='red', linestyle='--', linewidth=1, label='t=2.0 (5% Sig.)')
    plt.axhline(-2.0, color='red', linestyle='--', linewidth=1)

    plt.title('Alpha t-statistic Comparison (CAPM vs. 4-Factor)')
    plt.xlabel('Coin ID')
    plt.ylabel('Alpha t-statistic')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Model')
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(chart_dir, '03_alpha_t_stat_comparison.png'))
    plt.close()
    print("  -> Chart 3 saved: 03_alpha_t_stat_comparison.png")


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    
    print("--- Starting 03_analyze_results.py (Analyzing Factor Model Performance) ---")
    
    # 1. Load Data
    try:
        df_results = pd.read_csv(REGRESSION_RESULTS_PATH)
        df_factors = pd.read_csv(FACTORS_PATH, index_col='date', parse_dates=True)
        # Drop the R_MKT factor if it contains the risk-free rate, as we'll calculate Sharpe on excess returns
        df_factors.index.name = 'date'
        
    except FileNotFoundError as e:
        print(f"❌ ERROR: Required file not found: {e}. Ensure 01/02_regression scripts are run.")
        exit()

    # 2. Setup Chart Directory
    os.makedirs(CHART_OUTPUT_DIR, exist_ok=True)
    
    # 3. Run Analysis and Generate Charts
    generate_comparison_charts(df_results, df_factors, CHART_OUTPUT_DIR)
    
    print("\n✅ ANALYSIS COMPLETE: Charts are ready for inclusion in your report.")
    print(f"Check the directory: {CHART_OUTPUT_DIR}")