import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Used for generating the heatmap
import os

# ==================== CONFIGURATION ====================

# Input File (Factor Time Series)
FACTORS_PATH = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/data/crypto_4_daily_factors_FINAL.csv'

# Output Files
OUTPUT_CORR_MATRIX_CSV = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/results/factor_correlation_matrix.csv'
OUTPUT_CORR_MATRIX_PLOT = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/results/charts/04_factor_correlation_heatmap.png'
# ==================== CORE FUNCTION ====================

def analyze_factor_correlation(factors_path: str, output_csv: str, output_plot: str):
    """
    Loads factor data, calculates the correlation matrix, saves it to CSV,
    and generates a heatmap visualization.
    """
    print("--- Starting Factor Correlation Analysis ---")
    
    # 1. Load Data
    try:
        df_factors = pd.read_csv(factors_path, index_col='date', parse_dates=True)
        # We only need the four factor columns
        df_factors = df_factors[['R_MKT', 'CSMB', 'CMH', 'CIHML']]
        
    except FileNotFoundError:
        print(f"❌ ERROR: Input file not found at {factors_path}. Please run 01_factor_sort.py first.")
        return

    # 2. Calculate Correlation Matrix
    correlation_matrix = df_factors.corr()
    print("\n✅ Calculated Correlation Matrix:")
    print(correlation_matrix.to_markdown(numalign="left", stralign="left"))

    # 3. Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    correlation_matrix.to_csv(output_csv)
    print(f"\n✅ Correlation matrix saved to {output_csv}")

    # 4. Generate Heatmap Visualization
    plt.figure(figsize=(8, 7))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=.5,
        cbar_kws={'label': 'Correlation Coefficient'}
    )
    plt.title('Correlation Heatmap of the Four Crypto Factors')
    plt.tight_layout()
    
    # Save Plot
    os.makedirs(os.path.dirname(output_plot), exist_ok=True)
    plt.savefig(output_plot)
    plt.close()
    print(f"✅ Correlation heatmap saved to {output_plot}")

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    
    # NOTE: You will need to correct the file path in the FACTORS_PATH variable 
    # if you are still running into FileNotFoundError issues.
    analyze_factor_correlation(FACTORS_PATH, OUTPUT_CORR_MATRIX_CSV, OUTPUT_CORR_MATRIX_PLOT)
