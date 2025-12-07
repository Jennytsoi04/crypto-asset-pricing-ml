# Define the Target Variable ($\mathbf{Y}$): The coin's excess return in the next period ($R_{i,t+1}$).
# Define the Features ($\mathbf{X}$): All characteristics and risk metrics from the current period ($t$).
# Merge Data: Combine all components and create the necessary time-lag to prevent look-ahead bias.

import pandas as pd
import numpy as np
import os

# ==================== CONFIGURATION ====================

# Input Files
PROCESSED_DATA_PATH = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/data/crypto_index_2025_daily_PREPARED.csv'
REGRESSION_RESULTS_PATH = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/results/regression_results.csv'
FACTORS_PATH = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/data/crypto_4_daily_factors_FINAL.csv'

# Output File
OUTPUT_ML_DATASET = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/data/ml_dataset_final.csv'

# Set the daily risk-free rate explicitly (assuming 0.0, which is standard for daily crypto factor models)
RISK_FREE_RATE_DAILY = 0.0 


# ==================== FEATURE ENGINEERING FUNCTION ====================

def create_ml_dataset(proc_path: str, reg_path: str, factor_path: str, output_path: str, rf_rate: float):
    """Loads, cleans, engineers features, and saves the final ML-ready dataset."""
    
    print("--- Starting ML Feature Engineering ---")

    # 1. Load Data
    try:
        # Load Daily Coin Returns (R_i,t) and Characteristics (Size, Illiquidity, etc.)
        df_returns = pd.read_csv(proc_path, parse_dates=['date']).rename(columns={'id': 'coin_id'})
        
        # 🟢 FIX: Use the explicitly defined risk-free rate for Excess Return calculation
        df_returns['EX_RETURN'] = df_returns['log_return'] - rf_rate 

        # Load Estimated Betas (Beta_i,k) from the 4-Factor Model
        df_betas_raw = pd.read_csv(reg_path)
        df_betas = df_betas_raw[df_betas_raw['Model'] == '4_Factor'].copy()
        
        # Load Factor Returns (F_k,t)
        df_factors = pd.read_csv(factor_path, parse_dates=['date'])
        
    except FileNotFoundError as e:
        print(f"❌ ERROR: Input file not found: {e}. Ensure all previous scripts ran successfully.")
        return

    # 2. Prepare Betas (Static Features for each coin)
    
    BETA_COLUMNS = ['Coin_ID', 'Beta_R_MKT', 'Beta_CSMB', 'Beta_CMH', 'Beta_CIHML'] 
    df_betas = df_betas[['Coin_ID', 'Beta_R_MKT', 'Beta_CSMB', 'Beta_CMH', 'Beta_CIHML']].copy()
    

    # Rename Beta_CLMW to Beta_CMH to reflect the final, validated factor
    df_betas.rename(columns={'Coin_ID': 'coin_id', 'Beta_CLMW': 'Beta_CMH'}, inplace=True)
    
    df_betas = df_betas[['coin_id', 'Beta_R_MKT', 'Beta_CSMB', 'Beta_CMH', 'Beta_CIHML']].drop_duplicates()
    
    print(f"Loaded and prepared {len(df_betas)} unique Beta sets.")

    # 3. Create the Target Variable (Y): Next Day's Excess Return (R_i,t+1)
    
    df_ml = df_returns[['date', 'coin_id', 'EX_RETURN', 'SIZE_SCORE', 'AMIHUD_SCORE', 'MOMENTUM_SCORE']].copy()
    
    # The return at time t+1 is the target variable for the features at time t.
    df_ml['Y_TARGET_RETURN'] = df_ml.groupby('coin_id')['EX_RETURN'].shift(-1)
    
    df_ml.dropna(subset=['Y_TARGET_RETURN'], inplace=True)
    
    print(f"Target variable (Y) created with {len(df_ml)} data points.")
    
    # 4. Feature Consolidation (X)
    
    # Feature Set 2: Static Risk Exposures (Betas)
    df_final = df_ml.merge(df_betas, on='coin_id', how='left')
    
    df_final.dropna(inplace=True) 

    # 5. Final Output
    os.makedirs(os.path.dirname(output_path), exist_ok=True) # Ensure directory exists
    df_final.to_csv(output_path, index=False)
    print(f"\n✅ ML-Ready Dataset created with {len(df_final)} observations.")
    print(f"File saved to {output_path}")
    print("\nNext step: Model Training.")


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    
    # Run the feature creation, passing the constant risk-free rate
    create_ml_dataset(PROCESSED_DATA_PATH, REGRESSION_RESULTS_PATH, FACTORS_PATH, OUTPUT_ML_DATASET, RISK_FREE_RATE_DAILY)