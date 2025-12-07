# code to run the CAPM regression for all 20 coins

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# ==================== CONFIGURATION ====================

# Input Files
PROCESSED_DATA_PATH = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/data/crypto_index_2025_daily_PREPARED.csv' # Individual coin data
FACTORS_PATH = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/data/crypto_4_daily_factors_FINAL.csv'            # Factor time series

# Output File
OUTPUT_REGRESSION_RESULTS = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/results/regression_results.csv'

# ==================== CORE FUNCTION ====================

def run_asset_pricing_regression(df_merged: pd.DataFrame, coin_id: int, factors: list, model_name: str) -> dict:
    """
    Runs an OLS regression for a single coin using the specified factors.
    """
    # Filter data for the specific coin
    df_coin = df_merged[df_merged['coin_id'] == coin_id].copy()
    
    # Define Dependent Variable (Y) and Independent Variables (X)
    Y = df_coin['EX_RETURN']
    X = df_coin[factors]
    
    # Add a constant (intercept) term for Alpha (Alpha is the intercept)
    X = sm.add_constant(X)
    
    # Run OLS Regression
    model = sm.OLS(Y, X, missing='drop').fit()
    
    # --- Collect Results ---
    
    results = {
        'Coin_ID': coin_id,
        'Model': model_name,
        'Observations': model.nobs,
        'R_Squared': model.rsquared,
        'Adj_R_Squared': model.rsquared_adj,
        'Alpha': model.params['const'],
        'Alpha_t_stat': model.tvalues['const']
    }
    
    # Collect Beta coefficients and t-stats dynamically
    for factor in factors:
        results[f'Beta_{factor}'] = model.params.get(factor, np.nan)
        results[f't_stat_{factor}'] = model.tvalues.get(factor, np.nan)
        
    return results

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    
    print("--- Starting 01_capm_regression.py (CAPM Model Test) ---")
    
    # 1. Load Data
    try:
        df_ind = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=['date'])
        df_factors = pd.read_csv(FACTORS_PATH, index_col='date', parse_dates=True)
    except FileNotFoundError as e:
        print(f"❌ ERROR: Input file not found: {e}. Ensure prior steps are run.")
        exit()

    # Rename column for consistency
    df_ind.rename(columns={'id': 'coin_id'}, inplace=True)
    df_ind = df_ind[['date', 'coin_id', 'EX_RETURN']].set_index('date')
    
    # Merge individual coin excess returns with factor returns
    df_merged = df_ind.join(df_factors, how='inner').dropna()
    
    # Identify unique coins to iterate over
    unique_coins = df_merged['coin_id'].unique()
    print(f"Loaded data with {len(unique_coins)} unique coins.")

    # 2. Define CAPM Factors
    CAPM_FACTORS = ['R_MKT']
    
    all_results = []
    
    # 3. Run CAPM for all coins
    for coin_id in unique_coins:
        results = run_asset_pricing_regression(
            df_merged=df_merged,
            coin_id=coin_id,
            factors=CAPM_FACTORS,
            model_name='CAPM'
        )
        all_results.append(results)
        
    # 4. Convert to DataFrame and Save Results
    df_results = pd.DataFrame(all_results)
    
    # Ensure the results directory exists
    os.makedirs(os.path.dirname(OUTPUT_REGRESSION_RESULTS), exist_ok=True)
    df_results.to_csv(OUTPUT_REGRESSION_RESULTS, index=False)
    
    print(f"\n✅ SUCCESS: CAPM regression run for {len(unique_coins)} coins.")
    print(f"Results saved to {OUTPUT_REGRESSION_RESULTS}")
    print("\nFirst 5 rows of CAPM Results:")
    print(df_results.head())