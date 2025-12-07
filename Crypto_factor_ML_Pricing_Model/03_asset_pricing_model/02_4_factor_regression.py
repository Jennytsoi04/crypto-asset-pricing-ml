#code to run the crypto 4-factor model regression for all 20 coins
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# ==================== CONFIGURATION ====================

# Input Files
PROCESSED_DATA_PATH = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/data/crypto_index_2025_daily_PREPARED.csv' # Individual coin data
FACTORS_PATH = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/data/crypto_4_daily_factors_FINAL.csv'            # Factor time series

# Output File (Appending to the file created by 01_capm_regression.py)
OUTPUT_REGRESSION_RESULTS = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/results/regression_results.csv'

# ==================== CORE FUNCTION (REUSED from 01_capm_regression) ====================

def run_asset_pricing_regression(df_merged: pd.DataFrame, coin_id: str, factors: list, model_name: str) -> dict:
    """
    Runs an OLS regression for a single coin using the specified factors.
    """
    # Filter data for the specific coin
    df_coin = df_merged[df_merged['coin_id'] == coin_id].copy()
    
    # Define Dependent Variable (Y) and Independent Variables (X)
    Y = df_coin['EX_RETURN']
    X = df_coin[factors]
    
    # Add a constant (intercept) term for Alpha
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
    for factor in ['R_MKT', 'CSMB', 'CMH', 'CIHML']:
        # Use .get() to safely handle cases where a factor is not in the model (e.g., CAPM)
        results[f'Beta_{factor}'] = model.params.get(factor, np.nan)
        results[f't_stat_{factor}'] = model.tvalues.get(factor, np.nan)
        
    return results

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    
    print("--- Starting 02_4factor_regression.py (4-Factor Model Test) ---")
    
    # 1. Load Data
    try:
        df_ind = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=['date'])
        df_factors = pd.read_csv(FACTORS_PATH, index_col='date', parse_dates=True)
        # Load existing CAPM results
        df_existing_results = pd.read_csv(OUTPUT_REGRESSION_RESULTS)
    except FileNotFoundError as e:
        print(f"❌ ERROR: Input file not found or CAPM results missing: {e}. Ensure prior steps are run.")
        exit()

    # Data Preparation
    df_ind.rename(columns={'id': 'coin_id'}, inplace=True)
    df_ind = df_ind[['date', 'coin_id', 'EX_RETURN']].set_index('date')
    
    # Merge individual coin excess returns with factor returns
    df_merged = df_ind.join(df_factors, how='inner').dropna()
    
    unique_coins = df_merged['coin_id'].unique()
    print(f"Loaded data with {len(unique_coins)} unique coins.")

    # 2. Define 4-Factor Factors
    FOUR_FACTOR_FACTORS = ['R_MKT', 'CSMB', 'CMH', 'CIHML']
    
    new_results = []
    
    # 3. Run 4-Factor Model for all coins
    for coin_id in unique_coins:
        results = run_asset_pricing_regression(
            df_merged=df_merged,
            coin_id=coin_id,
            factors=FOUR_FACTOR_FACTORS,
            model_name='4_Factor'
        )
        new_results.append(results)
        
    # 4. Combine New Results with Existing CAPM Results
    df_new_results = pd.DataFrame(new_results)
    df_final_results = pd.concat([df_existing_results, df_new_results], ignore_index=True)
    
    # 5. Save the Combined Results
    os.makedirs(os.path.dirname(OUTPUT_REGRESSION_RESULTS), exist_ok=True)
    df_final_results.to_csv(OUTPUT_REGRESSION_RESULTS, index=False)
    
    print(f"\n✅ SUCCESS: 4-Factor regression run for {len(unique_coins)} coins.")
    print(f"Combined CAPM and 4-Factor results saved to {OUTPUT_REGRESSION_RESULTS}")
    print("\nFirst 10 rows of Combined Regression Results:")
    print(df_final_results.head(10))