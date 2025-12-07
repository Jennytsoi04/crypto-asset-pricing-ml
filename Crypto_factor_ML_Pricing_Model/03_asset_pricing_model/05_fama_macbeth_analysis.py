import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

# ==================== CONFIGURATION ====================

# Input Files
PROCESSED_DATA_PATH = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/data/crypto_index_2025_daily_PREPARED.csv' # Individual coin data (for returns)
FACTORS_PATH = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/data/crypto_4_daily_factors_FINAL.csv'            # Factor time series (for dates)
REGRESSION_RESULTS_PATH = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/results/regression_results.csv'        # Your estimated Betas (from 02_4factor_regression.py)

# Output File
OUTPUT_FAMA_MACBETH = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/results/fama_macbeth_results.csv'

# ==================== CORE FUNCTION: FAMA-MACBETH ====================

def run_fama_macbeth(df_returns: pd.DataFrame, df_betas: pd.DataFrame, factor_names: list) -> pd.DataFrame:
    """
    Performs the two-step Fama-MacBeth cross-sectional regression.
    
    Step 1 (Betas) is assumed complete. This runs Step 2.
    """
    
    # 1. Prepare Data and Get Dates
    all_dates = df_returns.index.unique()
    daily_lambdas = []
    
    # Ensure Betas are ready for merge (indexed by Coin_ID)
    df_betas = df_betas.set_index('Coin_ID')
    
    print(f"Running cross-sectional regression for {len(all_dates)} days...")

    # 2. Loop through each day (Cross-Sectional Regression)
    for date in all_dates:
        # Get returns for all coins on this specific day (the dependent variable, Y)
        df_daily_returns = df_returns.loc[date].copy()
        
        # Merge daily returns with the pre-estimated Betas (the independent variables, X)
        df_regression = df_daily_returns.merge(df_betas, left_on='coin_id', right_index=True, how='inner')
        
        # Check if enough data is available for OLS (min 5-6 points usually needed)
        if len(df_regression) < 5: 
            continue 

        # Define Y and X for the regression
        Y = df_regression['EX_RETURN']
        X = df_regression[[f'Beta_{f}' for f in factor_names]] # Use the Betas as the predictors
        
        # Add a constant term for the intercept (Lambda_0)
        X = sm.add_constant(X)
        
        # Run OLS for this single day
        model = sm.OLS(Y, X, missing='drop').fit()
        
        # Store the daily Lambda coefficients (Factor Risk Premiums)
        lambdas = {'date': date}
        
        # Collect the daily coefficients (lambda_t)
        for factor in ['const'] + [f'Beta_{f}' for f in factor_names]:
            # Use .get() to handle potential missing data if the model didn't converge perfectly
            lambdas[factor] = model.params.get(factor, np.nan) 
            
        daily_lambdas.append(lambdas)

    # 3. Final Fama-MacBeth Output (Time-Series Average of Daily Lambdas)
    
    df_lambdas = pd.DataFrame(daily_lambdas).set_index('date').dropna()
    
    # Calculate the mean (Average Lambda) and standard deviation of the daily lambdas
    lambda_means = df_lambdas.mean()
    lambda_std_err = df_lambdas.std() / np.sqrt(len(df_lambdas)) # FM standard error calculation
    
    # Calculate the final FM t-statistic
    lambda_t_stat = lambda_means / lambda_std_err
    
    # Compile the final results table
    fm_results = pd.DataFrame({
        'Average_Lambda_Premium': lambda_means,
        'FM_T_Statistic': lambda_t_stat
    })
    
    # Clean up index names for presentation
    fm_results.index = [f.replace('Beta_', '') for f in fm_results.index]
    fm_results.rename(index={'const': 'Alpha_Intercept (Lambda 0)'}, inplace=True)
    
    return fm_results.round(4)


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    
    print("--- Starting 05_fama_macbeth_analysis.py ---")
    
    # 1. Load Data
    try:
        # Returns (for Y in Step 2)
        df_ind = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=['date']).rename(columns={'id': 'coin_id'})
        df_factors = pd.read_csv(FACTORS_PATH, parse_dates=['date'])
        df_returns = df_ind[['date', 'coin_id', 'EX_RETURN']].set_index('date').join(df_factors[['date', 'R_MKT']].set_index('date'), how='inner').dropna()

        # Betas (for X in Step 2)
        df_betas_raw = pd.read_csv(REGRESSION_RESULTS_PATH)
        
    except FileNotFoundError as e:
        print(f"❌ ERROR: Input file not found: {e}. Ensure prior steps (01-02) are run.")
        exit()
        
    # 2. Filter and Prepare Betas
    # We only need the Betas from the final 4-Factor model run
    df_betas = df_betas_raw[df_betas_raw['Model'] == '4_Factor'].copy()
    
    # Rename the factors to match the names used throughout your analysis
    FACTOR_NAMES = ['R_MKT', 'CSMB', 'CMH', 'CIHML'] # Use CMH instead of CLMW
    
    # Filter for necessary beta columns (assuming you renamed the CLMW column in your file)
    BETA_COLUMNS = ['Coin_ID'] + [f'Beta_{f}' for f in FACTOR_NAMES]
    df_betas = df_betas[BETA_COLUMNS].copy()

    # 3. Run Fama-MacBeth
    fm_results = run_fama_macbeth(
        df_returns=df_returns,
        df_betas=df_betas,
        factor_names=FACTOR_NAMES
    )
    
    # 4. Save and Print Results
    os.makedirs(os.path.dirname(OUTPUT_FAMA_MACBETH), exist_ok=True)
    fm_results.to_csv(OUTPUT_FAMA_MACBETH)
    
    print("\n" + "="*40)
    print("✅ FAMA-MACBETH FINAL RESULTS (Factor Premiums)")
    print(fm_results.to_markdown(numalign="left", stralign="left"))
    print("="*40)
    print(f"Results saved to {OUTPUT_FAMA_MACBETH}")

    print("\nYour econometric analysis is now complete. The next step is Machine Learning.")