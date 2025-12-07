# Code to compute log returns, risk-free returns, momentum score, illiquidity (Amihud) ratio, and initial market beta calculation
# Output to be stored in: data

import pandas as pd
import numpy as np
import os

# ==================== CONFIGURATION ====================

# Input Files 
INDEX_INPUT_FILE = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/data/cmc_index_2025_daily.csv'
COIN_INPUT_FILE = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/data/crypto_index_2025_daily_FULL.csv'

# Output File (The final prepared data for Phase 2 & 3)
PROCESSED_DATA_PATH = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/data/crypto_index_2025_daily_PREPARED.csv'

# Calculation Parameters
RISK_FREE_RATE_DAILY = 0 
MOMENTUM_LOOKBACK_DAYS = 28 
ILLIQUIDITY_LOOKBACK_DAYS = 7 
VOLATILITY_LOOKBACK_DAYS = 20 # New metric: standard deviation over 20 days

# ==================== FUNCTIONS ====================

def calculate_market_factor(df_index: pd.DataFrame) -> pd.DataFrame:
    """Calculates Log Returns and the Market Factor (Rm-Rf) for the Index."""
    df_index['date'] = pd.to_datetime(df_index['date'])
    df_index.set_index('date', inplace=True)
    
    # Calculate Log Returns
    df_index['index_log_return'] = np.log(df_index['index_value'] / df_index['index_value'].shift(1))
    
    # Calculate Market Factor (R_MKT = Rm - Rf)
    df_index['R_MKT'] = df_index['index_log_return'] - RISK_FREE_RATE_DAILY
    
    print(f"  -> Calculated Market Factor (R_MKT) for {len(df_index)} days.")
    return df_index[['index_log_return', 'R_MKT']].reset_index()


def calculate_coin_metrics(df_coins: pd.DataFrame) -> pd.DataFrame:
    """Calculates Returns, Paper's Sorting Scores, and Alternative Daily Rolling Metrics."""
    
    # 1. Data Cleaning and Setup
    df_coins['date'] = pd.to_datetime(df_coins['date'])
    df_coins = df_coins[df_coins['price_usd'] > 0].dropna(subset=['price_usd', 'volume_24h', 'market_cap'])
    df_coins.sort_values(['id', 'date'], inplace=True)
    
    # 2. Calculate Daily Log Returns and Excess Returns
    df_coins['log_return'] = df_coins.groupby('id')['price_usd'].apply(
        lambda x: np.log(x / x.shift(1))
    )
    df_coins['EX_RETURN'] = df_coins['log_return'] - RISK_FREE_RATE_DAILY

    
    # --- PAPER'S METRICS (for weekly sorting) ---
    # These metrics are designed to be stable and are lagged by 1 day to prevent look-ahead bias
    
    # SIZE_SCORE: Lagged Market Cap
    df_coins['SIZE_SCORE'] = df_coins.groupby('id')['market_cap'].shift(1) 
    
    # AMIHUD_SCORE (Illiquidity): Lagged rolling average of |Return|/Volume
    df_coins['illiquidity_daily'] = np.abs(df_coins['log_return']) / df_coins['volume_24h']
    df_coins['AMIHUD_SCORE'] = df_coins.groupby('id')['illiquidity_daily'].rolling(
        window=ILLIQUIDITY_LOOKBACK_DAYS
    ).mean().reset_index(level=0, drop=True).shift(1)
    
    # MOMENTUM_SCORE: Lagged rolling average return
    df_coins['MOMENTUM_SCORE'] = df_coins.groupby('id')['log_return'].rolling(
        window=MOMENTUM_LOOKBACK_DAYS
    ).mean().reset_index(level=0, drop=True).shift(1)


    # --- ALTERNATIVE DAILY ROLLING METRICS (for comparison) ---
    # These metrics are designed to be used for sorting *every day*.
    # They are also calculated using lagged data (shift(1)) to maintain integrity.
    
    # VOLATILITY_SCORE (Standard Deviation): Inverse of Low Volatility Anomaly
    # This is a new factor metric to test!
    df_coins['VOLATILITY_SCORE'] = df_coins.groupby('id')['log_return'].rolling(
        window=VOLATILITY_LOOKBACK_DAYS
    ).std().reset_index(level=0, drop=True).shift(1)
    
    # DAILY_MOMENTUM: A very short-term momentum proxy
    df_coins['DAILY_MOMENTUM'] = df_coins.groupby('id')['log_return'].rolling(
        window=5 # 5-day average
    ).mean().reset_index(level=0, drop=True).shift(1)
    
    # DAILY_AMIHUD: A very short-term illiquidity proxy
    df_coins['DAILY_AMIHUD'] = df_coins.groupby('id')['illiquidity_daily'].rolling(
        window=3 # 3-day average
    ).mean().reset_index(level=0, drop=True).shift(1)


    print(f"  -> Calculated all returns and sorting scores for {df_coins['id'].nunique()} coins.")
    
    # Select final columns to save
    cols_to_keep = ['date', 'id', 'symbol', 'EX_RETURN', 'log_return', 
                    'SIZE_SCORE', 'MOMENTUM_SCORE', 'AMIHUD_SCORE', # Paper's metrics
                    'VOLATILITY_SCORE', 'DAILY_MOMENTUM', 'DAILY_AMIHUD' # Alternative metrics
                   ]
    
    return df_coins[cols_to_keep]


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    
    print("--- Starting 02_calculate_metrics.py (Calculating Paper and Alternative Metrics) ---")
    
    # 1. Load Data
    try:
        df_index_raw = pd.read_csv(INDEX_INPUT_FILE)
        df_coins_raw = pd.read_csv(COIN_INPUT_FILE)
    except FileNotFoundError as e:
        print(f"❌ ERROR: File not found. Ensure raw files are in the data/ folder. {e}")
        exit()

    # 2. Process Index Data
    print("\nProcessing Index Data...")
    df_market = calculate_market_factor(df_index_raw.copy())

    # 3. Process Individual Coin Data
    print("\nProcessing Individual Coin Data...")
    df_individual = calculate_coin_metrics(df_coins_raw.copy())
    
    # 4. Merge Data (Merge the Market Factor R_MKT with individual coin data)
    print("\nMerging data...")
    df_final = pd.merge(
        df_individual,
        df_market[['date', 'R_MKT']],
        on='date',
        how='inner'
    )
    
    # Remove rows with NaN values (start of time series due to rolling/shift operations)
    df_final.dropna(subset=['EX_RETURN', 'SIZE_SCORE', 'MOMENTUM_SCORE', 'AMIHUD_SCORE'], inplace=True)
    
    print(f"✅ FINAL SHAPE: {len(df_final['date'].unique())} days and {df_final['id'].nunique()} coins.")
    
    # 5. Save the final prepared dataset
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df_final.to_csv(PROCESSED_DATA_PATH, index=False)
    
    print(f"\n✅ SUCCESS: Final prepared data saved to {PROCESSED_DATA_PATH}")