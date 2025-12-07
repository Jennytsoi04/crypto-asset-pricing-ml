# The core code implementing the 2X3 sorting scheme (Size vs charateristics) to create the six portfolios

import pandas as pd
import numpy as np
import os

# ==================== CONFIGURATION ====================

# Input File (from 02_calculate_metrics.py)
PROCESSED_DATA_PATH = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/data/crypto_index_2025_daily_PREPARED.csv'

# Output File for the final factor time series
OUTPUT_FULL_FACTORS = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/data/crypto_4_daily_factors_FINAL.csv'

# Sorting Frequency: Define the day of the week for re-sorting (0=Monday, 6=Sunday)
RESORT_DAY_OF_WEEK = 0 # Monday

# ==================== CORE FACTOR FUNCTIONS ====================

def perform_2x3_sort_and_group(df_week: pd.DataFrame, char_col: str) -> pd.DataFrame:
    """
    Performs the 2x3 sort on the designated resort day data (t-1 metrics) 
    and assigns coins to one of the 6 portfolios (e.g., Small-Low).
    
    Args:
        df_week: DataFrame slice containing data for one sorting week.
        char_col: The characteristic column name for the inner 3-way sort.
    
    Returns:
        DataFrame with coin_id, date, SIZE_GROUP, and CHAR_GROUP assignments.
    """
    # Use the sorting metrics from the day BEFORE the sort date (x['date'].min())
    # This represents the information available at the time of the sort.
    df_sort = df_week[df_week['date'] == df_week['date'].min()].copy()
    
    # Drop coins with missing sorting scores (NaNs at the start of the time series)
    df_sort.dropna(subset=['SIZE_SCORE', char_col], inplace=True)
    
    if df_sort.empty:
        return pd.DataFrame()

    # --- 1. Size Sort (2 Groups: Small/Big) ---
    # Breakpoint: Median of the SIZE_SCORE distribution.
    median_size = df_sort['SIZE_SCORE'].median()
    df_sort['SIZE_GROUP'] = np.where(df_sort['SIZE_SCORE'] <= median_size, 'Small', 'Big')

    # --- 2. Characteristic Sort (3 Groups: Low/Medium/High) ---
    # Breakpoints: 33rd and 67th percentiles of the characteristic score.
    breakpoints = df_sort[char_col].quantile([1/3, 2/3])
    
    def assign_char_group(score):
        if score <= breakpoints.iloc[0]:
            return 'Low'
        elif score <= breakpoints.iloc[1]:
            return 'Medium'
        else:
            return 'High'
            
    df_sort['CHAR_GROUP'] = df_sort[char_col].apply(assign_char_group)

    # Return only the necessary assignment columns for merging later
    df_sort['coin_id'] = df_sort['coin_id']
    return df_sort[['coin_id', 'date', 'SIZE_GROUP', 'CHAR_GROUP']]


def calculate_factor_time_series(df_data: pd.DataFrame, sorting_char: str, factor_name: str, long_char_is_high: bool = True) -> pd.Series:
    """
    Constructs the daily factor returns (CSMB, CIHML, or CLMW) over the entire sample period.
    """
    
    # --- 1. Identify Weekly Sorting Dates ---
    df_data['day_of_week'] = df_data['date'].dt.dayofweek
    df_data['resort_day'] = np.where(df_data['day_of_week'] == RESORT_DAY_OF_WEEK, 
                                     df_data['date'], pd.NaT)
    # Forward-fill the week start date to all days within that week
    df_data['resort_day'] = df_data['resort_day'].ffill() 

    # --- 2. Weekly Portfolio Formation (Apply the 2x3 Sort) ---
    
    # Group by the 'resort_day' and apply the sort, which assigns groups (Small-Low, etc.) for that week
    df_assignments = df_data.groupby('resort_day').apply(
        lambda x: perform_2x3_sort_and_group(x, sorting_char)
    ).reset_index(level=0, drop=True)
    
    # Merge the portfolio assignments back to the original daily data
    # We join the assignment back to the coin_id and the date the assignment was made
    df_merged = df_data.merge(
        df_assignments[['coin_id', 'date', 'SIZE_GROUP', 'CHAR_GROUP']], 
        on=['coin_id', 'date'], 
        how='left'
    )
    
    # --- 3. Forward-Fill Assignments and Calculate Daily Portfolio Returns ---
    
    # The portfolio assignment (SIZE_GROUP, CHAR_GROUP) is fixed for the week. 
    # We use ffill() to propagate the assignment from the resort day to the rest of the week's trading days.
    df_merged = df_merged.sort_values(['coin_id', 'date'])
    df_merged[['SIZE_GROUP', 'CHAR_GROUP']] = df_merged.groupby('coin_id')[['SIZE_GROUP', 'CHAR_GROUP']].ffill()

    # Group by the combined portfolio (e.g., Small-Low) and calculate the daily Equal-Weighted return (R_ew)
    # The paper uses value-weighted returns, but Equal-Weighting is a common, simpler proxy.
    portfolio_daily_returns = df_merged.groupby(['date', 'SIZE_GROUP', 'CHAR_GROUP']).agg(
        R_ew=('EX_RETURN', 'mean')
    ).reset_index()

    # --- 4. Calculate the Final Factor (HML) ---
    
    # Define the long and short legs based on the characteristic sort (CHAR_GROUP)
    long_char = 'High' if long_char_is_high else 'Low'
    short_char = 'Low' if long_char_is_high else 'High'

    # Filter for the relevant characteristic groups
    df_factor_components = portfolio_daily_returns[
        portfolio_daily_returns['CHAR_GROUP'].isin([long_char, short_char])
    ].copy()
    
    # Calculate the average return of the "Long" and "Short" legs for the two Size groups (Small and Big)
    # e.g., For CIHML, the Long leg is (Small-High + Big-High) / 2
    
    df_factor_components['factor_leg'] = np.where(df_factor_components['CHAR_GROUP'] == long_char, 'LONG', 'SHORT')
    
    factor_returns = df_factor_components.groupby(['date', 'factor_leg']).agg(
        factor_leg_return=('R_ew', 'mean')
    ).unstack(level='factor_leg')

    # Factor Return = LONG Leg Return - SHORT Leg Return
    factor_series = (factor_returns['factor_leg_return']['LONG'] - 
                     factor_returns['factor_leg_return']['SHORT']).rename(factor_name)

    return factor_series


def calculate_csmb(df_data: pd.DataFrame) -> pd.Series:
    """
    Calculates the daily CSMB (Crypto Small Minus Big) factor return.
    CSMB = Average of (Small-X Portfolios) - Average of (Big-X Portfolios), 
    where X is Low, Medium, and High characteristics.
    """
    
    # --- 1. Identify Weekly Sorting Dates ---
    df_data['day_of_week'] = df_data['date'].dt.dayofweek
    df_data['resort_day'] = np.where(df_data['day_of_week'] == RESORT_DAY_OF_WEEK, # When day_of_week == 0 (Monday)     
                                     df_data['date'], pd.NaT)                       #fill the resort_day with that date, else NaT
    df_data['resort_day'] = df_data['resort_day'].ffill()                           # Forward-fill the week start date to all days within that week
    
    # --- 2. Weekly Portfolio Formation (Sort on MOMENTUM_SCORE for the 2x3 grid) ---
    # We need to run a 2x3 sort once per week to define the 6 portfolios
    df_assignments = df_data.groupby('resort_day').apply(                      # each week has only one resort_day > so one portfolio for each week
        lambda x: perform_2x3_sort_and_group(x, 'MOMENTUM_SCORE') # Use any char score for the 2x3 split
    ).reset_index(level=0, drop=True)
    
    df_merged = df_data.merge(
        df_assignments[['coin_id', 'date', 'SIZE_GROUP', 'CHAR_GROUP']], 
        on=['coin_id', 'date'], 
        how='left'
    )
    
    # Forward-Fill Assignments
    df_merged = df_merged.sort_values(['coin_id', 'date'])
    df_merged[['SIZE_GROUP', 'CHAR_GROUP']] = df_merged.groupby('coin_id')[['SIZE_GROUP', 'CHAR_GROUP']].ffill()

    # --- 3. Calculate Daily Portfolio Returns ---
    # Group by Size (Small vs. Big) and calculate the daily average return for each group
    portfolio_daily_returns = df_merged.groupby(['date', 'SIZE_GROUP']).agg(
        R_size=('EX_RETURN', 'mean')
    ).unstack(level='SIZE_GROUP')
    
    # --- 4. Calculate CSMB Factor ---
    # CSMB = R_Small - R_Big
    csmb_series = (portfolio_daily_returns['R_size']['Small'] - 
                   portfolio_daily_returns['R_size']['Big']).rename('CSMB')
    
    return csmb_series

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    
    print("--- Starting 01_factor_sort.py (2x3 Factor Construction) ---")
    
    # Load data
    try:
        df_raw = pd.read_csv(PROCESSED_DATA_PATH)
        df_raw['date'] = pd.to_datetime(df_raw['date'])
        df_raw.rename(columns={'id': 'coin_id'}, inplace=True) 
        
    except FileNotFoundError:
        print(f"❌ ERROR: Input file not found at {PROCESSED_DATA_PATH}. Please run 02_calculate_metrics.py first.")
        exit()

    # --- 1. Calculate the HML Factors (using the calculate_factor_time_series function) ---

    # CLMW (Crypto Low Minus High / Reversal):   Long Low Momentum - Short High Momentum
    # Sorting Metric: MOMENTUM_SCORE
    # Factor definition: LONG Low Char - SHORT High Char --> long_char_is_high=False
    df_clmw = calculate_factor_time_series(df_raw.copy(), 'MOMENTUM_SCORE', 'CLMW', long_char_is_high=False)
    print("  -> CLMW (Momentum) Factor calculated.")
    # CIHML (Crypto Illiquidity High Minus Low): Long High Illiquidity - Short Low Illiquidity
    # Sorting Metric: AMIHUD_SCORE
    # Factor definition: LONG High Char - SHORT Low Char --> long_char_is_high=True
    df_cihml = calculate_factor_time_series(df_raw.copy(), 'AMIHUD_SCORE', 'CIHML', long_char_is_high=True)
    print("  -> CIHML (Illiquidity) Factor calculated.")
    
    # --- 2. Calculate the Size Factor ---

    # CSMB (Crypto Small Minus Big): Average Small Portfolio - Average Big Portfolio
    df_csmb = calculate_csmb(df_raw.copy())
    print("  -> CSMB (Size) Factor calculated.")

    # --- 3. Merge all factors (4 Factors Total) ---
    
    # Extract R_MKT from the original data (unique daily values)
    df_mkt = df_raw[['date', 'R_MKT']].drop_duplicates().set_index('date')

    # Combine the four daily factor time series
    df_factors = pd.concat([df_mkt, df_csmb, df_clmw, df_cihml], axis=1).dropna()
    df_factors.index.name = 'date'
    
    # --- 4. Save the final factor time series ---
    os.makedirs(os.path.dirname(OUTPUT_FULL_FACTORS), exist_ok=True)
    df_factors.to_csv(OUTPUT_FULL_FACTORS)
    
    print(f"\n✅ SUCCESS: Calculated 4 daily factors and saved {len(df_factors)} records to {OUTPUT_FULL_FACTORS}")
    print("\nFirst 5 rows of the Factor Time Series:")
    print(df_factors.head())