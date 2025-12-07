import pandas as pd
import numpy as np
import os


# Input File 
PROCESSED_DATA_PATH = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/data/crypto_4_daily_factors_FINAL.csv'

# Output File for the final factor time series
OUTPUT_FULL_FACTORS = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/result/crypto_4_daily_factors_stats.csv'

df = pd.read_csv(PROCESSED_DATA_PATH, index_col='date', parse_dates=True)

stats = {}

for col in df.columns:
    mean_daily = df[col].mean()
    vol_daily = df[col].std()
    sharpe_annual = np.sqrt(365) * mean_daily / vol_daily
    factor = col
    
    stats[col] = {
        'Factor': factor,
        'Mean Return (Daily)': mean_daily,
        'Volatility (Daily σ)': vol_daily,
        'Sharpe Ratio (Annualized)': sharpe_annual
    }

df_stats = pd.DataFrame(stats).T
os.makedirs(os.path.dirname(OUTPUT_FULL_FACTORS), exist_ok=True)
df_stats.to_csv(OUTPUT_FULL_FACTORS)
print(df_stats)
