import pandas as pd
import os

# Input & Output
INPUT_PATH = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/results/regression_results.csv'
OUTPUT_PATH = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/results/CAPM' \
'_reg_stats.csv'

# Load data
df = pd.read_csv(INPUT_PATH)

# Keep only CAPM results
df = df[df['Model'] == 'CAPM']

# Compute summary statistics
stats = {
    'Mean_R_Squared': df['R_Squared'].mean(),
    'Mean_Adj_R_Squared': df['Adj_R_Squared'].mean(),
    'Mean_Alpha': df['Alpha'].mean(),
    'Mean_Alpha_t_stat': df['Alpha_t_stat'].mean(),
    'Mean_|Alpha_t_stat|': df['Alpha_t_stat'].abs().mean(),
    'Share_Significant_Alpha': (df['Alpha_t_stat'].abs() > 1.96).mean(),
    'Mean_Beta_R_MKT': df['Beta_R_MKT'].mean(),
    'Mean_t_stat_Beta_R_MKT_': df['t_stat_R_MKT'].mean()
}

# Create summary table
df_stats = pd.DataFrame([stats])

# Save
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df_stats.to_csv(OUTPUT_PATH, index=False)

print(df_stats)
