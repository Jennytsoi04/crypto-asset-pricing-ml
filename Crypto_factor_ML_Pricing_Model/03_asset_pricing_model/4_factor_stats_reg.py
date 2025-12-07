import pandas as pd
import os

# Input & Output
INPUT_PATH = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/results/regression_results.csv'
OUTPUT_PATH = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/results/4_factor' \
'_reg_stats.csv'

# Load data
df = pd.read_csv(INPUT_PATH)

# Keep only 4_Factor results
df = df[df['Model'] == '4_Factor']

# Compute summary statistics
stats = {
    'Mean_R_Squared': df['R_Squared'].mean(),
    'Mean_Adj_R_Squared': df['Adj_R_Squared'].mean(),
    'Mean_Alpha': df['Alpha'].mean(),
    'Mean_Alpha_t_stat': df['Alpha_t_stat'].mean(),
    'Mean_|Alpha_t_stat|': df['Alpha_t_stat'].abs().mean(),
    'Share_Significant_Alpha': (df['Alpha_t_stat'].abs() > 1.96).mean(),
    'Mean_Beta_R_MKT': df['Beta_R_MKT'].mean(),
    'Mean_t_stat_Beta_R_MKT_': df['t_stat_R_MKT'].mean(),
    'Mean_Beta_CSMB': df['Beta_CSMB'].mean(),
    'Mean_t_stat_Beta_CSMB': df['t_stat_CSMB'].mean(),
    'Mean_Beta_CMH': df['Beta_CMH'].mean(),
    'Mean_t_stat_Beta_CMH': df['t_stat_CMH'].mean(),
    'Mean_Beta_CIHML': df['Beta_CIHML'].mean(),
    'Mean_t_stat_Beta_CIHML': df['t_stat_CIHML'].mean()
}

# Create summary table
df_stats = pd.DataFrame([stats])

# Save
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df_stats.to_csv(OUTPUT_PATH, index=False)

print(df_stats)
