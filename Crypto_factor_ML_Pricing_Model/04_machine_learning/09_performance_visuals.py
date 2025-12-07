import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==================== CONFIGURATION ====================

# Input File (Daily returns from Step 8)
DAILY_RETURNS_PATH = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/results/daily_backtest_returns.csv'

# Output File for the Plot
OUTPUT_PLOT_PATH = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/results/cumulative_returns_plot.png'

# ==================== PLOTTING FUNCTION ====================

def plot_cumulative_performance(daily_returns_path: str, output_path: str):
    """Calculates and plots the cumulative returns of the portfolio vs. benchmark."""
    
    print("--- Starting Cumulative Returns Plotting ---")
    
    # Load daily returns (Assuming you saved the 'df_metrics' from Step 8 under this name)
    # Note: If the file was not explicitly saved, you might need to run the backtest script (Step 8) again
    # and adjust it to save 'daily_returns_df' to this path.
    try:
        df_daily = pd.read_csv(daily_returns_path, parse_dates=['date'])
    except FileNotFoundError:
        print(f"❌ ERROR: Daily returns file not found at {daily_returns_path}. Please check file saving in Step 8.")
        return

    # 1. Calculate Cumulative Returns
    # We use (1 + return).cumprod() to calculate the growth of a $1 investment.
    # We drop the first row if it contains NaN/non-numeric data from the previous steps' split.
    df_cumulative = df_daily[['PORTFOLIO_RETURN', 'MARKET_RETURN']].dropna()
    
    # Cumulative return calculation: (1 + R_t) * (1 + R_{t-1}) * ...
    # We assume log returns were converted to simple returns (e.g., e^log_return - 1)
    # If the returns are log returns, use np.exp(df.cumsum())
    # Assuming simple returns for plotting:
    df_cumulative['CUM_PORTFOLIO'] = (1 + df_cumulative['PORTFOLIO_RETURN']).cumprod()
    df_cumulative['CUM_MARKET'] = (1 + df_cumulative['MARKET_RETURN']).cumprod()
    
    # Reindex the cumulative returns using the date
    df_cumulative.set_index(df_daily['date'].iloc[df_cumulative.index], inplace=True)
    
    # 2. Plotting
    plt.figure(figsize=(14, 7))
    
    # Plot the portfolio and market cumulative returns
    df_cumulative['CUM_PORTFOLIO'].plot(label='RF Portfolio (Long Top 4)', linewidth=2.5, color='green')
    df_cumulative['CUM_MARKET'].plot(label='Market Benchmark (Equal-Weighted)', linewidth=2.5, color='red', linestyle='--')
    
    # Add context and formatting
    plt.title('Portfolio Cumulative Performance vs. Market Benchmark', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (Growth of $1)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    
    # 3. Save Output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    
    print(f"✅ Cumulative returns plot saved to {output_path}")

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    
    # Before running this, ensure the 'df_metrics' from Step 8 (which contains daily returns)
    # has been saved to the DAILY_RETURNS_PATH.
    
    # ******* ASSUMPTION CHECK *******
    # The previous script (08_backtesting_simulation.py) needs to save 'daily_returns_df' 
    # to the DAILY_RETURNS_PATH before running this script.
    # You might need to add a line to Step 8 like: daily_returns_df.to_csv(DAILY_RETURNS_PATH, index=True)
    # ********************************
    
    plot_cumulative_performance(DAILY_RETURNS_PATH, OUTPUT_PLOT_PATH)