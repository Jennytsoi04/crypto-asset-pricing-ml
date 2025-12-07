import pandas as pd
import numpy as np
import os

# ==================== CONFIGURATION ====================

# Input File (The Random Forest predictions saved in Step 7)
PREDICTIONS_PATH = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/results/rf_test_predictions.csv'

# Output File
OUTPUT_BACKTEST_RESULTS = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/results/backtest_performance_summary.csv'

# --- Strategy Parameters ---
TOP_N_COINS = 4  # Long the top 4 predicted performers (20% of 20 coins)
ANNUAL_TRADING_DAYS = 252 # Standard assumption for annualized metrics

# ==================== BACKTESTING FUNCTION ====================

def run_backtest(predictions_path: str, n_coins: int, annual_days: int):
    """
    Runs an equal-weighted long-only trading simulation based on predicted returns,
    and calculates performance metrics.
    """
    print("--- Starting Backtesting Simulation ---")
    
    # 1. Load Data
    try:
        df = pd.read_csv(predictions_path, parse_dates=['date'])
    except FileNotFoundError:
        print(f"❌ ERROR: Prediction file not found at {predictions_path}.")
        return

    # 2. Daily Ranking and Portfolio Formation
    
    # Group the predictions by day and apply a ranking (descending order)
    # The 'rank' is done on the predicted return (Y_PREDICTED)
    df['PREDICT_RANK'] = df.groupby('date')['Y_PREDICTED'].rank(ascending=False, method='first')
    
    # Identify coins to be included in the Long portfolio (top N)
    df['LONG_SIGNAL'] = df['PREDICT_RANK'] <= n_coins
    
    # 3. Calculate Portfolio Returns
    
    # Filter for the long-only portfolio coins
    df_portfolio = df[df['LONG_SIGNAL']].copy()
    
    # Calculate the equal-weighted daily return of the portfolio
    # (Sum of actual returns for the selected coins) / (Number of coins selected)
    df_daily_returns = df_portfolio.groupby('date')['Y_ACTUAL'].mean().rename('PORTFOLIO_RETURN')
    
    print(f"Portfolio formed successfully. Total trading days simulated: {len(df_daily_returns)}")

    # 4. Calculate Market Benchmark Return (Average of all coins on that day)
    # The market return in excess is the average excess return of all 20 coins
    df_market_return = df.groupby('date')['Y_ACTUAL'].mean().rename('MARKET_RETURN')
    
    df_metrics = pd.DataFrame(df_daily_returns).join(df_market_return)
    
    # 5. Calculate Performance Metrics
    
    # Daily Mean and Standard Deviation
    mean_port = df_metrics['PORTFOLIO_RETURN'].mean()
    std_port = df_metrics['PORTFOLIO_RETURN'].std()
    
    mean_mkt = df_metrics['MARKET_RETURN'].mean()
    std_mkt = df_metrics['MARKET_RETURN'].std()
    
    # Annualized Sharpe Ratio = (Mean Daily Return / Std Dev Daily Return) * sqrt(Trading Days)
    sharpe_port = (mean_port / std_port) * np.sqrt(annual_days)
    sharpe_mkt = (mean_mkt / std_mkt) * np.sqrt(annual_days)
    
    results = pd.DataFrame({
        'Metric': ['Daily Mean Return', 'Daily Std Dev', 'Annualized Return', 'Annualized Volatility', 'Sharpe Ratio'],
        'Portfolio': [
            mean_port, 
            std_port,
            mean_port * annual_days,
            std_port * np.sqrt(annual_days),
            sharpe_port
        ],
        'Market_Benchmark': [
            mean_mkt, 
            std_mkt,
            mean_mkt * annual_days,
            std_mkt * np.sqrt(annual_days),
            sharpe_mkt
        ]
    })
    
    return results, df_metrics

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    
    # Run the backtest
    performance_df, daily_returns_df = run_backtest(PREDICTIONS_PATH, TOP_N_COINS, ANNUAL_TRADING_DAYS)
    
    # Save results
    os.makedirs(os.path.dirname(OUTPUT_BACKTEST_RESULTS), exist_ok=True)
    performance_df.to_csv(OUTPUT_BACKTEST_RESULTS, index=False)

    # 🟢 FIX: Save the daily returns DataFrame for the plotting script (Step 9)
    DAILY_RETURNS_PATH = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/results/daily_backtest_returns.csv'
    daily_returns_df.to_csv(DAILY_RETURNS_PATH, index=True) # index=True saves the 'date' column
    
    print("\n" + "="*50)
    print("✅ BACKTEST PERFORMANCE SUMMARY (Random Forest Signal)")
    print(performance_df.to_markdown(numalign="left", stralign="left"))
    print("="*50)

    # Proceed to final analysis
    print("\nNext step: Analyze the Sharpe Ratio and final trade-off between ML complexity and performance.")