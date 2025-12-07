# crypto-asset-pricing-ml
Factor models and machine learning for explaining and predicting cryptocurrency returns

## Project Overview

This project evaluates the **empirical performance of classical factor models and machine learning techniques** in explaining and predicting the cross-section of **cryptocurrency returns**.

Specifically, the study:
- Constructs **crypto-specific risk factors** (market, size, momentum, illiquidity)
- Tests asset pricing implications using **CAPM** and a **four-factor model**
- Applies **machine learning (Random Forest)** for return prediction
- Assesses **economic usefulness via backtesting**, rather than predictive accuracy alone

The project follows a full research pipeline: **data → factor construction → econometric testing → machine learning → portfolio backtesting**.

---

## Research Objective

> **Can risk factors and machine learning models explain and improve return outcomes in cryptocurrency markets, where traditional asset pricing performs poorly?**

---

## Project Structure

```text
.
├── 01_data_prep/
│   ├── 01_get_raw_index20_data.py
│   ├── 02_get_raw_individualcoins_data.py
│   ├── 03_calculate_metrics.py
│
├── 02_factor_construction/
│   ├── 01_factor_sort_CLMW.py
│   ├── 01_factor_sort_CMH.py
│
├── 03_asset_pricing_model/
│   ├── 01_capm_regression.py
│   ├── 02_4_factor_regression.py
│
├── machine_learning/
│   ├── 01_capm_regression.py
│   ├── 02_4_factor_regression.py
│   ├── 03_analysis_and_graphs.py
│   ├── 04_fama_macbeth_analysis.py
│
├── data/
│   ├── cmc_index_2025_daily.csv
│   ├── crypto_index_2025_daily_FULL.csv
│   ├── crypto_index_2025_daily_PREPARED.csv
│   ├── crypto_4_daily_factors_FINAL.csv
│   ├── ml_dataset_final.csv
│
└── README.md


### Step 1 – Data Preparation
1. **Market Index Data**  
   `01_get_raw_index20_data.py` – Fetches daily data for the CMC Crypto 20 Index (market-cap weighted)  
   → `cmc_index_2025_daily.csv`

2. **Individual Coin Data**  
   `02_get_raw_individualcoins_data.py` – Top 20 coins from the CMC 20 Index (price, volume, market cap)  
   → `crypto_index_2025_daily_FULL.csv`

3. **Feature & Metric Construction**  
   `03_calculate_metrics.py`  
   - Market factor (RMKT): log return of CMC 20 Index (risk-free rate = 0)  
   - Individual coin excess returns  
   - Sorting characteristics:  
     - `SIZE_SCORE` → market capitalization  
     - `AMIHUD_SCORE` → 7-day rolling `|R_{i,t}| / Volume_{i,t}` (illiquidity)  
     - `MOMENTUM_SCORE` → 28-day average log return  
   → `crypto_index_2025_daily_PREPARED.csv`

### Step 2 – Factor Construction
`01_factor_sort_CMH.py` (and CLMW variant)  
Uses **2×3 independent sorts** (Size × Characteristic) to form six portfolios per factor.

Constructed factors:
- **CSMB** – Crypto Small Minus Big (size premium)
- **CMH** – Crypto Momentum High Minus Low (trend-following)
- **CIHML** – Crypto Illiquidity High Minus Low (liquidity premium)

Outputs:  
`crypto_4_daily_factors_FINAL.csv` – daily factor returns  
`crypto_4_daily_factors_stats.csv` – mean, volatility, Sharpe ratios

### Step 3 – Linear Asset Pricing Models
#### CAPM
`01_capm_regression.py`

For each coin:
R_{i,t}^{ex} = α_i + β_{i,MKT} · R_{MKT,t} + ε_{i,t}

Stores R², alpha (t-stat), market beta (t-stat) for each coin.

#### Four-Factor Model
`02_4_factor_regression.py`

R_{i,t}^{ex} = α_i
            + β_{MKT} · R_{MKT,t}
            + β_{CSMB} · CSMB_t
            + β_{CMH} · CMH_t
            + β_{CIHML} · CIHML_t
            + ε_{i,t}​

Evaluates factor explanatory power and alpha reduction vs CAPM.  
Results saved in `regression_results.csv`.

### Step 4 – Machine Learning
#### Feature Engineering (`06_ml_feature_engineering.py`)
- **Target (Y)**: next-day excess return  
- **Features (X)**: size, momentum, illiquidity, volatility + factor exposures (RMKT, CSMB, CMH, CIHML)  
→ `ml_dataset_final.csv`

#### Model Training (`07_ml_model_training.py`)
- OLS (linear benchmark)
- Random Forest Regressor (non-linear)

Evaluation: out-of-sample R², MSE

#### Backtesting (`08_backtesting_simulation.py`)
- Rank coins by predicted returns → form long-only portfolio (rebalanced daily)
- Compare against equal-weighted market benchmark
- Output: `backtest_performance_summary.csv`

#### Visualization (`09_performance_visuals.py`)
Cumulative returns, drawdowns, risk-adjusted metrics.

### Key Insight
> **Low predictive accuracy ≠ low economic value**  
> Both OLS and Random Forest show weak out-of-sample R², yet the **ML-based portfolio consistently outperforms** the market benchmark on a risk-adjusted basis. This highlights the importance of **cross-sectional ranking** over point prediction accuracy.

## Requirements
```bash
Python >= 3.9
pandas
numpy
scikit-learn
statsmodels
matplotlib

## Disclaimer

This project is for academic and research purposes only. Results do not account for transaction costs and should not be interpreted as investment advice.

** Author **
Tsoi Ching Yi - Undergraduate Course Project — Financial Engineering

