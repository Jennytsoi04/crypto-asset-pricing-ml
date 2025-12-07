import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os

# ==================== CONFIGURATION ====================

# Input File
ML_DATASET_PATH = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/data/ml_dataset_final.csv'

# Output File
OUTPUT_MODEL_RESULTS = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/results/ml_prediction_metrics.csv'
OUTPUT_FEATURE_IMPORTANCE = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/results/rf_feature_importance.csv'


# --- Feature Definitions ---
# These are the X columns created in the previous step
FEATURES = [
    # Time-Varying Characteristics
    'SIZE_SCORE', 'AMIHUD_SCORE', 'MOMENTUM_SCORE', 
    # Static Risk Exposures (Betas)
    'Beta_R_MKT', 'Beta_CSMB', 'Beta_CMH', 'Beta_CIHML'
]
TARGET = 'Y_TARGET_RETURN'

# --- Time Series Split Configuration ---
# We will use the first 80% of data for training and the last 20% for testing.
TEST_SIZE_RATIO = 0.20 

# ==================== MODEL TRAINING AND EVALUATION ====================

def train_and_evaluate_model(df, model_name, model, features, target):
    """Trains a model on the historical data and predicts on the test set."""
    
    # Define X and Y
    X = df[features]
    y = df[target]

    # Time-Series Split (Crucial for preventing look-ahead bias)
    # We split based on the index (which is time-ordered)
    
    # 🟢 FIX 1: Calculate train_size FIRST
    train_size = int(len(df) * (1 - TEST_SIZE_RATIO))
    
    # Now split the data using the calculated train_size
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # 🟢 FIX 2: Define df_identifiers AFTER calculating train_size
    df_identifiers = df[['date', 'coin_id']].iloc[train_size:].reset_index(drop=True) 

    print(f"\n--- Training {model_name} ---")
    print(f"Train period length: {len(X_train)} | Test period length: {len(X_test)}")

    # Train Model
    model.fit(X_train, y_train)

    # Predict on the out-of-sample Test Set
    y_pred = model.predict(X_test)
    
    # --- Evaluation ---
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    
    # Store predictions and actuals for trading simulation later
    # Resetting index on X_test and y_test ensures clean merge with df_identifiers
    df_test_results = X_test.reset_index(drop=True).copy()
    df_test_results['Y_ACTUAL'] = y_test.reset_index(drop=True)
    df_test_results['Y_PREDICTED'] = y_pred
    
    # Merge identifiers
    df_test_results = df_test_results.merge(
        df_identifiers, 
        left_index=True, 
        right_index=True, 
        how='left'
    )

    return {
        'Model': model_name,
        'R2_Score': r2,
        'Mean_Squared_Error': mse,
        'Test_Set_Size': len(X_test)
    }, df_test_results

# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    
    # Load the consolidated ML dataset
    try:
        df_ml = pd.read_csv(ML_DATASET_PATH)
    except FileNotFoundError:
        print(f"❌ ERROR: ML dataset not found at {ML_DATASET_PATH}. Please ensure step 6 ran successfully.")
        exit()

    # Sort the data by date to ensure the time split is correct
    df_ml = df_ml.sort_values(by='date').reset_index(drop=True)

    all_results = []
    all_predictions = {}
    
    # --- 1. Linear Regression (Benchmark) ---
    lr_model = LinearRegression()
    lr_metrics, lr_preds = train_and_evaluate_model(df_ml, "OLS_Benchmark", lr_model, FEATURES, TARGET)
    all_results.append(lr_metrics)
    all_predictions['OLS_Benchmark'] = lr_preds

    # --- 2. Random Forest Regressor (ML Model) ---
    # Hyperparameters set for a good start (can be tuned later)
    rf_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1, min_samples_leaf=10)
    rf_metrics, rf_preds = train_and_evaluate_model(df_ml, "RandomForest", rf_model, FEATURES, TARGET)
    all_results.append(rf_metrics)
    all_predictions['RandomForest'] = rf_preds

        # --- Feature Importance Analysis (Random Forest) ---
    feature_importance_df = pd.DataFrame({
        'Feature': FEATURES,
        'Importance': rf_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # Save to CSV (no console printing)
    feature_importance_df.to_csv(OUTPUT_FEATURE_IMPORTANCE, index=False)


        # --- Final Output ---
    df_metrics = pd.DataFrame(all_results)
    
    # Save the prediction metrics
    os.makedirs(os.path.dirname(OUTPUT_MODEL_RESULTS), exist_ok=True)
    df_metrics.to_csv(OUTPUT_MODEL_RESULTS, index=False)
    
    print("\n" + "="*50)
    print("✅ MODEL PREDICTION METRICS")
    print(df_metrics.to_markdown(numalign="left", stralign="left"))
    print("="*50)

    # Save the actual and predicted returns for the best model (for trading simulation)
    # We will prioritize the Random Forest predictions for the next step (trading strategy)
    rf_preds.to_csv('/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/results/rf_test_predictions.csv', index=False)
    
    print("\nTraining complete. You can now compare the R2 and MSE of the OLS vs. Random Forest.")