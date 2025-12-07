# This file fetches daily historical data for the CMC 20 Index for the year 2025

import os
from requests import Session
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time 

# ==================== CONFIGURATION ====================
load_dotenv()
API_KEY = os.getenv('CMC_API_KEY')
BASE_URL = 'https://pro-api.coinmarketcap.com/v3/index/cmc20-historical'

# Target Time range (Jan 1, 2025 to Dec 1, 2025)
START_DATE_TARGET = datetime(2025, 1, 1)
END_DATE_TARGET = datetime(2025, 12, 1) 

HEADERS = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': API_KEY,
}

# ==================== BATCHING AND FETCH LOGIC ====================
def fetch_index_data_batched():
    """Fetches the full year of CMC 20 Index data using small batches to bypass API limits."""
    
    session = Session()
    session.headers.update(HEADERS)

    all_records = []
    current_start_date = START_DATE_TARGET
    
    # Conservative settings to bypass the 400 (limit) and 429 (rate) errors
    BATCH_DAYS = 10 
    DAILY_INTERVAL = 'daily'
    SLEEP_SECONDS = 1.5 # Pause 1.5 seconds between each 10-day request

    print(f"Starting batched retrieval for {START_DATE_TARGET.date()} to {END_DATE_TARGET.date()}...")

    while current_start_date <= END_DATE_TARGET:
        
        count_to_request = BATCH_DAYS 
        remaining_days = (END_DATE_TARGET - current_start_date).days + 1
        
        if remaining_days <= 0:
            break
        if remaining_days < BATCH_DAYS:
            count_to_request = remaining_days

        time_start_iso = current_start_date.isoformat() + 'Z'

        params = {
            'time_start': time_start_iso,
            'count': count_to_request,
            'interval': DAILY_INTERVAL
        }
        
        print(f"  -> Requesting batch starting {current_start_date.date()} for {count_to_request} days...")

        try:
            response = session.get(BASE_URL, params=params)
            response.raise_for_status() 
            raw = response.json()
            
            # Add the retrieved data to the main list
            all_records.extend(raw['data'])
            
            # Move the start date forward by the number of days retrieved
            current_start_date += relativedelta(days=count_to_request)
            
            # Pause to respect the rate limit
            time.sleep(SLEEP_SECONDS) 

        except Exception as e:
            print(f"Error fetching batch data: {e}. Stopping retrieval.")
            break
            
    return all_records

# ==================== MAIN EXECUTION AND DATA PROCESSING ====================

if __name__ == "__main__":
    index_records = fetch_index_data_batched()

    if index_records:
        # Convert list of records (dictionaries) into a DataFrame
        df_index = pd.DataFrame(index_records)
        
        # Data Cleaning and Formatting
        df_index = df_index.rename(columns={'update_time': 'date', 'value': 'index_value'})
        df_index['date'] = pd.to_datetime(df_index['date'].str[:10])
        
        # Drop potential duplicate dates if any batch overlapped
        df_index.drop_duplicates(subset=['date'], inplace=True)
        df_index.sort_values('date', inplace=True)
        
        master_file = '/users/a1808/Desktop/IEDA 3330/Project/Crypto_factor_ML_Pricing_Model/data/cmc_index_2025_daily.csv'
        df_index[['date', 'index_value']].to_csv(master_file, index=False) 
        
        print(f"\n✅ SUCCESS: Retrieved and saved {len(df_index)} days of index data to {master_file}")
        print("\nFirst 5 rows:")
        print(df_index[['date', 'index_value']].head())
    else:
        print("\n❌ FAILED to retrieve any index data.")