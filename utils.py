import os
import pandas as pd
import numpy as np
import math

def load_stock_data(path):
    """
    Load and merge CSV files containing stock 'Adjusted Close' prices.
    Assumes each CSV has a 'Date' column as index.
    """
    files = os.listdir(path)
    df_list = []
    for file in files:
        # Read CSV, use Date as index, select 'Adjusted Close' column
        df = pd.read_csv(os.path.join(path, file), index_col='Date', parse_dates=True)[['Adj Close']]
        
        # Rename 'Adj Close' column to stock name
        stock_name = file.replace(".csv", "")
        df.rename(columns={'Adj Close': stock_name}, inplace=True)

        # Append to list
        df_list.append(df)
    
    # Merge on Date
    df_merged = pd.concat(df_list, axis=1)

    # Forward fill missing values
    df_merged.ffill(inplace=True)

    return df_merged

def compute_monthly_returns(daily_prices):
    """
    Resample daily prices to monthly and compute percentage returns.
    """
    # Convert index to datetime
    daily_prices.index = pd.to_datetime(daily_prices.index, utc=True)
    
    # use first trading day of month as timestamp
    monthly_prices = daily_prices.resample('ME').first() 

    # Calculate monthly returns 
    monthly_returns = monthly_prices.pct_change() * 100

    # Drop first row with NaN
    monthly_returns = monthly_returns.dropna()

    return monthly_returns
