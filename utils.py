import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from pypfopt.efficient_frontier import EfficientFrontier

def load_stock_data(path):
    """
    Load and merge CSV files containing stock 'Adjusted Close' prices.
    Assumes each CSV has a 'Date' column as index.
    """
    files = os.listdir(path)
    df_list = []
    for file in files:
        # Read CSV, use Date as index, select 'Adjusted Close' column
        df = pd.read_csv(os.path.join(path, file), index_col='Date', parse_dates=True)[['Close']]
        
        # Rename 'Adj Close' column to stock name
        stock_name = file.replace(".csv", "")
        # df.rename(columns={'Adj Close': stock_name}, inplace=True)
        df.rename(columns={'Close': stock_name}, inplace=True)

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
    monthly_returns = monthly_prices.pct_change() 

    # Drop first row with NaN
    monthly_returns = monthly_returns.dropna()

    return monthly_returns

def setup_logger(config: DictConfig):
    logger.remove()
    logger.add(
        os.path.join(".hydra", "run.log"),
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        backtrace=True,
        diagnose=True,
        colorize=True,
        level="DEBUG",
    )
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}",
        backtrace=True,
        diagnose=True,
        colorize=True,
        level="DEBUG",
    )
    logger.info("------Configuration Details:")
    logger.info(OmegaConf.to_yaml(config))
    return logger

def gen_efficient_frontier(monthly_returns, risk_free_rate, min_weight):
    mu = monthly_returns.mean() * 12
    S = monthly_returns.cov() * 12
    target_returns = np.linspace(mu.min(), mu.max(), num=50)
    efficient_frontier = []
    for target in target_returns:
        try:
            ef = EfficientFrontier(mu, S, weight_bounds=(min_weight, 1))
            ef.efficient_return(target)
            ret, vol, sr = ef.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)
            efficient_frontier.append((vol, ret))
        except Exception as e:
            continue

    efficient_frontier = np.array(efficient_frontier)
    risk_frontier = efficient_frontier[:, 0]   # Volatility (risk)
    return_frontier = efficient_frontier[:, 1] # Expected return
    return risk_frontier, return_frontier

def save_benchmark_img(risk_frontier, return_frontier, best_history, overall_best_volatility, overall_best_return, save_name):
    # Plot Efficient Frontier and overlay heuristic best points
    plt.figure(figsize=(8, 6))
    plt.plot(risk_frontier, return_frontier, label="Efficient Frontier", color="blue", linewidth=2)

    # choose color based on the save_name
    try:
        color = save_name.lower()
        if save_name == "GA":
            color = "red"
        elif save_name == "PSO":
            color = "orange"
        elif save_name == "SA":
            color = "purple"
    except Exception as e:
        logger.error(f"Error determining color for {save_name}: {e}")
        color = "black"
        
    # Plot the best portfolios from the heuristic
    if len(best_history) > 0:
        print("len(GA_best_history):", len(best_history))
        plt.scatter(best_history[:, 0], best_history[:, 1],
                    label=f"{save_name} Best Portfolios", color=color, marker="o", s=10)
        plt.plot(overall_best_volatility, overall_best_return,
                 label=f"{save_name} Overall Best Portfolio", color="green", marker="*", markersize=10)
    plt.xlabel("Risk (Volatility)")
    plt.ylabel("Expected Return")
    plt.title(f"Efficient Frontier vs. {save_name}")
    plt.legend()
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{save_name}.png", dpi=300)
    plt.close()

def calculate_tracking_error(portfolio_returns, benchmark_returns):
    """
    Calculate the Tracking Error (TE) as the standard deviation
    of the difference between portfolio returns and benchmark returns.
    """
    diff = portfolio_returns - benchmark_returns
    return np.std(diff, ddof=1)
