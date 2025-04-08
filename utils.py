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
    :param path: Directory containing CSV files
    :return: DataFrame with dates as index and stock prices as columns
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
    :param daily_prices: DataFrame of daily prices
    :return: DataFrame of monthly returns
    """
    # Convert index to datetime
    daily_prices.index = pd.to_datetime(daily_prices.index, utc=True)
    
    # use first trading day of month as timestamp
    monthly_prices = daily_prices.resample('ME').first() 

    # Calculate monthly returns 
    monthly_returns = monthly_prices.pct_change() 

    # Drop first row with NaN
    monthly_returns = monthly_returns.dropna()

    return monthly_returns*100

def setup_logger(config: DictConfig):
    """
    Setup logger for the application.
    :param config: Configuration object
    """
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
    """
    Generate the Efficient Frontier using the given monthly returns.
    :param monthly_returns: DataFrame of monthly returns
    :param risk_free_rate: Risk-free rate for performance calculations
    :param min_weight: Minimum weight for each asset in the portfolio
    :return: risk_frontier, return_frontier
    """
    # Calculate mean returns and covariance matrix
    mu = monthly_returns.mean() * 12
    S = monthly_returns.cov() * 12
    target_returns = np.linspace(mu.min(), mu.max(), num=50)

    # Generate Efficient Frontier
    efficient_frontier = []
    for target in target_returns:
        try:
            ef = EfficientFrontier(mu, S, weight_bounds=(min_weight, 1))
            ef.efficient_return(target)
            ret, vol, sr = ef.portfolio_performance(risk_free_rate=risk_free_rate, verbose=False)
            efficient_frontier.append((vol, ret))
        except Exception as e:
            continue

    # Convert to numpy array for easier indexing
    efficient_frontier = np.array(efficient_frontier)
    risk_frontier = efficient_frontier[:, 0]   # Volatility (risk)
    return_frontier = efficient_frontier[:, 1] # Expected return
    return risk_frontier, return_frontier

def save_benchmark_img(risk_frontier, return_frontier, best_history, overall_best_volatility, overall_best_return, save_name):
    """
    Save the benchmark image comparing the Efficient Frontier with heuristic best points.
    :param risk_frontier: Risk values for the Efficient Frontier
    :param return_frontier: Return values for the Efficient Frontier
    :param best_history: Best portfolios from the heuristic
    :param overall_best_volatility: Overall best portfolio volatility
    :param overall_best_return: Overall best portfolio return
    :param save_name: Name to save the image with
    """
    # Plot Efficient Frontier and overlay heuristic best points
    plt.figure(figsize=(8, 6))
    plt.plot(risk_frontier, return_frontier, label="Efficient Frontier", color="blue", linewidth=2, alpha=0.5)

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
        plt.scatter(best_history[:, 0], best_history[:, 1],
                    label=f"{save_name} Best Portfolios", color=color, marker="o", s=10, alpha=0.3)
        plt.plot(overall_best_volatility, overall_best_return,
                 label=f"{save_name} Overall Best Portfolio", color="green", marker="*", markersize=10)
    plt.xlabel("Volatility (%)")
    plt.ylabel("Expected Return (%)")
    plt.title(f"Efficient Frontier vs. {save_name}")
    plt.legend()
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{save_name}_{len(best_history)}.png", dpi=300)
    plt.close()

def calculate_tracking_error(portfolio_returns, benchmark_returns):
    """
    Calculate the Tracking Error (TE) as the standard deviation
    of the difference between portfolio returns and benchmark returns.
    :param portfolio_returns: Series of portfolio returns
    :param benchmark_returns: Series of benchmark returns
    :return: Tracking Error as a percentage
    """
    # ddof=1 This gives the sample standard deviation, 
    # which is the unbiased estimator when we calculating std from a sample of data. 
    # This is common in finance because typically we have a finite sample of historical returns.
    diff = portfolio_returns - benchmark_returns
    te_percentage = np.std(diff, ddof=1)   # Convert to percentage
    return te_percentage

def tracking_error_violin_plot(tracking_error_sa, tracking_error_ga, tracking_error_pso):
    """
    Plot the tracking error for different algorithms using a violin plot.
    :param tracking_error_sa: Tracking error for Simulated Annealing
    :param tracking_error_ga: Tracking error for Genetic Algorithm
    :param tracking_error_pso: Tracking error for Particle Swarm Optimization
    """
    data = [tracking_error_sa, tracking_error_ga, tracking_error_pso]

    # Define colors for each algorithm
    colors = ['purple', 'red', 'orange']
    base_labels = ['SA', 'GA', 'PSO']

    # Create the violin plot
    fig, ax = plt.subplots(figsize=(8, 6))
    vp = ax.violinplot(data, showmeans=True, showmedians=True)

    # Set the facecolor of each violin body
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(colors[i])
        body.set_edgecolor('black')
        body.set_alpha(0.3)

    # Set median color
    # vp['cmedians'] is a LineCollection, so we set its color directly.
    if hasattr(vp['cmedians'], 'set_color'):
        vp['cmedians'].set_color('black')
    else:
        # If it's a list, iterate over it.
        for m in vp['cmedians']:
            m.set_color('black')
    
    # Set mean color similarly, if available.
    if 'cmeans' in vp:
        if hasattr(vp['cmeans'], 'set_color'):
            vp['cmeans'].set_color('green')
        else:
            for m in vp['cmeans']:
                m.set_color('green')

    # Compute new x-tick labels including the mean and median values
    new_labels = []
    for i, dataset in enumerate(data):
        mean_val = np.mean(dataset)
        median_val = np.median(dataset)
        # Use Greek letter μ for mean and "Med" for median
        label = f"{base_labels[i]}\nμ={mean_val:.3f}%, Med={median_val:.3f}%"
        new_labels.append(label)
    
    # Set the new x-tick labels
    ax.set_xticks(np.arange(1, len(new_labels) + 1))
    ax.set_xticklabels(new_labels)
    
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Tracking Error (%)")
    ax.set_title("Distribution of Tracking Error for Portfolio Optimization")
    plt.tight_layout()
    plt.savefig("annotated_violin_plot.png", dpi=300)
    plt.close()

def save_weights_plot(tickers, ga_avg_weights, pso_avg_weights, sa_avg_weights):
    """
    Save the best portfolio image comparing the average weights of different algorithms.
    :param average_weights_ga: Average weights from Genetic Algorithm
    :param average_weights_sa: Average weights from Simulated Annealing
    :param average_weights_pso: Average weights from Particle Swarm Optimization
    :param save_name: Name to save the image with
    """
    # Create a bar plot for the average weights
    plt.figure(figsize=(10, 3))
    bar_width = 0.1
    x = np.arange(len(ga_avg_weights))
    benchmark_weights = np.array([1/len(ga_avg_weights)* 100] * len(ga_avg_weights))

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar(x - bar_width*1.5, benchmark_weights, bar_width, label='Benchmark', color='blue', alpha=0.3)
    rects2 = ax.bar(x - bar_width/2, ga_avg_weights, bar_width, label='GA', color='red', alpha=0.3)
    rects3 = ax.bar(x + bar_width/2, pso_avg_weights, bar_width, label='PSO', color='orange', alpha=0.3)
    rects4 = ax.bar(x + bar_width*1.5, sa_avg_weights, bar_width, label='SA', color='purple', alpha=0.3)

    ax.set_xlabel('Asset Name')
    ax.set_ylabel('Asset Weight (%)')
    ax.set_title('Average Asset Weight Distributions')
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.legend()

    plt.tight_layout()
    plt.savefig("average_asset_weights.png", dpi=300)
    plt.show()