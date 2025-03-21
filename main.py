import yaml
import math
import hydra
import os
import sys
from omegaconf import DictConfig, OmegaConf
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from Genetic_algorithm import Genetic_algorithm
from Particle_Swarm_Optimisation import Particle_Swarm_Optimisation
from Simulated_Annealing import Simulated_Annealing
from Ant_Colony_Optimisation import Ant_Colony_Optimisation
from Markowitz_Model import Markowitz_Model
from utils import load_stock_data, compute_monthly_returns

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

@hydra.main(config_path=os.path.join(os.path.dirname(__file__), "config"), config_name="config")
def main(config):
    # setup logger
    setup_logger(config)
    
    # Read global settings from config, or use defaults
    tickers = config.get("tickers", [])
    risk_free_rate = config.get("risk_free_rate", 0.02)
    min_weight = config.get("min_weight", 0.01)
    metric = config.get("metric", "sharpe_ratio")
    max_iterations = config.get("max_iterations", 1000)

    data_folder = config.get("stock_data_path", "data")
    data_folder = os.path.join(os.path.dirname(__file__), config.stock_data_path)
    # Load daily prices (using Adjusted Close if available; adjust load_stock_data accordingly)
    daily_prices = load_stock_data(data_folder)
    
    # Convert daily prices to monthly returns
    monthly_returns = compute_monthly_returns(daily_prices)
    
    # Compute correlation matrix from monthly returns
    corr = monthly_returns.corr()
    corr_matrix = corr.values

    # Set up TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir="logs")
    
    # --- Genetic Algorithm ---
    ga_config = config.get("Genetic_Algorithm", {})
    GA = Genetic_algorithm(monthly_returns, corr_matrix, risk_free_rate, min_weight, ga_config)
    GA.run(max_iterations, metric)
    best_ga = GA.overall_best
    print("Genetic Algorithm Best Portfolio:")
    print("Weights:", best_ga.get_weights())
    print(metric, "=", getattr(best_ga, metric))
    print("---------------")

    # Log PSO analysis metrics to TensorBoard
    try:
        ga_analysis = GA.get_analysis()
        for i, row in ga_analysis.iterrows():
            writer.add_scalar('Genetic_Algorithm/best_fitness', row['best_fitness'], i)
            writer.add_scalar('Genetic_Algorithm/mean_fitness', row['fitness_mean'], i)
    except Exception as e:
        print("PSO get_analysis() not available:", e)

    # # --- Particle Swarm Optimization ---
    # pso_config = config.get("Particle_Swarm_Optimization", {})
    # num_particles = pso_config.get("num_particles", 10)
    # inertia = pso_config.get("inertia", 0.5)
    # cognitive = pso_config.get("cognitive", 0.5)
    # social = pso_config.get("social", 0.5)
    # PSO = Particle_Swarm_Optimisation(monthly_returns, corr_matrix, risk_free_rate, num_particles, min_weight, inertia, cognitive, social)
    # PSO.run(max_iterations, metric)
    # best_pso = PSO.get_best_solution(metric)
    # print("Particle Swarm Optimization Best Portfolio:")
    # print("Weights:", best_pso.get_weights())
    # print(metric, "=", getattr(best_pso, metric))
    # print("---------------")

    # # Log PSO analysis metrics to TensorBoard
    # try:
    #     pso_analysis = PSO.get_analysis()
    #     for i, row in pso_analysis.iterrows():
    #         writer.add_scalar('Particle_Swarm_Optimisation/best_fitness', row['best_fitness'], i)
    #         writer.add_scalar('Particle_Swarm_Optimisation/mean_fitness', row['fitness_mean'], i)
    # except Exception as e:
    #     print("PSO get_analysis() not available:", e)
    
    # # --- Simulated Annealing ---
    # sa_config = config.get("Simulated_Annealing", {})
    # initial_temperature = sa_config.get("initial_temperature", 100)
    # cooling_rate = sa_config.get("cooling_rate", 0.003)
    # max_iterations_sa = sa_config.get("max_iterations", 1000)
    # SA = Simulated_Annealing(monthly_returns, corr_matrix, risk_free_rate, min_weight, initial_temperature, cooling_rate)
    # SA.run(max_iterations_sa, metric)
    # best_sa = SA.get_best_solution()
    # print("Simulated Annealing Best Portfolio:")
    # print("Weights:", best_sa.get_weights())
    # print(metric, "=", getattr(best_sa, metric))
    # print("---------------")

    # # Log SA analysis metrics to TensorBoard
    # try:
    #     sa_analysis = SA.get_analysis()
    #     for i, row in sa_analysis.iterrows():
    #         writer.add_scalar('Simulated_Annealing/best_fitness', row['best_fitness'], i)
    #         writer.add_scalar('Simulated_Annealing/current_fitness', row['current_fitness'], i)
    #         writer.add_scalar('Simulated_Annealing/temperature', row['temperature'], i)
    # except Exception as e:
    #     print("SA get_analysis() not available:", e)
    
    # # --- Ant Colony Optimization ---
    # aco_config = config.get("Ant_Colony_Optimization", {})
    # num_ants = aco_config.get("num_ants", 10)
    # ACO = Ant_Colony_Optimisation(monthly_returns, corr_matrix, risk_free_rate, min_weight, num_ants)
    # ACO.run(max_iterations, metric)
    # best_aco = ACO.get_best_solution(metric)
    # print("Ant Colony Optimization Best Portfolio:")
    # print("Weights:", best_aco.get_weights())
    # print(metric, "=", getattr(best_aco, metric))
    # print("---------------")

    # # Log ACO analysis metrics to TensorBoard
    # try:
    #     aco_analysis = ACO.get_analysis()
    #     for i, row in aco_analysis.iterrows():
    #         writer.add_scalar('Ant_Colony_Optimisation/best_fitness', row['best_fitness'], i)
    #         writer.add_scalar('Ant_Colony_Optimisation/mean_fitness', row['fitness_mean'], i)
    # except Exception as e:
    #     print("ACO get_analysis() not available:", e)
    
    # # --- Markowitz Optimization ---
    # # For Markowitz, compute expected returns and covariance matrix from monthly returns.
    # # Assume monthly_returns are in percentages; convert to decimals.
    # expected_returns = monthly_returns.mean().values / 100
    # # Compute covariance of monthly returns and scale appropriately.
    # cov_matrix = monthly_returns.cov().values / (100**2)
    # opt_weights, port_return, port_variance = Markowitz_Model(expected_returns, cov_matrix,
    #                                                                   target_return=None, allow_short=False)
    # port_volatility = math.sqrt(port_variance)
    # port_sharpe = (port_return - risk_free_rate) / port_volatility
    # print("Markowitz Optimization Portfolio:")
    # print("Weights:", opt_weights)
    # print("Expected Return:", port_return)
    # print("Volatility:", port_volatility)
    # print("Sharpe Ratio:", port_sharpe)
    # print("---------------")

    # # Log Markowitz analysis metrics to TensorBoard
    # try: 
    #     writer.add_scalar('Markowitz_Model/return', port_return)
    #     writer.add_scalar('Markowitz_Model/volatility', port_volatility)
    #     writer.add_scalar('Markowitz_Model/sharpe_ratio', port_sharpe)
    # except Exception as e:
    #     print("Markowitz get_analysis() not available:", e)

    writer.close()

if __name__ == "__main__":
    main()
