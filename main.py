import hydra
import os
import time
from torch.utils.tensorboard import SummaryWriter
from Genetic_algorithm import Genetic_algorithm
from Particle_Swarm_Optimisation import Particle_Swarm_Optimisation
from Simulated_Annealing import Simulated_Annealing
from Markowitz_Model import Markowitz_Model
from utils import setup_logger
from utils import load_stock_data, compute_monthly_returns

@hydra.main(config_path=os.path.join(os.path.dirname(__file__), "config"), config_name="config")
def main(config):
    # setup logger
    logger = setup_logger(config)
    
    # Read global settings from config, or use defaults
    risk_free_rate = config.get("risk_free_rate", 0.02)
    min_weight = config.get("min_weight", 0.01)
    metric = config.get("metric", "sharpe_ratio")
    max_iterations = config.get("max_iterations", 1000)

    data_folder = config.get("stock_data_path", "data")
    data_folder = os.path.join(os.path.dirname(__file__), config.stock_data_path)
    # Load daily prices using Adjusted Close
    daily_prices = load_stock_data(data_folder)
    
    # Convert daily prices to monthly returns
    monthly_returns = compute_monthly_returns(daily_prices)
    
    # Compute correlation matrix from monthly returns
    corr = monthly_returns.corr()
    corr_matrix = corr.values

    # Set up TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir="logs")

    # Flags to control which algorithms to run
    # Set to True to run the corresponding algorithm, False to skip it
    run_ga = True
    run_pso = True
    run_sa = True
    run_markowitz = True
    
    # --- Genetic Algorithm ---
    if run_ga:
        start = time.time()
        ga_config = config.get("Genetic_Algorithm", {})
        GA = Genetic_algorithm(monthly_returns, corr_matrix, risk_free_rate, min_weight, ga_config)
        GA.run(max_iterations, metric)
        best_ga = GA.overall_best_portfolio
        end = time.time()
        print("Genetic Algorithm Best Portfolio:")
        logger.info("Genetic Algorithm Best Portfolio:")
        logger.info("Weights: {}".format(best_ga.get_weights()))
        logger.info("Sharpe Ratio: {}".format(best_ga.get_sharpe_ratio()))
        logger.info("Expected Return: {}".format(best_ga.get_expected_return()))
        logger.info("Volatility: {}".format(best_ga.get_volatility()))
        logger.info("Execution Time: {:.2f} seconds".format(end - start))
        logger.info("---------------")

        # Log GA analysis metrics to TensorBoard
        try:
            ga_analysis = GA.get_analysis()
            for i, row in ga_analysis.iterrows():
                writer.add_scalar('Genetic_Algorithm/best_fitness', row['best_fitness'], i)
                writer.add_scalar('Genetic_Algorithm/mean_fitness', row['mean_fitness'], i)
        except Exception as e:
            print("GA get_analysis() not available:", e)

    # --- Particle Swarm Optimization ---
    if run_pso:
        start = time.time()
        pso_config = config.get("Particle_Swarm_Optimization", {})
        PSO = Particle_Swarm_Optimisation(monthly_returns, corr_matrix, risk_free_rate, min_weight, pso_config)
        PSO.run(max_iterations, metric)
        best_pso = PSO.overall_best_portfolio
        end = time.time()
        logger.info("Particle Swarm Optimization Best Portfolio:")
        logger.info("Weights: {}".format(best_pso.get_weights()))
        logger.info("Sharpe Ratio: {}".format(best_pso.get_sharpe_ratio()))
        logger.info("Expected Return: {}".format(best_pso.get_expected_return()))
        logger.info("Volatility: {}".format(best_pso.get_volatility()))
        logger.info("Execution Time: {:.2f} seconds".format(end - start))
        logger.info("---------------")

        # Log PSO analysis metrics to TensorBoard
        try:
            pso_analysis = PSO.get_analysis()
            for i, row in pso_analysis.iterrows():
                writer.add_scalar('Particle_Swarm_Optimisation/best_fitness', row['best_fitness'], i)
                writer.add_scalar('Particle_Swarm_Optimisation/mean_fitness', row['mean_fitness'], i)
        except Exception as e:
            print("PSO get_analysis() not available:", e)
    
    # # --- Simulated Annealing ---
    if run_sa:
        start = time.time()
        sa_config = config.get("Simulated_Annealing", {})
        SA = Simulated_Annealing(monthly_returns, corr_matrix, risk_free_rate, min_weight, sa_config)
        SA.run(max_iterations, metric)
        best_sa = SA.get_best_solution()
        end = time.time()
        logger.info("Simulated Annealing Best Portfolio:")
        logger.info("Weights: {}".format(best_sa.get_weights()))
        logger.info("Sharpe Ratio: {}".format(best_sa.get_sharpe_ratio()))
        logger.info("Expected Return: {}".format(best_sa.get_expected_return()))
        logger.info("Volatility: {}".format(best_sa.get_volatility()))
        logger.info("Execution Time: {:.2f} seconds".format(end - start))
        logger.info("---------------")

        # Log SA analysis metrics to TensorBoard
        try:
            sa_analysis = SA.get_analysis()
            for i, row in sa_analysis.iterrows():
                writer.add_scalar('Simulated_Annealing/best_fitness', row['best_fitness'], i)
                writer.add_scalar('Simulated_Annealing/current_fitness', row['current_fitness'], i)
                writer.add_scalar('Simulated_Annealing/temperature', row['temperature'], i)
        except Exception as e:
            print("SA get_analysis() not available:", e)
    
    # --- Markowitz Optimization ---
    if run_markowitz:
        start = time.time()
        # For Markowitz, compute expected returns and covariance matrix from monthly returns.
        # Assume monthly_returns are in percentages; convert to decimals.
        markowitz = Markowitz_Model(monthly_returns, corr_matrix, risk_free_rate, min_weight, metric, max_iterations)
        optimal_weights, port_return, port_volatility, port_sharpe = markowitz.run()
        end = time.time()
        logger.info("Markowitz Optimization Portfolio:")
        logger.info("Weights: {}".format(optimal_weights))
        logger.info("Sharpe Ratio: {}".format(port_sharpe))
        logger.info("Expected Return: {}".format(port_return))
        logger.info("Volatility: {}".format(port_volatility))
        logger.info("Execution Time: {:.2f} seconds".format(end - start))
        logger.info("---------------")

        # Log Markowitz analysis metrics to TensorBoard
        try: 
            writer.add_scalar('Markowitz_Model/return', port_return)
            writer.add_scalar('Markowitz_Model/volatility', port_volatility)
            writer.add_scalar('Markowitz_Model/sharpe_ratio', port_sharpe)
        except Exception as e:
            print("Markowitz get_analysis() not available:", e)

    writer.close()

if __name__ == "__main__":
    for _ in range(10):
        main()
