import hydra
import os
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from Genetic_algorithm import Genetic_algorithm
from Particle_Swarm_Optimisation import Particle_Swarm_Optimisation
from Simulated_Annealing import Simulated_Annealing
import utils

@hydra.main(config_path=os.path.join(os.path.dirname(__file__), "config"), config_name="config")
def main(config):
    # setup logger
    logger = utils.setup_logger(config)
    
    # Read global settings from config, or use defaults
    risk_free_rate = config.get("risk_free_rate", 0.02)
    min_weight = config.get("min_weight", 0.01)
    metric = config.get("metric", "sharpe_ratio")
    max_iterations = config.get("max_iterations", 1000)
    convergence_threshold = config.get("convergence_threshold", 1e-6)
    convergence_window = config.get("convergence_window", 100)

    data_folder = config.get("stock_data_path", "data")
    data_folder = os.path.join(os.path.dirname(__file__), config.stock_data_path)
    # Load daily prices using Adjusted Close
    daily_prices = utils.load_stock_data(data_folder)
    
    # Convert daily prices to monthly returns
    monthly_returns = utils.compute_monthly_returns(daily_prices)

    # Compute benchmark returns (equal-weighted portfolio)
    benchmark_weights = np.ones(monthly_returns.shape[1]) / monthly_returns.shape[1]
    benchmark_returns = monthly_returns.dot(benchmark_weights)
    
    # Compute correlation matrix from monthly returns
    corr = monthly_returns.corr()
    corr_matrix = corr.values

    # set up efficient frontier
    risk_frontier, return_frontier = utils.gen_efficient_frontier(monthly_returns, risk_free_rate, min_weight)

    # Set up TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir="logs")

    # Flags to control which algorithms to run, default to False
    run_ga = config.get("run_ga", False)
    run_pso = config.get("run_pso", False)
    run_sa = config.get("run_sa", False)

    # --- Genetic Algorithm ---
    if run_ga:
        start = time.time()
        ga_config = config.get("Genetic_Algorithm", {})
        GA = Genetic_algorithm(monthly_returns, corr_matrix, risk_free_rate, min_weight, ga_config)
        GA.run(max_iterations, metric, convergence_threshold, convergence_window)
        end = time.time()

        best_ga = GA.overall_best_portfolio
        best_ga_weights = best_ga.get_weights()  # e.g., an array of shape (n_assets,)
        portfolio_returns = monthly_returns.dot(best_ga_weights)
        ga_te = utils.calculate_tracking_error(portfolio_returns, benchmark_returns)

        GA_best_history = np.array(GA.best_history)
        utils.save_benchmark_img(risk_frontier, return_frontier, GA_best_history, best_ga.get_volatility(), best_ga.get_expected_return(), "GA")

        # Log GA analysis metrics to TensorBoard
        try:
            ga_analysis = GA.get_analysis()
            for i, row in ga_analysis.iterrows():
                writer.add_scalar('Genetic_Algorithm/best_fitness', row['best_fitness'], i)
                writer.add_scalar('Genetic_Algorithm/mean_fitness', row['mean_fitness'], i)
        except Exception as e:
            print("GA get_analysis() not available:", e)

        logger.info("Genetic Algorithm Best Portfolio:")
        logger.info(f"Convergence criteria met at iteration {len(ga_analysis)}")
        logger.info("Weights: {}".format(best_ga.get_weights()))
        logger.info("Sharpe Ratio: {}".format(best_ga.get_sharpe_ratio()))
        logger.info("Expected Return: {}".format(best_ga.get_expected_return()))
        logger.info("Volatility: {}".format(best_ga.get_volatility()))
        logger.info("Tracking Error: {}".format(ga_te))
        logger.info("Execution Time: {:.2f} seconds".format(end - start))
        logger.info("---------------")
        
    # --- Particle Swarm Optimization ---
    if run_pso:
        start = time.time()
        pso_config = config.get("Particle_Swarm_Optimization", {})
        PSO = Particle_Swarm_Optimisation(monthly_returns, corr_matrix, risk_free_rate, min_weight, pso_config)
        PSO.run(max_iterations, metric, convergence_threshold, convergence_window)
        end = time.time()

        best_pso = PSO.overall_best_portfolio
        best_pso_weights = best_pso.get_weights()  # e.g., an array of shape (n_assets,)
        portfolio_returns = monthly_returns.dot(best_pso_weights)
        pso_te = utils.calculate_tracking_error(portfolio_returns, benchmark_returns)

        PSO_best_history = np.array(PSO.best_history)
        utils.save_benchmark_img(risk_frontier, return_frontier, PSO_best_history, best_pso.get_volatility(), best_pso.get_expected_return(), "PSO")

        # Log PSO analysis metrics to TensorBoard
        try:
            pso_analysis = PSO.get_analysis()
            for i, row in pso_analysis.iterrows():
                writer.add_scalar('Particle_Swarm_Optimisation/best_fitness', row['best_fitness'], i)
                writer.add_scalar('Particle_Swarm_Optimisation/mean_fitness', row['mean_fitness'], i)
        except Exception as e:
            print("PSO get_analysis() not available:", e)

        logger.info("Particle Swarm Optimization Best Portfolio:")
        logger.info(f"Convergence criteria met at iteration {len(pso_analysis)}")
        logger.info("Weights: {}".format(best_pso.get_weights()))
        logger.info("Sharpe Ratio: {}".format(best_pso.get_sharpe_ratio()))
        logger.info("Expected Return: {}".format(best_pso.get_expected_return()))
        logger.info("Volatility: {}".format(best_pso.get_volatility()))
        logger.info("Tracking Error: {}".format(pso_te))
        logger.info("Execution Time: {:.2f} seconds".format(end - start))
        logger.info("---------------")
    
    # # --- Simulated Annealing ---
    if run_sa:
        start = time.time()
        sa_config = config.get("Simulated_Annealing", {})
        SA = Simulated_Annealing(monthly_returns, corr_matrix, risk_free_rate, min_weight, sa_config)
        SA.run(max_iterations, metric, convergence_threshold, convergence_window)
        end = time.time()

        best_sa = SA.get_best_solution()
        best_sa_weights = best_sa.get_weights()  # e.g., an array of shape (n_assets,)
        portfolio_returns = monthly_returns.dot(best_sa_weights)
        sa_te = utils.calculate_tracking_error(portfolio_returns, benchmark_returns)

        SA_best_history = np.array(SA.best_history)
        utils.save_benchmark_img(risk_frontier, return_frontier, SA_best_history, best_sa.get_volatility(), best_sa.get_expected_return(), "SA")

        # Log SA analysis metrics to TensorBoard
        try:
            sa_analysis = SA.get_analysis()
            for i, row in sa_analysis.iterrows():
                writer.add_scalar('Simulated_Annealing/best_fitness', row['best_fitness'], i)
                writer.add_scalar('Simulated_Annealing/current_fitness', row['current_fitness'], i)
                writer.add_scalar('Simulated_Annealing/temperature', row['temperature'], i)
        except Exception as e:
            print("SA get_analysis() not available:", e)

        logger.info("Simulated Annealing Best Portfolio:")
        logger.info(f"Convergence criteria met at iteration {len(sa_analysis)}")
        logger.info("Weights: {}".format(best_sa.get_weights()))
        logger.info("Sharpe Ratio: {}".format(best_sa.get_sharpe_ratio()))
        logger.info("Expected Return: {}".format(best_sa.get_expected_return()))
        logger.info("Volatility: {}".format(best_sa.get_volatility()))
        logger.info("Tracking Error: {}".format(sa_te))
        logger.info("Execution Time: {:.2f} seconds".format(end - start))
        logger.info("---------------")

    writer.close()

if __name__ == "__main__":
    for _ in range(1):
        main()
