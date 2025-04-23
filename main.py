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
    """
    Main function to run the optimization algorithms for portfolio management.
    """
    # setup logger
    logger = utils.setup_logger(config)
    
    # Read global settings from config, or use defaults
    tickers = config.get("tickers", None)
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
    runs = config.get("runs", 30)

    # --- Genetic Algorithm ---
    if run_ga:
        tracking_error_ga = []
        time_used_ga = []
        convergence_iterations_ga = []
        all_best_portfolio_weights_ga = []
        for i in range(runs):
            start = time.time()
            ga_config = config.get("Genetic_Algorithm", {})
            GA = Genetic_algorithm(monthly_returns, corr_matrix, risk_free_rate, min_weight, ga_config)
            GA.run(metric, convergence_threshold, convergence_window)
            end = time.time()

            best_ga = GA.overall_best_portfolio
            best_ga_volatility = best_ga.get_volatility()
            best_ga_expected_return = best_ga.get_expected_return()
            best_ga_sharpe_ratio = best_ga.get_sharpe_ratio()
            best_ga_weights = best_ga.get_weights()  # e.g., an array of shape (n_assets,)
            portfolio_returns = monthly_returns.dot(best_ga_weights)
            all_best_portfolio_weights_ga.append(best_ga_weights)

            ga_te = utils.calculate_tracking_error(portfolio_returns, benchmark_returns)

            GA_best_history = np.array(GA.best_history)
            utils.save_benchmark_img(risk_frontier, return_frontier, GA_best_history, best_ga_volatility, best_ga_expected_return, "GA")

            tracking_error_ga.append(ga_te)
            time_used_ga.append(end - start)
            convergence_iterations_ga.append(len(GA.best_history))

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
            logger.info("Weights: {}".format(best_ga_weights))
            logger.info("Sharpe Ratio: {:.3f}".format(best_ga_sharpe_ratio))
            logger.info("Expected Return: {:.3f}%".format(best_ga_expected_return))
            logger.info("Volatility: {:.3f}%".format(best_ga_volatility))
            logger.info("Tracking Error: {:.3f}%".format(ga_te))
            logger.info("Execution Time: {:.2f} seconds".format(end - start))
            logger.info("---------------\n")
        
    # --- Particle Swarm Optimization ---
    if run_pso:
        tracking_error_pso = []
        time_used_pso = []
        convergence_iterations_pso = []
        all_best_portfolio_weights_pso = []
        for i in range(runs):
            start = time.time()
            pso_config = config.get("Particle_Swarm_Optimization", {})
            PSO = Particle_Swarm_Optimisation(monthly_returns, corr_matrix, risk_free_rate, min_weight, pso_config)
            PSO.run(metric, convergence_threshold, convergence_window)
            end = time.time()

            best_pso = PSO.overall_best_portfolio
            best_pso_volatility = best_pso.get_volatility()
            best_pso_expected_return = best_pso.get_expected_return()
            best_pso_sharpe_ratio = best_pso.get_sharpe_ratio()
            best_pso_weights = best_pso.get_weights()  # e.g., an array of shape (n_assets,)
            portfolio_returns = monthly_returns.dot(best_pso_weights)
            all_best_portfolio_weights_pso.append(best_pso_weights)
            pso_te = utils.calculate_tracking_error(portfolio_returns, benchmark_returns)

            PSO_best_history = np.array(PSO.best_history)
            utils.save_benchmark_img(risk_frontier, return_frontier, PSO_best_history, best_pso_volatility, best_pso_expected_return, "PSO")

            tracking_error_pso.append(pso_te)
            time_used_pso.append(end - start)
            convergence_iterations_pso.append(len(PSO.best_history))

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
            logger.info("Weights: {}".format(best_pso_weights))
            logger.info("Sharpe Ratio: {:.3f}".format(best_pso_sharpe_ratio))
            logger.info("Expected Return: {:.3f}%".format(best_pso_expected_return))
            logger.info("Volatility: {:.3f}%".format(best_pso_volatility))
            logger.info("Tracking Error: {:.3f}%".format(pso_te))
            logger.info("Execution Time: {:.2f} seconds".format(end - start))
            logger.info("---------------\n")
        
    # # --- Simulated Annealing ---
    if run_sa:
        tracking_error_sa = []
        time_used_sa = []
        convergence_iterations_sa = []
        all_best_portfolio_weights_sa = []
        for i in range(runs):
            start = time.time()
            sa_config = config.get("Simulated_Annealing", {})
            SA = Simulated_Annealing(monthly_returns, corr_matrix, risk_free_rate, min_weight, sa_config)
            SA.run(metric, convergence_threshold, convergence_window)
            end = time.time()

            best_sa = SA.get_best_solution()
            best_sa_volatility = best_sa.get_volatility()
            best_sa_expected_return = best_sa.get_expected_return()
            best_sa_sharpe_ratio = best_sa.get_sharpe_ratio()
            best_sa_weights = best_sa.get_weights()  # e.g., an array of shape (n_assets,)
            all_best_portfolio_weights_sa.append(best_sa_weights)
            portfolio_returns = monthly_returns.dot(best_sa_weights)
            sa_te = utils.calculate_tracking_error(portfolio_returns, benchmark_returns)

            SA_best_history = np.array(SA.best_history)
            utils.save_benchmark_img(risk_frontier, return_frontier, SA_best_history, best_sa_volatility, best_sa_expected_return, "SA")

            tracking_error_sa.append(sa_te)
            time_used_sa.append(end - start)
            convergence_iterations_sa.append(len(SA.best_history))

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
            logger.info("Sharpe Ratio: {}".format(best_sa_sharpe_ratio))
            logger.info("Expected Return: {:.3f}%".format(best_sa_expected_return))
            logger.info("Volatility: {:.3f}%".format(best_sa_volatility))
            logger.info("Tracking Error: {:.3f}%".format(sa_te))
            logger.info("Execution Time: {:.2f} seconds".format(end - start))
            logger.info("---------------\n")

    # log the time used for each algorithm
    if run_ga:
        weights_array_ga = np.array(all_best_portfolio_weights_ga)
        average_weights_ga = np.mean(weights_array_ga, axis=0)
        std_weights_ga = np.std(weights_array_ga, axis=0)
        logger.info(f"Mean time used for GA: {np.mean(time_used_ga):.2f} seconds")
        logger.info(f"Standard Deviation time used for GA: {np.std(time_used_ga):.2f} seconds")
        logger.info(f"Mean tracking error for GA: {np.mean(tracking_error_ga):.2f}%")
        logger.info(f"Mean convergence iterations for GA: {np.mean(convergence_iterations_ga):.2f}")
        logger.info(f"Standard Deviation convergence iterations for GA: {np.std(convergence_iterations_ga):.2f}")

    if run_pso:
        weights_array_pso = np.array(all_best_portfolio_weights_pso)
        average_weights_pso = np.mean(weights_array_pso, axis=0)
        std_weights_pso = np.std(weights_array_pso, axis=0)
        logger.info(f"Mean time used for PSO: {np.mean(time_used_pso):.2f} seconds")
        logger.info(f"Standard Deviation time used for PSO: {np.std(time_used_pso):.2f} seconds")
        logger.info(f"Mean tracking error for PSO: {np.mean(tracking_error_pso):.2f}%")
        logger.info(f"Mean convergence iterations for PSO: {np.mean(convergence_iterations_pso):.2f}")
        logger.info(f"Standard Deviation convergence iterations for PSO: {np.std(convergence_iterations_pso):.2f}")

    if run_sa:
        weights_array_sa = np.array(all_best_portfolio_weights_sa)
        average_weights_sa = np.mean(weights_array_sa, axis=0)
        std_weights_sa = np.std(weights_array_sa, axis=0)
        logger.info(f"Mean time used for SA: {np.mean(time_used_sa):.2f} seconds")
        logger.info(f"Standard Deviation time used for SA: {np.std(time_used_sa):.2f} seconds")
        logger.info(f"Mean tracking error for SA: {np.mean(tracking_error_sa):.2f}%")
        logger.info(f"Mean convergence iterations for SA: {np.mean(convergence_iterations_sa):.2f}")
        logger.info(f"Standard Deviation convergence iterations for SA: {np.std(convergence_iterations_sa):.2f}")

    n_assets = len(tickers)
    for i in range(n_assets):
        if run_ga:
            logger.info(f"GA: {tickers[i]}: {average_weights_ga[i]:.2f} ± {std_weights_ga[i]:.2f}")
        if run_pso:
            logger.info(f"PSO: {tickers[i]}: {average_weights_pso[i]:.2f} ± {std_weights_pso[i]:.2f}")
        if run_sa:
            logger.info(f"SA: {tickers[i]}: {average_weights_sa[i]:.2f} ± {std_weights_sa[i]:.2f}")
    
    # Save the violin plot and weights plot
    if run_ga and run_pso and run_sa:
        utils.tracking_error_violin_plot(tracking_error_sa, tracking_error_ga, tracking_error_pso)
        utils.save_weights_plot(tickers, average_weights_ga, average_weights_sa, average_weights_pso)

    logger.info("All algorithms completed.")

    # Close the TensorBoard writer
    writer.close()

    # instructions to run the script
    print("To view the TensorBoard logs, run the following command in your terminal:")
    print("tensorboard --logdir=outputs --port=6006")
    print("Then open your web browser and go to http://localhost:6006/")
    print("To view the weights plot, check the 'weights_plot.png' file in the current directory.")
    print("To view the tracking error violin plot, check the 'tracking_error_violin_plot.png' file in the current directory.")

if __name__ == "__main__":
    for _ in range(1):
        main()
