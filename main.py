import yaml
import math

# Import your optimization classes and utility functions
from Genetic_algorithm import Genetic_algorithm
from Particle_Swarm_Optimisation import Particle_Swarm_Optimisation
from Simulated_Annealing import Simulated_Annealing
from Ant_Colony_Optimisation import Ant_Colony_Optimisation
from Markowitz_Model import Markowitz_Model
from utils import load_stock_data, compute_monthly_returns

def main():
    # Load configuration from config.yaml
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Read global settings from config
    tickers = config.get("tickers", [])
    risk_free_rate = config.get("risk_free_rate", 0.02)
    min_weight = config.get("min_weight", 0.01)
    metric = config.get("metric", "sharpe_ratio")
    max_iterations = config.get("max_iterations", 1000)
    data_folder = config.get("stock_data_path", "data")
    
    # Load daily prices (using Adjusted Close if available; adjust load_stock_data accordingly)
    daily_prices = load_stock_data(data_folder)
    
    # Convert daily prices to monthly returns
    monthly_returns = compute_monthly_returns(daily_prices)
    
    # Compute correlation matrix from monthly returns
    corr = monthly_returns.corr()
    corr_matrix = corr.values
    
    # --- Genetic Algorithm ---
    ga_config = config.get("Genetic_Algorithm", {})
    population_size = ga_config.get("population_size", 10)
    GA = Genetic_algorithm(monthly_returns, corr_matrix, risk_free_rate, population_size, min_weight)
    GA.run(max_iterations, metric)
    best_ga = GA.get_population(metric)
    print("Genetic Algorithm Best Portfolio:")
    print("Weights:", best_ga.get_weights())
    print(metric, "=", getattr(best_ga, metric))
    print("---------------")
    
    # --- Particle Swarm Optimization ---
    pso_config = config.get("Particle_Swarm_Optimization", {})
    num_particles = pso_config.get("num_particles", 10)
    inertia = pso_config.get("inertia", 0.5)
    cognitive = pso_config.get("cognitive", 0.5)
    social = pso_config.get("social", 0.5)
    PSO = Particle_Swarm_Optimisation(monthly_returns, corr_matrix, risk_free_rate, num_particles, min_weight, inertia, cognitive, social)
    PSO.run(max_iterations, metric)
    best_pso = PSO.get_best_solution(metric)
    print("Particle Swarm Optimization Best Portfolio:")
    print("Weights:", best_pso.get_weights())
    print(metric, "=", getattr(best_pso, metric))
    print("---------------")
    
    # --- Simulated Annealing ---
    sa_config = config.get("Simulated_Annealing", {})
    initial_temperature = sa_config.get("initial_temperature", 100)
    cooling_rate = sa_config.get("cooling_rate", 0.003)
    max_iterations_sa = sa_config.get("max_iterations", 1000)
    SA = Simulated_Annealing(monthly_returns, corr_matrix, risk_free_rate, min_weight, initial_temperature, cooling_rate)
    SA.run(max_iterations_sa, metric)
    best_sa = SA.get_best_solution()
    print("Simulated Annealing Best Portfolio:")
    print("Weights:", best_sa.get_weights())
    print(metric, "=", getattr(best_sa, metric))
    print("---------------")
    
    # --- Ant Colony Optimization ---
    aco_config = config.get("Ant_Colony_Optimization", {})
    num_ants = aco_config.get("num_ants", 10)
    ACO = Ant_Colony_Optimisation(monthly_returns, corr_matrix, risk_free_rate, num_ants)
    ACO.run(max_iterations, metric)
    best_aco = ACO.get_best_solution(metric)
    print("Ant Colony Optimization Best Portfolio:")
    print("Weights:", best_aco.get_weights())
    print(metric, "=", getattr(best_aco, metric))
    print("---------------")
    
    # --- Markowitz Optimization ---
    # For Markowitz, compute expected returns and covariance matrix from monthly returns.
    # Assume monthly_returns are in percentages; convert to decimals.
    expected_returns = monthly_returns.mean().values / 100
    # Compute covariance of monthly returns and scale appropriately.
    cov_matrix = monthly_returns.cov().values / (100**2)
    opt_weights, port_return, port_variance = Markowitz_Model(expected_returns, cov_matrix,
                                                                      target_return=None, allow_short=False)
    port_volatility = math.sqrt(port_variance)
    port_sharpe = (port_return - risk_free_rate) / port_volatility
    print("Markowitz Optimization Portfolio:")
    print("Weights:", opt_weights)
    print("Expected Return:", port_return)
    print("Volatility:", port_volatility)
    print("Sharpe Ratio:", port_sharpe)
    print("---------------")

if __name__ == "__main__":
    main()
