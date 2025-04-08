import random
import numpy as np
import pandas as pd
from Portfolio import Portfolio

class Simulated_Annealing:
    def __init__(self, returns_df, corr_matrix, risk_free, min_weight, sa_config):
        """
        :param returns_df: DataFrame of returns (for Portfolio calculations)
        :param corr_matrix: Correlation matrix (or used to compute covariance)
        :param risk_free: Risk-free rate
        :param initial_temperature: Starting temperature for SA
        :param cooling_rate: Factor by which the temperature is multiplied each iteration (0 < cooling_rate < 1)
        """
        self.returns = returns_df
        self.corr = corr_matrix
        self.risk_free = risk_free 
        self.min_weight = min_weight
        self.initial_temperature = sa_config.get("initial_temperature", 1000)
        self.temperature = self.initial_temperature
        self.alpha = sa_config.get("alpha", 0.95)
        self.beta = sa_config.get("beta", 0.99)
        self.schedule = sa_config.get("schedule", "linear")
        
        # Initialize the current solution and best solution as a Portfolio instance.
        self.current_solution = Portfolio(self.returns, self.corr, self.risk_free, self.min_weight)
        self.best_solution = None
        self.current_temperature = []
        self.overall_best_metric = []
        self.current_fitness = []
        self.best_history = []
        self.iteration = 0

    def perturb_solution(self, solution):
        """
        Generate a new solution by perturbing only two weights, while keeping the others unchanged.
        This strategy ensures that the two modified weights still sum to their original total,
        and that each weight is at least min_weight.
        :param solution: Current Portfolio solution to be perturbed.
        :return: A new Portfolio instance with modified weights.
        """
        # Create a new Portfolio instance to store the perturbed solution.
        new_solution = Portfolio(self.returns, self.corr, self.risk_free, self.min_weight)
        current_weights = solution.get_weights().copy()
        n_assets = len(current_weights)

        # Randomly pick two distinct indices.
        idx1, idx2 = random.sample(range(n_assets), 2)
        
        # Compute the sum of the weights for these two assets.
        total = current_weights[idx1] + current_weights[idx2]
        
        # Ensure total is large enough to allow both weights to be at least min_weight.
        if total < 2 * self.min_weight:
            # Cannot perturb these weights; return the original solution.
            return new_solution
        
        # Sample new weight for asset idx1 uniformly from [min_weight, total - min_weight]
        new_weight_i = random.uniform(self.min_weight, total - self.min_weight)
        # The new weight for asset idx2 is then:
        new_weight_j = total - new_weight_i

        # Create new weights by copying the current ones.
        new_weights = current_weights.copy()
        new_weights[idx1] = new_weight_i
        new_weights[idx2] = new_weight_j

        # Set the new weights in the new solution (they already sum to the same total as before,
        # and the others are unchanged).
        new_solution.set_weights(new_weights)
        return new_solution

    def run(self, max_iterations, metric, convergence_threshold=1e-6, convergence_window=100):
        """
        Run the Simulated Annealing algorithm for a given number of iterations,
        stopping early if the performance metric converges.
        c
        :param max_iterations: Maximum number of iterations to run.
        :param metric: The performance metric to optimize ('volatility', 'return', or 'sharpe_ratio').
        :param convergence_threshold: Convergence threshold for improvement (e.g., 0.001).
        :param window: Number of consecutive iterations to consider for convergence.
        """
        self.max_iterations = max_iterations
        self.iteration = 0
        # Initialize current solution as the first solution
        # (Assuming self.current_solution is defined elsewhere; otherwise, set it to an initial portfolio)
        
        for i in range(max_iterations):
            # Evaluate current and new solutions
            new_solution = self.perturb_solution(self.current_solution)
            current_obj = Portfolio.evaluate_solution([self.current_solution])
            current_metric = current_obj[metric][0]
            new_obj = Portfolio.evaluate_solution([new_solution])
            new_metric = new_obj[metric][0]

            # Store the best portfolio's volatility and expected return.
            self.best_history.append((new_solution.get_volatility(),
                    new_solution.get_expected_return()))

            # Store the current metric in current_fitness
            self.current_fitness.append(current_metric)
            
            # Compute the change in metric
            delta = new_metric - current_metric
            
            # Acceptance rule: if new solution is better, or accept with a probability if worse.

            # If first iteration, initialize best_solution and best metric
            if self.iteration == 0:
                self.best_solution = self.current_solution
                self.overall_best_metric.append(current_metric)
            else:
                r = np.random.rand()
                if delta > 0 or r < np.exp(delta / self.temperature):
                    self.current_solution = new_solution
                    # Update overall best if new metric is better.
                    if new_metric > self.overall_best_metric[-1]:
                        self.best_solution = new_solution
                        self.overall_best_metric.append(new_metric)
                    else:
                        self.overall_best_metric.append(self.overall_best_metric[-1])
                else:
                    self.overall_best_metric.append(self.overall_best_metric[-1])
            
            # Update temperature and iteration count.
            self.temperature = self.adjust_temperature()
            self.iteration += 1

            # Check convergence: if improvement over the last 'window' iterations is below convergence_threshold.
            if len(self.overall_best_metric) >= convergence_window:
                recent_changes = np.abs(np.diff(self.overall_best_metric[-convergence_window:]))
                if np.all(recent_changes < convergence_threshold):
                    break
            
        return None
    
    def get_best_solution(self):
        """
        Return the best solution found during the simulated annealing process.
        """
        return self.best_solution

    def adjust_temperature(self):
        """
        Adjust the temperature based on the cooling schedule.
        """
        # append the current temperature to the list for analysis
        self.current_temperature.append(self.temperature)
        if self.schedule == "linear":
            # Linear cooling
            return self.temperature - (self.initial_temperature / (self.max_iterations))
        elif self.schedule == "geometric":
            # Geometric cooling
            return self.temperature * self.alpha
        elif self.schedule == "lundy_mees":
            # Lundy-Meeson cooling
            return self.temperature / (1 + self.beta * self.temperature)
        else:
            raise ValueError("Unsupported cooling schedule. Use 'linear', 'geometric' or 'lundy_meeson'.")

    def get_analysis(self):
        """
        Get the analysis of the simulated annealing process.
        Returns a DataFrame with the best and current fitness values at each iteration.
        """
        analysis_df = pd.DataFrame({
            'best_fitness': self.overall_best_metric,
            'current_fitness': self.current_fitness,
            'temperature': self.current_temperature
        })
        return analysis_df

