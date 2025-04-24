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
        self.stopping_temperature = sa_config.get("stopping_temperature", 1e-6)
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
            return solution
        
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

    def run(self, metric, max_iterations, convergence_threshold=1e-6, convergence_window=100):
        """
        Run the Simulated Annealing algorithm for a given number of iterations,
        stopping early if the performance metric converges.
        """
        self.iteration = 0
        self.max_iterations = max_iterations

        # --- initialize diagnostics all to iteration 0 ---
        start_val = Portfolio.evaluate_solution([self.current_solution])[metric].iloc[0]
        self.best_solution          = self.current_solution
        self.overall_best_metric    = [start_val]
        self.current_fitness        = [start_val]
        self.current_temperature    = [self.temperature]

        convergence_met = False
        while not convergence_met and self.iteration < self.max_iterations:
            # 1) propose neighbor
            new_solution = self.perturb_solution(self.current_solution)

            # 2) evaluate current & new
            current_val = Portfolio.evaluate_solution([self.current_solution])[metric].iloc[0]
            new_val     = Portfolio.evaluate_solution([new_solution])[metric].iloc[0]

            # 3) compute delta > 0 when new is better
            if metric == 'volatility':
                delta = current_val - new_val
            else:
                delta = new_val - current_val

            # 4) metropolis acceptance
            if delta > 0 or np.random.rand() < np.exp(delta / self.temperature):
                self.current_solution = new_solution

            # 5) update global best (correctly handling minimization vs. maximization)
            last_best = self.overall_best_metric[-1]
            candidate = Portfolio.evaluate_solution([self.current_solution])[metric].iloc[0]
            improved = (
                (metric == 'volatility' and candidate < last_best) or
                (metric != 'volatility' and candidate > last_best)
            )
            if improved:
                self.best_solution = self.current_solution
                self.overall_best_metric.append(candidate)
                print(f"New Overall Best Portfolio Found at Iteration {self.iteration}!")
                print("Weights:", self.best_solution.get_weights())
                print(f"{metric} =", candidate)
                print("---------------")
            else:
                self.overall_best_metric.append(last_best)

            # Store the best portfolio's volatility and expected return.
            self.best_history.append((self.best_solution.get_volatility(),
                    self.best_solution.get_expected_return()))

            # 6) record diagnostics for this iteration
            self.current_fitness.append(current_val)

            # 7) cool down
            self.adjust_temperature()
            # append the current temperature to the list for analysis
            self.current_temperature.append(self.temperature)

            self.iteration += 1

            # 8) convergence checks
            if len(self.overall_best_metric) >= convergence_window:
                recent = np.abs(np.diff(self.overall_best_metric[-convergence_window:]))
                if np.all(recent < convergence_threshold):
                    convergence_met = True
            if self.temperature < self.stopping_temperature:
                convergence_met = True

        return None

    def adjust_temperature(self):
        """
        Adjust the temperature based on the cooling schedule.
        """
        if self.schedule == "linear":
            # Linear cooling
            self.temperature = self.temperature - (self.initial_temperature / (self.max_iterations))
        elif self.schedule == "geometric":
            # Geometric cooling
            self.temperature = self.temperature * self.alpha
        elif self.schedule == "lundy_mees":
            # Lundy-Meeson cooling
            self.temperature = self.temperature / (1 + self.beta * self.temperature)
        else:
            raise ValueError("Unsupported cooling schedule. Use 'linear', 'geometric' or 'lundy_mees'.")
        
        # Ensure temperature does not go below zero.
        # This is important to avoid negative temperatures in the exponential function.
        # Negative temperatures can lead to undefined behavior in the acceptance probability.
        if self.temperature < 0:
            self.temperature = 0
    
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

