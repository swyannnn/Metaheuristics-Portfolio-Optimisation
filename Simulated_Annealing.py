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
        self.iteration = 0

    def perturb_solution(self, solution):
        """
        Generate a new solution by perturbing only two weights, while keeping the others unchanged.
        This strategy ensures that the two modified weights still sum to their original total,
        and that each weight is at least min_weight.
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

    def run(self, iterations, metric):
        """
        Run the Simulated Annealing algorithm for a number of iterations.
        :param iterations: Number of iterations to run.
        :param metric: The metric metric to optimize ('volatility', 'return', or 'sharpe_ratio').
        """
        self.max_iterations = iterations
        for _ in range(iterations):
            # Generate a neighboring solution by perturbation.
            new_solution = self.perturb_solution(self.current_solution)
            
            # Calculate metric values for current and new solutions.
            current_obj = Portfolio.evaluate_solution([self.current_solution])
            current_metric = current_obj[metric][0]
            new_obj = Portfolio.evaluate_solution([new_solution])
            new_metric = new_obj[metric][0]
            
            # Store the current metric in current_fitness
            self.current_fitness.append(current_metric)

            # Compute the change in metric (delta)
            delta = new_metric - current_metric

            # Decide whether to accept the new solution if it is better or with some probability if it is worse.
            r = np.random.rand()
            if self.best_solution is None or len(self.overall_best_metric) == 0:
                # First iteration: accept the first solution.
                self.best_solution = new_solution
                self.overall_best_metric.append(new_metric)
            else:
                # If the new solution is better, accept it. Or if the new solution is worse, accept it with a probability based on the temperature.
                if delta > 0 or r < np.exp(delta / self.temperature):
                    self.current_solution = new_solution
                    # Update best solution.
                    if new_metric > self.overall_best_metric[-1]:
                        # print("New best solution found at iteration", self.iteration, "with fitness", new_obj)
                        # print("New weights:", new_solution.get_weights())
                        # print(f"New {metric}:", new_solution.get_sharpe_ratio())
                        # print()
                        self.overall_best_metric.append(new_metric)
                        self.best_solution = new_solution
                    else: 
                        # New solution is worse but accepted.
                        self.overall_best_metric.append(self.overall_best_metric[-1])
                else:
                    # New solution is worse and not accepted.
                    self.overall_best_metric.append(self.overall_best_metric[-1])
            
            # Cool down the temperature.
            self.temperature = self.adjust_temperature()
            self.iteration += 1
        return None
    
    def get_best_solution(self):
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
            'iteration': range(self.iteration),
            'best_fitness': self.overall_best_metric,
            'current_fitness': self.current_fitness,
            'temperature': self.current_temperature
        })
        return analysis_df

