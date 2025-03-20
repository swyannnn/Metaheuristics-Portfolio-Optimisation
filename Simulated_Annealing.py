import random
import numpy as np
import pandas as pd
from Portfolio import Portfolio

class Simulated_Annealing:
    def __init__(self, returns_df, corr_matrix, risk_free, min_weight, initial_temperature, cooling_rate):
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
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        
        # Initialize the current solution and best solution as a Portfolio instance.
        self.current_solution = Portfolio(self.returns, self.corr, self.risk_free, self.min_weight)
        self.best_solution = self.current_solution
        self.history = []  # Track objective values (fitness) over iterations
        self.iteration = 0

    def objective(self, portfolio, variable):
        """
        Compute the objective value for a given portfolio based on the chosen variable.
        For volatility, lower is better (minimize risk);
        for return or sharpe_ratio, higher is better (so we invert them for minimization).
        """
        if variable == 'volatility':
            return -portfolio.get_volatility()
        elif variable == 'return':
            # Since we want to maximize return, we can maximizing the return.
            return portfolio.get_expected_return()
        elif variable == 'sharpe_ratio':
            # Similarly, maximize Sharpe ratio by maximizing it.
            return portfolio.get_sharpe_ratio()
        else:
            raise ValueError("Unsupported objective variable.")

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

    def run(self, iterations, variable):
        """
        Run the Simulated Annealing algorithm for a number of iterations.
        :param iterations: Number of iterations to run.
        :param variable: The objective variable to optimize ('volatility', 'return', or 'sharpe_ratio').
        """
        for _ in range(iterations):
            # Generate a neighboring solution by perturbation.
            new_solution = self.perturb_solution(self.current_solution)
            
            # Calculate objective values for current and new solutions.
            current_obj = self.objective(self.current_solution, variable)
            new_obj = self.objective(new_solution, variable)
            
            # Compute the change in objective (delta)
            delta = new_obj - current_obj

            # Decide whether to accept the new solution if it is better or with some probability if it is worse.
            r = np.random.rand()
            if delta > 0 or r < np.exp(delta / self.temperature):
                self.current_solution = new_solution
                # Update best solution.
                if self.objective(new_solution, variable) > self.objective(self.best_solution, variable):
                    self.best_solution = new_solution
            else:
                # New solution is worse and not accepted.
                pass
            
            # Record the current objective value for analysis.
            self.history.append((self.objective(self.current_solution, variable), self.temperature))
            
            # Cool down the temperature.
            self.temperature -= (self.temperature * self.cooling_rate)
            self.iteration += 1
        return None

    def get_best_solution(self):
        return self.best_solution
    
    def get_analysis(self):
        """
        Returns a DataFrame containing:
        - iteration: iteration index
        - current_fitness: objective value at each iteration (from self.history)
        - best_fitness: the best objective value encountered so far up to that iteration.
        """
        best_so_far = []
        current_best = -float('inf')
        fitness_values = []
        temperatures = []
        for obj_val, temp in self.history:
            fitness_values.append(obj_val)
            temperatures.append(temp)
            if obj_val > current_best:
                current_best = obj_val
            best_so_far.append(current_best)
        
        df = pd.DataFrame({
            'iteration': list(range(len(self.history))),
            'current_fitness': fitness_values,
            'temperature': temperatures,
            'best_fitness': best_so_far
        })
        return df

