import random
import math
import numpy as np
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
            return portfolio.get_volatility()
        elif variable == 'return':
            # Since we want to maximize return, we can minimize the negative return.
            return -portfolio.get_expected_return()
        elif variable == 'sharpe_ratio':
            # Similarly, maximize Sharpe ratio by minimizing its negative.
            return -portfolio.get_sharpe_ratio()
        else:
            raise ValueError("Unsupported objective variable.")

    def perturb_solution(self, solution):
        """
        Generate a new solution by slightly perturbing the weights of the given solution.
        Randomly adjust a pair of weights.
        """
        # Create a new Portfolio instance to store the perturbed solution
        new_solution = Portfolio(self.returns, self.corr, self.risk_free, self.min_weight)
        current_weights = solution.get_weights().copy()
        n_assets = len(current_weights)

        # Randomly pick two distinct indices
        idx1, idx2 = random.sample(range(n_assets), 2)

        # Decide on a random shift amount that does not exceed the smaller weight among the two
        max_shift = min(current_weights[idx1], current_weights[idx2])
        shift = random.uniform(0, max_shift)

        # Shift weights: increase one and decrease the other
        current_weights[idx1] += shift
        current_weights[idx2] -= shift

        # Ensure non-negativity and renormalize to sum to 1
        current_weights = np.maximum(current_weights, 0)
        current_weights = current_weights / current_weights.sum()

        # Set the new weights
        new_solution.set_weights(current_weights)

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
            # print("self.temperature: ", self.temperature, "self.cooling_rate: ", self.cooling_rate, "self.iteration: ", self.iteration, "delta: ", delta)
            # Decide whether to accept the new solution.
            if delta < 0:
                # New solution is better: accept it.
                self.current_solution = new_solution
                # Update best solution if necessary.
                if self.objective(new_solution, variable) < self.objective(self.best_solution, variable):
                    self.best_solution = new_solution
            else:
                # New solution is worse: accept it with probability exp(-delta / temperature)
                acceptance_probability = math.exp(-delta / self.temperature)
                if random.random() < acceptance_probability:
                    self.current_solution = new_solution
            
            # Record the current objective value for analysis.
            self.history.append(self.objective(self.current_solution, variable))
            
            # Cool down the temperature.
            self.temperature -= (self.temperature * self.cooling_rate)
            self.iteration += 1
        return None

    def get_best_solution(self):
        return self.best_solution

    def get_history(self):
        return self.history
