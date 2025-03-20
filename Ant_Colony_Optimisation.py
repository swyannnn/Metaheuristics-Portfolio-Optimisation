import random
import pandas as pd
import numpy as np
from Portfolio import Portfolio

class Ant_Colony_Optimisation:
    def __init__(self, returns_df, corr_matrix, risk_free, min_weight, size, evaporation_rate=0.1, alpha=1, beta=2):
        """
        :param returns_df: DataFrame of returns, used to compute expected returns etc.
        :param corr_matrix: Correlation matrix (or can be combined with std to form covariance info)
        :param risk_free: Risk-free rate
        :param size: Number of ants (solutions) in the colony
        :param evaporation_rate: Rate at which pheromone evaporates
        :param alpha: Influence of pheromone
        :param beta: Influence of heuristic desirability
        """
        self.returns = returns_df
        self.corr = corr_matrix
        self.risk_free = risk_free
        self.min_weight = min_weight
        self.size = size
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        
        # Initialize pheromone vector for each asset.
        # We assume there are n_assets; initial pheromone can be uniform.
        sample_portfolio = Portfolio(self.returns, self.corr, self.risk_free, self.min_weight)
        self.n_assets = len(sample_portfolio.get_weights())
        self.pheromone = np.ones(self.n_assets)  # one pheromone level per asset
        
        # List to store colony solutions and tracking statistics.
        self.colony = []
        self.colony_best = []
        self.colony_fitness = []
        self.colony_mean  = []
        self.tree = []  # to store generations (iterations)
        return None
    
    def run(self, iterations, variable):
        # Initialize the colony for the first generation
        self.initialize()
        for i in range(iterations):
            new_colony = []
            # For each ant in the colony, construct a new solution and update the ant.
            for _ in range(self.size):
                weights = self.construct_solution()  # generate new weights based on pheromone & heuristic
                p = Portfolio(self.returns, self.corr, self.risk_free, self.min_weight)
                p.set_weights(weights)
                # Optionally, recalc portfolio metrics after updating weights if needed:
                p.set_expected_return()
                p.set_volatility()
                p.set_sharpe_ratio()
                new_colony.append(p)
            # Replace the old colony with the new one
            self.colony = new_colony
            
            # Evaluate fitness for the new colony
            self.set_fitness(variable)
            
            # Update pheromone based on new solutions' fitness
            self.update_pheromone()
            
            # Record generation information
            self.pass_generation(variable)
        return None
    
    def initialize(self):
        # Initialize the colony with random portfolios (solutions)
        self.colony = []
        self.tree = []
        for i in range(self.size):
            # Instead of completely random weights, we can use a probabilistic approach guided by pheromones.
            weights = self.construct_solution()
            p = Portfolio(self.returns, self.corr, self.risk_free, self.min_weight)
            p.set_weights(weights)
            self.colony.append(p)
        self.colony_df = self.to_table(self.colony)
        self.tree.append(self.colony)
        return None

    def construct_solution(self):
        """
        Constructs a solution (portfolio weights) probabilistically.
        We use the pheromone levels (raised to alpha) and a heuristic desirability (for instance, normalized expected return)
        raised to beta, to determine the probability of assigning weight to each asset.
        For simplicity, here we assume the heuristic is based on the asset's average return.
        """
        # Get heuristic desirability from returns (assuming self.returns has a row 'mean')
        heuristics = self.returns.describe().loc['mean'].values
        # Normalize heuristics to avoid scale issues:
        if heuristics.sum() != 0:
            heuristics = heuristics / heuristics.sum()
        else:
            heuristics = np.ones(self.n_assets) / self.n_assets
        
        # Compute probability for each asset: proportional to pheromone^alpha * heuristic^beta.
        prob = (self.pheromone ** self.alpha) * (heuristics ** self.beta)
        prob = prob / prob.sum()
        
        # To construct a weight vector that sums to 1, we sample weights proportionally.
        # One approach: assign each asset a weight proportional to its probability plus some random perturbation.
        weights = np.array([random.random() * p for p in prob])
        # Normalize weights:
        weights = weights / weights.sum()
        return weights
    
    def set_fitness(self, variable):
        """
        Evaluate each solution's performance. The variable can be 'volatility', 'return', or 'sharpe_ratio'.
        Lower volatility is better, while higher return or sharpe_ratio is better.
        """
        self.colony_df = self.to_table(self.colony)
        if variable == 'volatility':
            max_vol = self.colony_df[variable].max()
            self.colony_df.sort_values(by=variable, inplace=True, ascending=True)
            self.colony_df['fitness'] = max_vol - self.colony_df[variable] + 1
            self.colony_df['fitness'] = self.colony_df['fitness'] / self.colony_df['fitness'].sum()
        else:
            self.colony_df.sort_values(by=variable, inplace=True, ascending=False)
            self.colony_df['fitness'] = self.colony_df[variable] / self.colony_df[variable].sum()
        # Build cumulative selection probability for reference (not always used in ACO, but useful for analysis)
        self.colony_df['selection_prob'] = self.colony_df['fitness']
        for i in range(1, len(self.colony_df)):
            self.colony_df.loc[self.colony_df.index[i], 'selection_prob'] = (
                self.colony_df.loc[self.colony_df.index[i-1], 'selection_prob'] +
                self.colony_df.loc[self.colony_df.index[i], 'fitness']
            )
        return self.colony_df
    
    def update_pheromone(self):
        """
        Update pheromone levels based on the fitness of the colony.
        A simple update rule:
          pheromone = (1 - evaporation_rate) * pheromone + delta_pheromone
        where delta_pheromone is added based on the quality of solutions.
        Here, we add more pheromone to assets that appear in better portfolios.
        """
        # First, evaporate some pheromone:
        self.pheromone = (1 - self.evaporation_rate) * self.pheromone
        
        # Accumulate contributions from each ant:
        # For simplicity, we add pheromone proportional to the weight of an asset times the solution's fitness.
        for i, portfolio in enumerate(self.colony):
            weights = portfolio.get_weights()
            fitness = self.colony_df.iloc[i]['fitness']
            self.pheromone += fitness * weights  # add fitness-weighted contribution
        
        # Normalize pheromone (optional, to keep values in a reasonable range)
        self.pheromone = self.pheromone / self.pheromone.sum()
        return None
    
    def pass_generation(self, variable):
        # In ACO, we simply use the newly constructed colony as the next generation.
        self.tree.append(self.colony)
        best = self.colony[0]
        mean_fit = self.colony_df[variable].mean()
        if variable == 'volatility':
            best_fit = self.colony_df.sort_values(by=variable, ascending=True).head(1)[variable].iloc[0]
        else:
            best_fit = self.colony_df.sort_values(by=variable, ascending=False).head(1)[variable].iloc[0]
        self.colony_best.append(best)
        self.colony_fitness.append(best_fit)
        self.colony_mean.append(mean_fit)
        return None
    
    def to_table(self, array):
        exp_returns = [s.get_expected_return() for s in array]
        volatilities = [s.get_volatility() for s in array]
        sharpe_ratios = [s.get_sharpe_ratio() for s in array]
        d = {'return': exp_returns, 'volatility': volatilities, 'sharpe_ratio': sharpe_ratios}
        df = pd.DataFrame(data=d)
        return df
    
    def get_best_solution(self, variable):
        # Return the best portfolio in the current colony based on variable
        self.colony_df.sort_values(by=variable, inplace=True, ascending=(variable=='volatility'))
        idx = self.colony_df.head(1).index.values[0]
        return self.colony[idx]
    
    def get_tree(self):
        return self.tree
    
    def get_analysis(self):
        d = {'best_fitness': self.colony_fitness, 'fitness_mean': self.colony_mean}
        output = pd.DataFrame(data=d)
        return output
