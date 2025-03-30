import numpy as np
import pandas as pd
import math
import random

class Portfolio:
    def __init__(self, returns_df, corr_matrix, risk_free, min_weight):
        self.returns = returns_df 
        self.corr = corr_matrix
        self.description = self.returns.describe()
        self.n_assets = self.returns.shape[1]
        self.risk_free = risk_free
        self.min_weight = min_weight
        self.initialize_weights()
        self.set_expected_return()
        self.set_volatility()
        self.set_sharpe_ratio()
        return None
    
    def initialize_weights(self):
        # Total weight reserved for minimum allocations
        total_min = self.n_assets * self.min_weight

        # Remaining weight to distribute
        free_weight = 1.0 - total_min
        
        weights = []
        # For all but the last asset, allocate a random portion of the free_weight.
        for _ in range(self.n_assets - 1):
            # Random weight between 0 and free_weight
            w = random.uniform(0, free_weight)
            weights.append(w)
            free_weight -= w  # update free weight for remaining assets
        
        # Last asset gets the remaining free weight.
        weights.append(free_weight)
        
        # Add the minimum weight back to each allocation.
        weights = [w + self.min_weight for w in weights]
        
        # Convert to numpy array and assign to self.weights.
        self.weights = np.array(weights)
        return None
    
    def repair_weights(self):
        """
        Adjusts a candidate weights vector so that each weight is at least min_weight 
        and the total sums to 1.
        """
        n = len(self.weights)
        total_min = n * self.min_weight
        if total_min > 1:
            raise ValueError("Infeasible: the minimum weight is too high for the number of assets.")
        
        # Calculate the surplus above the minimum for each asset.
        surplus = np.maximum(self.weights - self.min_weight, 0)
        total_surplus = surplus.sum()
        
        # If no surplus exists, assign equal distribution for the free weight.
        if total_surplus == 0:
            return np.full(n, self.min_weight) + (1 - total_min) / n
        
        # Reallocate the free weight (1 - total_min) proportionally to the surplus.
        new_weights = self.min_weight + (surplus / total_surplus) * (1 - total_min)
        return new_weights
    
    def set_weights(self, weights):
        self.weights = weights 

        # Check if weights are valid
        tolerance = 1e-8
        # check if weights are less than min_weight
        if ((self.weights + tolerance) < self.min_weight).any():
            self.weights = self.repair_weights()
        # check if weights are greater than 1
        if (self.weights > 1).any():
            self.weights = self.repair_weights()
        # check if sum of weight is near 1
        if not np.isclose(self.weights.sum(), 1.0, atol=1e-2):
            # normalize weights to sum to 1
            self.weights = self.weights / self.weights.sum()
            
        # update expected return, volatility, and sharpe ratio
        self.set_expected_return()
        self.set_volatility()
        self.set_sharpe_ratio()

        return None
    
    def set_expected_return(self):
        # Calculate the monthly weighted expected return
        monthly_return = (self.weights * self.description.loc['mean']).sum()
        # Annualize by multiplying by 12
        self.expected_return = monthly_return * 12
    
    def set_volatility(self):
        std = self.description.loc['std'].values
        m1 = (self.weights * std).reshape(self.n_assets, 1)
        m2 = m1.reshape(1, self.n_assets)
        # Calculate the monthly portfolio variance using the correlation matrix
        monthly_variance = (m1 * self.corr * m2).sum()
        monthly_volatility = math.sqrt(monthly_variance)
        # Annualize volatility by multiplying by sqrt(12)
        self.volatility = monthly_volatility * math.sqrt(12)
    
    def set_sharpe_ratio(self):
        self.sharpe_ratio = (self.expected_return-self.risk_free)/self.volatility
        return None
    
    def get_weights(self):
        return self.weights
    
    def get_expected_return(self):
        return self.expected_return
    
    def get_volatility(self):
        return self.volatility
    
    def get_sharpe_ratio(self):
        return self.sharpe_ratio
    
    @staticmethod
    def evaluate_solution(array):
        # Convert swarm (array of Portfolio instances) into a DataFrame with computed metrics
        exp_returns = [p.get_expected_return() for p in array]
        volatilities = [p.get_volatility() for p in array]
        sharpe_ratios = [p.get_sharpe_ratio() for p in array]
        d = {'return': exp_returns, 'volatility': volatilities, 'sharpe_ratio': sharpe_ratios}
        df = pd.DataFrame(data=d)
        return df