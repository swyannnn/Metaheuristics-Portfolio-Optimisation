import numpy as np
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
        for i in range(self.n_assets - 1):
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
    
    def set_weights(self, weights):
        # Normalize weights to sum to 1.
        self.weights = (weights/weights.sum())
        return None
    
    def set_expected_return(self):
        # Compute expected return as weighted sum of asset means.
        weighted = self.weights*self.description.loc['mean']

        # Round to 2 decimal places.
        self.expected_return = weighted.sum().round(2)
        return None
    
    def set_volatility(self):
        # Compute weighted standard deviation of assets.
        std = self.description.loc['std'].values

        # Compute covariance matrix of assets
        m1 = (self.weights*std).reshape(self.n_assets,1)
        m2 = m1.reshape(1,self.n_assets)

        # Compute volatility as square root of weighted sum of covariances.
        self.volatility = math.sqrt((m1*self.corr*m2).sum())
        return None
    
    def set_sharpe_ratio(self):
        # Compute Sharpe ratio as (expected return - risk-free rate) / volatility.
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