import cvxpy as cp
import numpy as np

def Markowitz_Model(expected_returns, cov_matrix, target_return=None, allow_short=False):
    """
    Solve the Markowitz portfolio optimization problem.
    
    :param expected_returns: A numpy array of expected returns for each asset.
    :param cov_matrix: A numpy 2D array representing the covariance matrix of asset returns.
    :param target_return: (Optional) A desired minimum expected return for the portfolio.
    :param allow_short: (Boolean) Whether short selling is allowed (if False, weights >= 0).
    :return: Optimal weights, portfolio return, and portfolio variance.
    """

    # Check the inputs if it returns none
    if expected_returns is None:
        raise ValueError("The expected returns cannot be none.")
    
    if cov_matrix is None:
        raise ValueError("The covariance matrix cannot be none.")
    
    if expected_returns.ndim !=1:
        raise ValueError("Expected returns must be a one dimensional array.")

    n = len(expected_returns)
    
    # Decision variable: weights of assets in the portfolio
    w = cp.Variable(n)
    
    # Portfolio expected return and variance
    portfolio_return = expected_returns.T @ w
    portfolio_variance = cp.quad_form(w, cov_matrix)
    
    # Define constraints: sum of weights equals 1, and optionally no short selling
    constraints = [cp.sum(w) == 1]
    if not allow_short:
        constraints.append(w >= 0)
    
    # If a target return is specified, add it as a constraint
    if target_return is not None:
        constraints.append(portfolio_return >= target_return)
    
    # Set up the optimization problem: minimize variance (risk)
    problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
    
    # Solve the problem
    problem.solve()
    
    # Retrieve optimal weights, computed portfolio return, and variance
    optimal_weights = w.value
    opt_return = portfolio_return.value
    opt_variance = portfolio_variance.value
    
    return optimal_weights, opt_return, opt_variance

# Example usage:
# Let's say we have 5 assets:


expected_returns = np.array([0.12, 0.10, 0.15, 0.09, 0.11])
# A 5x5 covariance matrix (symmetric, positive semidefinite)
cov_matrix = np.array([
    [0.005, -0.010, 0.004, -0.002, 0.003],
    [-0.010, 0.040, -0.002, 0.004, -0.003],
    [0.004, -0.002, 0.023, 0.002, 0.001],
    [-0.002, 0.004, 0.002, 0.018, 0.005],
    [0.003, -0.003, 0.001, 0.005, 0.030]
])

# Optionally set a target return, e.g., 0.11, and disallow short selling:
optimal_weights, portfolio_return, portfolio_variance = Markowitz_Model(
    expected_returns, cov_matrix, target_return=0.11, allow_short=False
)

print("Optimal Weights:", optimal_weights)
print("Portfolio Expected Return:", portfolio_return)
print("Portfolio Variance:", portfolio_variance)
