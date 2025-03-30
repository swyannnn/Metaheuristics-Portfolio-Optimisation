from pypfopt.efficient_frontier import EfficientFrontier

class Markowitz_Model:
    """
    A Markowitz (mean-variance) optimization model.
    
    Parameters:
    -----------
    monthly_returns : pd.DataFrame
        DataFrame of monthly returns for each asset (columns are tickers).
        If returns are provided in percentages (i.e. values > 1), they will be converted to decimals.
    corr_matrix : np.ndarray or pd.DataFrame
        Correlation matrix computed from monthly returns (stored but not directly used in optimization).
    risk_free_rate : float, default=0.02
        The annual risk free rate.
    min_weight : float, default=0.01
        Minimum allocation for each asset (used to set weight bounds).
    metric : str, default="sharpe_ratio"
        The optimization metric. Options are:
          - "sharpe_ratio": maximize Sharpe ratio.
          - "volatility": minimize portfolio volatility.
    max_iterations : int, default=1000
        Maximum iterations for the optimization algorithm (currently not directly used).
    """
    
    def __init__(self, monthly_returns, corr_matrix, risk_free_rate=0.02, min_weight=0.01, 
                 metric="sharpe_ratio", max_iterations=1000):
        self.returns = monthly_returns
        self.corr_matrix = corr_matrix  # stored for reference; not used directly in optimization
        self.risk_free_rate = risk_free_rate 
        self.min_weight = min_weight
        self.metric = metric
        self.max_iterations = max_iterations

        # Get asset tickers from the returns DataFrame
        self.tickers = self.returns.columns.tolist()
        self.n_assets = len(self.tickers)
        # Check feasibility of the minimum weight constraint
        if self.min_weight * self.n_assets > 1:
            raise ValueError("Infeasible constraint: min_weight too high for the number of assets.")
            
        # Annualize the expected returns and covariance matrix (assuming returns are monthly)
        self.mu = self.returns.mean() * 12
        self.S = self.returns.cov() * 12

    def run(self):
        """
        Run the optimization based on the selected metric.
        
        Returns:
        --------
        cleaned_weights : dict
            A dictionary mapping asset tickers to their optimal weights.
        port_return : float
            The annualized expected portfolio return.
        port_volatility : float
            The annualized portfolio volatility.
        port_sharpe : float
            The portfolio Sharpe ratio.
        """
        # Initialize the Efficient Frontier with weight bounds for each asset
        # The weight_bounds parameter enforces each weight to be at least min_weight (and at most 1)
        self.max_weight_per_asset = 1 - ((self.n_assets - 1) * self.min_weight)
        ef = EfficientFrontier(self.mu, self.S, weight_bounds=(self.min_weight, self.max_weight_per_asset))

        # Choose optimization objective based on the metric parameter
        if self.metric == "sharpe_ratio":
            # Optimize to maximize the Sharpe ratio
            _ = ef.max_sharpe(risk_free_rate=self.risk_free_rate)
        elif self.metric == "volatility":
            # Optimize to minimize volatility
            _ = ef.min_volatility()
        elif self.metric == "return":
            # Optimize to maximize return (not typical, but included for completeness)
            _ = ef.efficient_risk(0.4159)
        else:
            raise ValueError("Metric not supported. Use 'sharpe_ratio', 'return' or 'volatility'.")
        
        # Clean the weights (rounding small values to zero, etc.)
        cleaned_weights = ef.clean_weights()

        # Retrieve the portfolio performance metrics: expected return, volatility, and Sharpe ratio
        port_return, port_volatility, port_sharpe = ef.portfolio_performance(
            verbose=True, risk_free_rate=self.risk_free_rate
        )

        return cleaned_weights, port_return, port_volatility, port_sharpe
