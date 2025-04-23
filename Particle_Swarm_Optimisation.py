import pandas as pd
import numpy as np
import copy
from Portfolio import Portfolio

class Particle_Swarm_Optimisation:
    def __init__(self, returns_df, corr_matrix, risk_free, min_weight, pso_config):
        """
        :param returns_df: DataFrame of returns (used by Portfolio to compute metrics)
        :param corr_matrix: Correlation matrix (or covariance info) between assets
        :param risk_free: Risk-free rate
        :param size: Number of particles (candidate portfolios)
        :param inertia: Inertia weight controlling previous velocity influence
        :param cognitive: Cognitive coefficient (personal best influence)
        :param social: Social coefficient (global best influence)
        """
        self.returns = returns_df
        self.corr = corr_matrix
        self.risk_free = risk_free 
        self.min_weight = min_weight
        self.swarm_size = pso_config.get("num_particles", 10)
        self.inertia = pso_config.get("inertia", 0.65)
        self.cognitive = pso_config.get("cognitive", 2)
        self.social = pso_config.get("social", 2)

        # Create a swarm of particles (each a Portfolio instance)
        self.swarm = []
        for _ in range(self.swarm_size):
            particle = Portfolio(self.returns, self.corr, self.risk_free, self.min_weight)
            self.swarm.append(particle)

        # Initialize velocities for each particle as a numpy array with same dimensions as weights.
        self.n_assets = len(self.swarm[0].get_weights())
        self.velocities = [np.zeros(self.n_assets) for _ in range(self.swarm_size)]

        # Initialize personal best for each particle
        self.personal_best = [particle.get_weights().copy() for particle in self.swarm]
        self.personal_best_fitness = [None] * self.swarm_size

        # Evaluate initial fitness based on the chosen metric (e.g., 'volatility', 'return', 'sharpe_ratio')
        # Higher fitness is better. This will add a 'fitness' column to the swarm DataFrame.
        self.swarm_df = Portfolio.evaluate_solution(self.swarm)
        
        # Set the fitness metric to be maximized (e.g., 'sharpe_ratio'), this is the default.
        # The fitness metric can be changed later by calling set_fitness() with a different metric.
        self.set_fitness(metric='sharpe_ratio')
        self.current_best_index = self.swarm_df['fitness'].idxmax()
        self.current_best_weight = self.swarm[self.current_best_index].get_weights().copy()

        # To track evolution over iterations
        self.best_overall_fitness = []
        self.swarm_mean = []
        self.best_history = []

    def run(self, metric, convergence_threshold=1e-6, convergence_window=100):
        """
        Run the PSO algorithm for a specified number of iterations.
        :param metric: Metric to optimize ('volatility', 'return', 'sharpe_ratio').
        :param convergence_threshold: Threshold for convergence.
        :param convergence_window: Number of iterations to consider for convergence.
        """
        # Initialize overall best tracking.
        if metric == 'volatility':
            self.overall_best_portfolio_metric = float('inf')
        else:
            self.overall_best_portfolio_metric = -float('inf')
        self.overall_best_portfolio = None

        convergence_met = False
        iteration = 1
        while not convergence_met:
            # Evaluate fitness for each particle.
            self.set_fitness(metric)
            # immediately refresh gbest
            best_idx = self.swarm_df['fitness'].idxmax()
            self.current_best_weight = self.swarm[best_idx].get_weights().copy()

            # Update velocities and positions.
            self.update_velocity()
            self.update_position()

            # Get the best portfolio from the current iteration.
            current_best_portfolio = self.get_best_solution(metric)

            # Store the best portfolio's volatility and expected return.
            self.best_history.append((current_best_portfolio.get_volatility(),
                    current_best_portfolio.get_expected_return()))

            if metric == 'volatility':
                current_metric = current_best_portfolio.get_volatility()
            elif metric == 'return':
                current_metric = current_best_portfolio.get_expected_return()
            elif metric == 'sharpe_ratio':
                current_metric = current_best_portfolio.get_sharpe_ratio()
            else:
                raise ValueError("Unsupported metric. Supported metrics are: volatility, return, sharpe_ratio")

            # Store the mean fitness of the swarm.
            self.swarm_mean.append(self.swarm_df['fitness'].mean())

            # Update overall best if the current metric is better.
            if self.overall_best_portfolio is None:
                self.overall_best_portfolio = copy.deepcopy(current_best_portfolio)
                self.overall_best_portfolio_metric = current_metric
                self.best_overall_fitness.append(current_metric)
            else:
                if (metric == 'volatility' and current_metric < self.overall_best_portfolio_metric) or \
                (metric != 'volatility' and current_metric > self.overall_best_portfolio_metric):
                    self.overall_best_portfolio = copy.deepcopy(current_best_portfolio)
                    self.overall_best_portfolio_metric = current_metric
                    self.current_best_weight = self.overall_best_portfolio.get_weights().copy()
                    self.best_overall_fitness.append(current_metric)
                    print(f"New Overall Best Portfolio Found at Iteration {iteration}!")
                    print("Weights:", self.overall_best_portfolio.get_weights())
                    print(f"{metric} =", self.overall_best_portfolio_metric)
                    print("---------------")
                else:
                    self.best_overall_fitness.append(self.best_overall_fitness[-1])
                # Check convergence: if improvement over the last 'window' iterations is below convergence_threshold.
            
            if len(self.best_overall_fitness) >= convergence_window:
                recent_changes = np.abs(np.diff(self.best_overall_fitness[-convergence_window:]))
                if np.all(recent_changes < convergence_threshold):
                    convergence_met = True
            iteration += 1
        return None

    # def set_fitness(self, metric):
    #     """
    #     Evaluate fitness for each particle.
    #     For 'volatility', lower is better; for 'return' or 'sharpe_ratio', higher is better.
    #     This function will add a 'fitness' column to the swarm DataFrame.

    #     :param metric: Metric to optimize ('volatility', 'return', 'sharpe_ratio').
    #     :return: None
    #     """
    #     # If the metric is not the allowed metrics, raise an error.
    #     if metric not in ['volatility','return','sharpe_ratio']:
    #         raise ValueError("Invalid fitness metric '{metric}', must be one of volatility, return, sharpe_ratio")
    #     self.swarm_df = Portfolio.evaluate_solution(self.swarm)
    #     if metric == 'volatility':
    #         # Lower volatility is better.
    #         max_vol = self.swarm_df[metric].max()
    #         self.swarm_df.sort_values(by=metric, inplace=True, ascending=True)
    #         self.swarm_df['fitness'] = max_vol - self.swarm_df[metric] + 1
    #     else:
    #         # For return or sharpe_ratio, higher is better.
    #         self.swarm_df.sort_values(by=metric, inplace=True, ascending=False)
    #         self.swarm_df['fitness'] = self.swarm_df[metric]
        
    #     # Update personal best fitness if not already set.
    #     for i in range(self.swarm_size):
    #         current_fitness = self.swarm_df.iloc[i]['fitness']
    #         if self.personal_best_fitness[i] is None or current_fitness > self.personal_best_fitness[i]:
    #             self.personal_best[i] = self.swarm[i].get_weights().copy()
    #             self.personal_best_fitness[i] = current_fitness

    def set_fitness(self, metric):
        """
        Evaluate fitness for each particle.
        For 'volatility', lower is better; for 'return' or 'sharpe_ratio', higher is better.
        This function will add a 'fitness' column to the swarm DataFrame and update each
        particle's personal best if their fitness improves.

        :param metric: Metric to optimize ('volatility', 'return', 'sharpe_ratio').
        """
        # 1. Validate metric
        allowed = ['volatility', 'return', 'sharpe_ratio']
        if metric not in allowed:
            raise ValueError(
                f"Invalid fitness metric '{metric}', must be one of {allowed}"
            )

        # 2. Evaluate all particles and reset index to ensure 0..N-1 alignment
        df = Portfolio.evaluate_solution(self.swarm).reset_index(drop=True)

        # 3. Compute fitness values
        if metric == 'volatility':
            # Lower volatility is better → shift so larger=fitter
            max_vol = df['volatility'].max()
            df['fitness'] = max_vol - df['volatility'] + 1
        else:
            # For 'return' or 'sharpe_ratio', higher is directly fitter
            df['fitness'] = df[metric]

        # 4. Store DataFrame for later inspection
        self.swarm_df = df

        # 5. Update personal bests (aligned by row index = particle index)
        for i in range(self.swarm_size):
            current_f = df.loc[i, 'fitness']
            # If first time, or improved fitness, update p-best
            if (
                self.personal_best_fitness[i] is None or
                current_f > self.personal_best_fitness[i]
            ):
                self.personal_best[i] = self.swarm[i].get_weights().copy()
                self.personal_best_fitness[i] = current_f

    def update_velocity(self):
        """
        Update the velocity for each particle using the PSO formula:
            v = inertia * v + cognitive * r1 * (pbest - position) + social * r2 * (gbest - position)
        where r1 and r2 are random numbers between 0 and 1.
        """
        for i in range(self.swarm_size):
            current_position = self.swarm[i].get_weights()
            r1 = np.random.rand(self.n_assets)
            r2 = np.random.rand(self.n_assets)
            inertia_component = self.inertia * self.velocities[i]
            cognitive_component = self.cognitive * r1 * (self.personal_best[i] - current_position)
            social_component = self.social * r2 * (self.current_best_weight - current_position)
            self.velocities[i] = (inertia_component + cognitive_component + social_component)
        return None

    def update_position(self):
        """
        Update each particle's position (portfolio weights) using:
            new_position = current_position + velocity
        Then re-normalize the weights to sum to 1 and enforce that each weight is at least self.min_weight.
        """
        for i in range(self.swarm_size):
            # Update position based on velocity
            current_position = self.swarm[i].get_weights()
            new_position = current_position + self.velocities[i]

            # update weights
            self.swarm[i].set_weights(new_position)
        return None

    def get_best_solution(self, metric):
        """
        Return the best portfolio in the swarm based on the specified metric.
        :param metric: Metric to optimize ('volatility', 'return', 'sharpe_ratio').
        :return: Best portfolio (Portfolio instance).
        """

        # If the metric is not the allowed metrics, raise an error.
        if metric not in ['volatility','return','sharpe_ratio']:
            raise ValueError("Invalid fitness metric '{metric}', must be one of volatility, return, sharpe_ratio")
        
        # Evaluate fitness for each particle.
        self.set_fitness(metric)

        # Find the best particle based on the metric.
        current_best_idx = self.swarm_df['fitness'].idxmax()
        self.current_best_weight = self.swarm[current_best_idx].get_weights().copy()

        return self.swarm[current_best_idx]

    def get_analysis(self):
        """
        Return a DataFrame with the best fitness and mean fitness over iterations.
        :return: DataFrame with best fitness and mean fitness.
        """
        # Create a DataFrame with best fitness and mean fitness over iterations.
        d = {'best_fitness': self.best_overall_fitness, 
             'mean_fitness': self.swarm_mean}
        output = pd.DataFrame(data=d)
        return output
