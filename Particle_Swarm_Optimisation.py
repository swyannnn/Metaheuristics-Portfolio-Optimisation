import pandas as pd
import numpy as np
from Portfolio import Portfolio

class Particle_Swarm_Optimisation:
    def __init__(self, returns_df, corr_matrix, risk_free, size, min_weight, inertia, cognitive, social):
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
        self.size = size
        self.min_weight = min_weight
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social

        # Create a swarm of particles (each a Portfolio instance)
        self.swarm = []
        for _ in range(size):
            particle = Portfolio(self.returns, self.corr, self.risk_free, self.min_weight)
            self.swarm.append(particle)

        # Initialize velocities for each particle as a numpy array with same dimensions as weights.
        self.n_assets = len(self.swarm[0].get_weights())
        self.velocities = [np.zeros(self.n_assets) for _ in range(self.size)]

        # Initialize personal best for each particle
        self.personal_best = [particle.get_weights().copy() for particle in self.swarm]
        self.personal_best_fitness = [None] * self.size

        # Evaluate initial fitness based on the chosen variable (e.g., 'volatility', 'return', 'sharpe_ratio')
        # Higher fitness is better. This will add a 'fitness' column to the swarm DataFrame.
        self.swarm_df = self.to_table(self.swarm)
        self.global_best_index = self.swarm_df['fitness'].idxmax() if 'fitness' in self.swarm_df.columns else 0
        self.global_best = self.swarm[self.global_best_index].get_weights().copy()

        # To track evolution over iterations
        self.tree = []
        self.tree.append(self.swarm.copy())
        self.swarm_fitness = []
        self.swarm_mean = []
        
    def to_table(self, array):
        # Convert swarm into a DataFrame with computed metrics
        exp_returns = [p.get_expected_return() for p in array]
        volatilities = [p.get_volatility() for p in array]
        sharpe_ratios = [p.get_sharpe_ratio() for p in array]
        d = {'return': exp_returns, 'volatility': volatilities, 'sharpe_ratio': sharpe_ratios}
        df = pd.DataFrame(data=d)
        # For selection purposes, we need to define a fitness.
        # For example, if variable is 'sharpe_ratio', we treat that as fitness.
        # Otherwise, if using volatility as fitness, lower volatility means better fitness.
        # In this helper, we'll let the user call set_fitness() separately.
        return df

    def set_fitness(self, variable):
        """
        Evaluate fitness for each particle.
        For 'volatility', lower is better; for 'return' or 'sharpe_ratio', higher is better.
        This function will add a 'fitness' column to the swarm DataFrame.
        """
        self.swarm_df = self.to_table(self.swarm)
        if variable == 'volatility':
            # Lower volatility is better.
            max_vol = self.swarm_df[variable].max()
            self.swarm_df.sort_values(by=variable, inplace=True, ascending=True)
            self.swarm_df['fitness'] = max_vol - self.swarm_df[variable] + 1
        else:
            # For return or sharpe_ratio, higher is better.
            self.swarm_df.sort_values(by=variable, inplace=True, ascending=False)
            self.swarm_df['fitness'] = self.swarm_df[variable]

        # Update personal best fitness if not already set.
        for i in range(self.size):
            current_fitness = self.swarm_df.iloc[i]['fitness']
            if self.personal_best_fitness[i] is None or current_fitness > self.personal_best_fitness[i]:
                self.personal_best[i] = self.swarm[i].get_weights().copy()
                self.personal_best_fitness[i] = current_fitness

        # Update global best among all particles.
        best_idx = self.swarm_df['fitness'].idxmax()
        self.global_best = self.swarm[best_idx].get_weights().copy()
        return self.swarm_df

    def update_velocity(self):
        """
        Update the velocity for each particle using the PSO formula:
            v = inertia * v + cognitive * r1 * (pbest - position) + social * r2 * (gbest - position)
        where r1 and r2 are random numbers between 0 and 1.
        """
        for i in range(self.size):
            current_position = self.swarm[i].get_weights()
            r1 = np.random.rand(self.n_assets)
            r2 = np.random.rand(self.n_assets)
            cognitive_component = self.cognitive * r1 * (self.personal_best[i] - current_position)
            social_component = self.social * r2 * (self.global_best - current_position)
            self.velocities[i] = (self.inertia * self.velocities[i] + cognitive_component + social_component)
        return None

    def update_position(self):
        """
        Update each particle's position (portfolio weights) using:
            new_position = current_position + velocity
        Then re-normalize the weights to sum to 1.
        """
        for i in range(self.size):
            # Update position based on velocity
            current_position = self.swarm[i].get_weights()
            new_position = current_position + self.velocities[i]

            # Ensure weights are non-negative and sum to 1
            new_position = np.maximum(new_position, 0)  # enforce non-negativity

            # If all weights are zero, reset to equal weights
            if new_position.sum() == 0:
                new_position = np.ones(self.n_assets) / self.n_assets
            else:
                # Normalize to sum to 1
                new_position = new_position / new_position.sum()

            # Update the particle's position
            self.swarm[i].set_weights(new_position)

        return None

    def run(self, iterations, variable):
        """
        Run the PSO algorithm for a given number of iterations.
        The variable parameter determines which metric (e.g., 'volatility', 'return', 'sharpe_ratio')
        is used for evaluating fitness.
        """
        for _ in range(iterations):
            # Evaluate fitness for each particle.
            self.set_fitness(variable)

            # Update velocities based on personal best and global best.
            self.update_velocity()

            # Update positions based on the new velocities.
            self.update_position()

            # Record current state of the swarm.
            self.tree.append(self.swarm.copy())

            # record best and mean fitness.
            best_fitness = self.swarm_df['fitness'].max()
            mean_fitness = self.swarm_df['fitness'].mean()
            self.swarm_fitness.append(best_fitness)
            self.swarm_mean.append(mean_fitness)
        return None

    def get_best_solution(self, variable):
        """
        Return the best portfolio in the swarm based on the specified variable.
        """
        # Evaluate fitness for each particle.
        self.set_fitness(variable)

        # Find the best particle based on the variable.
        best_idx = self.swarm_df['fitness'].idxmax()

        return self.swarm[best_idx]

    def get_tree(self):
        return self.tree

    def get_analysis(self):
        # Create a DataFrame with best fitness and mean fitness over iterations.
        d = {'best_fitness': self.swarm_fitness, 'fitness_mean': self.swarm_mean}
        output = pd.DataFrame(data=d)
        return output
