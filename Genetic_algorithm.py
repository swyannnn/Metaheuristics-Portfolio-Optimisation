import random
import pandas as pd
import numpy as np
import copy
from Portfolio import Portfolio

class Genetic_algorithm:
    """
    A class to implement a genetic algorithm for portfolio optimization.
    """
    def __init__(self, returns_df, corr_matrix, risk_free, min_weight, ga_config):
        self.returns = returns_df
        self.corr = corr_matrix
        self.risk_free = risk_free 
        self.min_weight = min_weight
        self.population_size = ga_config.get("population_size", 10)
        self.selection_method = ga_config.get("selection_method", 'tournament')
        self.tournament_size = ga_config.get("tournament_size", 3)
        self.crossover_method = ga_config.get("crossover_method", 'uniform')
        self.crossover_rate = ga_config.get("crossover_rate", 0.8)
        self.mutation_rate = ga_config.get("mutation_rate", 0.1)
        self.replacement_method = ga_config.get("replacement_method", 'elitism')
        self.overall_best_portfolio = None
        self.overall_best_metric = []
        self.best_history = []
        return None
    
    def run(self, metric, max_generations, convergence_threshold=1e-6, convergence_window=100):
        """
        Run the genetic algorithm for a specified number of generations.
        Args:
            metric (str): The objective metric, e.g., 'volatility', 'return', or 'sharpe_ratio'.
            convergence_threshold (float): Threshold for convergence.
            convergence_window (int): Number of generations to check for convergence.
        """
        self.initialize()
        convergence_met = False
        iteration = 1
        # Loop until either convergence or max_generations is reached
        while not convergence_met and iteration < max_generations:
            self.set_fitness(metric)
            # Get the best portfolio from the current generation.
            current_best_portfolio = self.get_best_portfolio(metric)

            # Store the best portfolio's volatility and expected return.
            self.best_history.append((current_best_portfolio.get_volatility(),
                    current_best_portfolio.get_expected_return()))

            # Get the metric for the best portfolio.
            if metric == 'volatility':
                current_metric = current_best_portfolio.get_volatility()
            elif metric == 'return':
                current_metric = current_best_portfolio.get_expected_return()
            elif metric == 'sharpe_ratio':
                current_metric = current_best_portfolio.get_sharpe_ratio()
            else:
                raise ValueError("Unsupported metric. Choose 'volatility', 'return', or 'sharpe_ratio'.")
            
            # Update overall best: for volatility (minimization) lower is better;
            # for return or sharpe_ratio (maximization) higher is better.
            if self.overall_best_portfolio is None and len(self.overall_best_metric) == 0:
                self.overall_best_portfolio = copy.deepcopy(current_best_portfolio)
                self.overall_best_metric.append(current_metric)
            else:
                # Update best if an improvement is found
                if (metric == 'volatility' and current_metric < self.overall_best_metric[-1]) or \
                   (metric != 'volatility' and current_metric > self.overall_best_metric[-1]):
                    self.overall_best_portfolio = copy.deepcopy(current_best_portfolio)
                    self.overall_best_metric.append(current_metric)
                    # print("Improved current_metric", current_metric)
                    # print(f"New Overall Best Portfolio Found at Generation {iteration+1}!")
                    # print("Weights:", current_best_portfolio.get_weights())
                    # print("---------------")
                else: 
                    # append the previous best
                    self.overall_best_metric.append(self.overall_best_metric[-1])

            self.crossover() # selection() is called in this function
            self.mutation()
            self.replacement(metric)

            # Check convergence: if improvement over the last 'window' iterations is below convergence_threshold.
            if len(self.overall_best_metric) >= convergence_window:
                recent_changes = np.abs(np.diff(self.overall_best_metric[-convergence_window:]))
                if np.all(recent_changes < convergence_threshold):
                    convergence_met = True
            iteration += 1
        return None
    
    def initialize(self):
        """
        Initialize the population with random portfolios.
        """
        self.population = []
        self.offspring = []
        self.population_fitness = []
        self.population_mean  = []
        for i in range(self.population_size):
            self.population.append(Portfolio(self.returns, self.corr, self.risk_free, self.min_weight))
        self.population_df = Portfolio.evaluate_solution(self.population)
        return None
    
    def set_fitness(self, metric):
        """
        Set the fitness of each portfolio in the population based on the specified metric.
        Args:
            metric (str): The objective metric, e.g., 'volatility', 'return', or 'sharpe_ratio'.
        """
        if(metric == 'volatility'):
            max_volatility = self.population_df[metric].max()
            self.population_df.sort_values(by=metric, inplace=True, ascending=True)
            self.population_df['fitness'] = max_volatility - self.population_df[metric] + 1
            self.population_df['fitness'] = self.population_df['fitness']/self.population_df['fitness'].sum()
        else:
            self.population_df.sort_values(by=metric, inplace=True, ascending=False)
            self.population_df['fitness'] = self.population_df[metric]/self.population_df[metric].sum()
        self.population_df['selection_prob'] = self.population_df['fitness']
        for i in range(1, len(self.population_df)):
            self.population_df.loc[self.population_df.index[i], 'selection_prob'] = (
                self.population_df.loc[self.population_df.index[i-1], 'selection_prob'] +
                self.population_df.loc[self.population_df.index[i], 'fitness']
            )
        return self.population_df

    def selection(self):
        """
        Select a certain number of parents from the current population using the specified selection method.
        Args:
            selection_method (str): The selection method to use. Options: 'tournament', 'roulette_wheel', 'rank_based'
        Returns:
            list: A list of selected parent indices.
        """
        # To prevent same parents being selected, we use set instead of list
        parents_idx = set()
        
        while len(parents_idx) < 2:
            if self.selection_method == 'roulette_wheel':
                r = random.random()
                for i, prob in enumerate(self.population_df['selection_prob']):
                    if r < prob:
                        parents_idx.add(self.population_df.index[i])
                        break
        
            elif self.selection_method == 'tournament':
                # Randomly select tournament_size contestants (their indices) from the population.
                contestants = random.sample(list(self.population_df.index), self.tournament_size)
                best = None
                best_fitness = -float('inf')
                for contestant in contestants:
                    fitness = self.population_df.loc[contestant, 'fitness']
                    if fitness > best_fitness:
                        best = contestant
                        best_fitness = fitness
                parents_idx.add(best)
            
            elif self.selection_method == 'rank_based':
                # First, sort the population by fitness in descending order.
                sorted_indices = self.population_df.sort_values(by='fitness', ascending=False).index.tolist()
                # For a simple rank-based selection, assign weights inversely proportional to rank.
                # Best candidate (rank 1) gets highest probability.
                ranks = list(range(1, len(sorted_indices) + 1))
                weights = [1.0 / rank for rank in ranks]
                total_weight = sum(weights)
                probs = [w / total_weight for w in weights]
                cum_probs = np.cumsum(probs)
                r = random.random()
                for idx, cp in enumerate(cum_probs):
                    if r < cp:
                        parents_idx.add(sorted_indices[idx])
                        break
            else:
                raise ValueError("Unsupported selection method: choose 'tournament', 'roulette_wheel', or 'rank_based'")
        # convert back to list
        return list(parents_idx)
    
    def crossover(self):
        """
        Produce offspring using the specified crossover method.
        
        Args:
            crossover_method (str): 'uniform', 'one_point', or 'two_point'.
            crossover_rate (float): Probability that crossover will occur.
        
        This method generates a new offspring population of size self.population_size.
        """
        offspring = []
        
        # For each offspring to generate:
        for i in range(self.population_size):
            # Select exactly 2 parents randomly from self.parent.
            parent_indices = self.selection()
            idx_parent1, idx_parent2 = parent_indices[0], parent_indices[1]
            parent1 = self.population[idx_parent1].get_weights()
            parent2 = self.population[idx_parent2].get_weights()
            
            # Decide if crossover occurs.
            if random.random() < self.crossover_rate:
                if self.crossover_method == 'uniform':
                    child_weights = np.array([parent1[i] if random.random() < 0.5 else parent2[i]
                                            for i in range(len(parent1))])
                elif self.crossover_method == 'one_point':
                    point = random.randint(1, len(parent1)-1)
                    child_weights = np.concatenate([parent1[:point], parent2[point:]])
                elif self.crossover_method == 'two_point':
                    point1 = random.randint(1, len(parent1)-2)
                    point2 = random.randint(point1+1, len(parent1)-1)
                    child_weights = np.concatenate([parent1[:point1], parent2[point1:point2], parent1[point2:]])
                else:
                    raise ValueError("Unsupported crossover method. Choose 'uniform', 'one_point', or 'two_point'.")
            else:
                # No crossover: copy one parent's weights at random.
                child_weights = parent1.copy() if random.random() < 0.5 else parent2.copy()
            
            # Normalize child weights to sum to 1.
            child_weights = child_weights / np.sum(child_weights)

            # Create a new Portfolio for the offspring and assign the computed weights.
            child = Portfolio(self.returns, self.corr, self.risk_free, self.min_weight)
            child.set_weights(child_weights)
            offspring.append(child)
        
        # Update offspring list and population accordingly.
        self.offspring = offspring
        return None

    def mutation(self):
        """
        Apply mutation to the offspring population.

        For each child, with probability mutation_rate, select two random indices.
        Transfer a random amount of weight from the second index to the first,
        ensuring that the weight at the second index does not drop below min_weight.
        """
        n_assets = len(self.population[0].get_weights())
        for child in self.offspring:
            if random.random() < self.mutation_rate:
                # Select two different random indices
                idx1 = random.randrange(0, n_assets)
                idx2 = random.randrange(0, n_assets)
                while idx2 == idx1:
                    idx2 = random.randrange(0, n_assets)
                
                # Get a copy of current weights
                w = child.get_weights().copy()
                
                # Determine the maximum weight that can be subtracted from asset at idx2
                max_subtract = w[idx2] - self.min_weight
                if max_subtract > 0:
                    # Choose a random amount to subtract from idx2 (and add to idx1)
                    rand = random.uniform(0, max_subtract)
                    w[idx1] += rand
                    w[idx2] -= rand
                    child.set_weights(w)
                # If no mutation is possible because asset at idx2 is at the minimum weight, skip mutation.
        return None
    
    def replacement(self, metric, num_elites=0):
        """
        Replace the current population based on the specified strategy.
        
        Args:
            metric (str): The objective metric, e.g., 'volatility', 'return', or 'sharpe_ratio'.
            strategy (str): Replacement strategy ('generational' or 'elitism').
            num_elites (int): Number of parent individuals to preserve (if using elitism).
        
        For generational replacement, the entire population is replaced by the offspring.
        For elitism replacement, a specified number of elites are preserved from the parent population,
        and the remaining individuals are randomly selected from the offspring to maintain population size.
        """
        if self.replacement_method == 'generational':
            new_population = self.offspring.copy()
        elif self.replacement_method == 'elitism':
            # Ensure we have a valid number of elites.
            if num_elites < 0 or num_elites > self.population_size:
                raise ValueError("num_elites must be between 0 and the population size.")
            
            # For elitism, sort parent's population by fitness.
            # For 'volatility', lower is better (ascending); for others, higher is better (descending).
            ascending = True if metric == 'volatility' else False
            sorted_parents = self.population_df.sort_values(by=metric, ascending=ascending)
            
            # Randomly select num_elites from the top portion (elites) of the parent population.
            # (Here, we take the top half as elites and then randomly choose from that half.)
            elite_pool = list(sorted_parents.head(int(self.population_size/2)).index)
            parent_elites = random.sample(elite_pool, min(num_elites, len(elite_pool)))
            
            # Now, for offspring, sort their fitness similarly.
            offspring_df = Portfolio.evaluate_solution(self.offspring)
            sorted_offspring = offspring_df.sort_values(by=metric, ascending=ascending)
            num_offspring_needed = self.population_size - num_elites
            
            # Randomly select the required number of offspring from the sorted offspring.
            offspring_indices = random.sample(list(sorted_offspring.index), num_offspring_needed)
            
            # Build the new population:
            new_population = [self.population[i] for i in parent_elites] + \
                            [self.offspring[i] for i in offspring_indices]
        else:
            raise ValueError("Unsupported replacement strategy. Choose 'generational' or 'elitism'.")
        
        # Update population and reset offspring.
        self.population = new_population
        self.offspring = []
        
        # Update the population DataFrame and add to the tree.
        self.population_df = Portfolio.evaluate_solution(self.population)
        
        # Record analysis metrics.
        mean_fit = self.population_df[metric].mean()
        if metric == 'volatility':
            best_fit = self.population_df.sort_values(by=metric, ascending=True).head(1)[metric].iloc[0]
        else:
            best_fit = self.population_df.sort_values(by=metric, ascending=False).head(1)[metric].iloc[0]
        # Store the best solution; here we simply append the first one, which is the best (sorted).
        self.population_fitness.append(best_fit)
        self.population_mean.append(mean_fit)
        return None
    
    def get_best_portfolio1(self, metric):
        """
        Get the best portfolio from the population based on the specified metric.
        Args:
            metric (str): The objective metric, e.g., 'volatility', 'return', or 'sharpe_ratio'.
        Returns:
            Portfolio: The best portfolio object.
        """
        order = False if metric == 'volatility' else True
        self.population_df.sort_values(by=metric, inplace=True, ascending=order)
        idx = self.population_df.head(1).index.values[0]
        return self.population[idx]
    
    def get_best_portfolio(self, metric):
        """
        Get the best portfolio from the population based on the specified metric.
        Args:
            metric (str): The objective metric, one of 'volatility', 'return', or 'sharpe_ratio'.
        Returns:
            Portfolio: The best portfolio object according to the metric.
        """
        if metric == 'volatility':
            # lower volatility is better
            best_idx = self.population_df['volatility'].idxmin()
        elif metric == 'return':
            # higher return is better
            best_idx = self.population_df['return'].idxmax()
        elif metric == 'sharpe_ratio':
            # higher Sharpe is better
            best_idx = self.population_df['sharpe_ratio'].idxmax()
        else:
            raise ValueError("Unsupported metric. Choose 'volatility', 'return', or 'sharpe_ratio'.")
        
        return self.population[best_idx]

    def get_analysis(self):
        """
        Get the analysis of the genetic algorithm run.
        Returns:
            pd.DataFrame: DataFrame containing the best fitness and mean fitness of the population.
        """
        d = {'best_fitness':self.overall_best_metric,'mean_fitness':self.population_mean}
        output = pd.DataFrame(data=d) 
        return output
    
        