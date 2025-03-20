import random
import pandas as pd
import numpy as np
from Portfolio import Portfolio

class Genetic_algorithm:
    def __init__(self, returns_df, corr_matrix, risk_free, min_weight, ga_config):
        self.returns = returns_df
        self.corr = corr_matrix
        self.risk_free = risk_free
        self.min_weight = min_weight
        self.population_size = ga_config.get("population_size", 10)
        self.selection_method = ga_config.get("selection_method", 'tournament')
        self.tournament_size = ga_config.get("tournament_size", None)
        self.crossover_method = ga_config.get("crossover_method", 'uniform')
        self.crossover_rate = ga_config.get("crossover_rate", 0.8)
        self.mutation_rate = ga_config.get("mutation_rate", 0.1)
        self.replacement_method = ga_config.get("replacement_method", 'elitism')
        self.overall_best = None
        return None
    
    def run(self, max_generations, variable):
        self.initialize()
        for i in range(max_generations):
            self.set_fitness(variable)
            self.selection()
            self.crossover()
            self.mutation()
            self.replacement(variable)
            # Get the best portfolio from the current generation.
        
            current_best = self.get_population(variable)
            current_metric = getattr(current_best, variable)
            current_best = self.get_population(variable)

            # Update overall best: for volatility (minimization) lower is better;
            # for return or sharpe_ratio (maximization) higher is better.
            if self.overall_best is None:
                self.overall_best = current_best
                self.overall_best_metric = current_metric
            else:
                if current_metric > self.overall_best_metric:
                    print("New Overall Best Portfolio Found!")
                    self.overall_best = current_best
                    self.overall_best_metric = current_metric

            # print("Generation:", i)
            # print("Current Generation Best Weights:", current_best.get_weights())
            # if variable == 'volatility':
            #     print(f"{variable} =", current_best.get_volatility())
            # elif variable == 'return':
            #     print(f"{variable} =", current_best.get_expected_return())
            # elif variable == 'sharpe_ratio':
            #     print(f"{variable} =", current_best.get_sharpe_ratio())
            # print("Overall Best so far:", self.overall_best.get_weights())
            # print("---------------")
        return None
    
    def initialize(self):
        self.population = []
        self.offspring = []
        self.population_best = []
        self.population_fitness = []
        self.population_mean  = []
        for i in range(self.population_size):
            self.population.append(Portfolio(self.returns, self.corr, self.risk_free, self.min_weight))
        self.population_df = self.to_table(self.population)
        return None
    
    def set_fitness(self, metric):
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
    
    def selection(self, parent_candidates_num = 4):
        """
        Select a certain number of parents from the current population using the specified selection method.
        Args:
            selection_method (str): The selection method to use. Options: 'tournament', 'roulette_wheel', 'rank_based'
        Returns:
            list: A list of selected parent indices.
        """
        parent_candidates = []
        
        if self.selection_method == 'roulette_wheel':
            for _ in range(parent_candidates_num):
                r = random.random()
                for i, prob in enumerate(self.population_df['selection_prob']):
                    if r < prob:
                        parent_candidates.append(self.population_df.index[i])
                        break
    
        elif self.selection_method == 'tournament':
            for _ in range(parent_candidates_num):
                # Randomly select tournament_size contestants (their indices) from the population.
                contestants = random.sample(list(self.population_df.index), self.tournament_size)
                best = None
                best_fitness = -float('inf')
                for contestant in contestants:
                    fitness = self.population_df.loc[contestant, 'fitness']
                    if fitness > best_fitness:
                        best = contestant
                        best_fitness = fitness
                parent_candidates.append(best)
        
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
            for _ in range(parent_candidates_num):
                r = random.random()
                for idx, cp in enumerate(cum_probs):
                    if r < cp:
                        parent_candidates.append(sorted_indices[idx])
                        break
        else:
            raise ValueError("Unsupported selection method: choose 'tournament', 'roulette_wheel', or 'rank_based'")
        
        self.parent_candidates = parent_candidates
        return None
    
    def crossover(self):
        """
        Produce offspring using the specified crossover method.
        
        Args:
            crossover_method (str): 'uniform', 'one_point', or 'two_point'.
            crossover_rate (float): Probability that crossover will occur.
        
        This method generates a new offspring population of size self.population_size.
        """
        offspring = []
        # Ensure that self.parent_candidates has been populated by the selection method.
        if not hasattr(self, "parent_candidates") or len(self.parent_candidates) < 2:
            raise ValueError("Not enough parent candidates available for crossover.")
        
        # For each offspring to generate:
        for i in range(self.population_size):
            # Select exactly 2 parents randomly from self.parent_candidates.
            parent_indices = random.sample(self.parent_candidates, 2)
            idx_parent1, idx_parent2 = parent_indices
            w1 = self.population[idx_parent1].get_weights()
            w2 = self.population[idx_parent2].get_weights()
            
            # Decide if crossover occurs.
            if random.random() < self.crossover_rate:
                if self.crossover_method == 'uniform':
                    child_weights = np.array([w1[i] if random.random() < 0.5 else w2[i]
                                            for i in range(len(w1))])
                elif self.crossover_method == 'one_point':
                    point = random.randint(1, len(w1)-1)
                    child_weights = np.concatenate([w1[:point], w2[point:]])
                elif self.crossover_method == 'two_point':
                    point1 = random.randint(1, len(w1)-2)
                    point2 = random.randint(point1+1, len(w1)-1)
                    child_weights = np.concatenate([w1[:point1], w2[point1:point2], w1[point2:]])
                else:
                    raise ValueError("Unsupported crossover method. Choose 'uniform', 'one_point', or 'two_point'.")
            else:
                # No crossover: copy one parent's weights at random.
                child_weights = w1.copy() if random.random() < 0.5 else w2.copy()
            
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

        Args:
            mutation_rate (float): The probability that mutation will occur.

        This method mutates the weights of each offspring in the population.
        """
        n_assets = len(self.population[0].get_weights())
        for child in self.offspring:
            # Decide if mutation occurs.
            if random.random() < self.mutation_rate:
                # Select a random asset and assign a random weight to it.
                idx1 = random.randrange(0,n_assets)
                idx2 = random.randrange(0,n_assets)
                w = child.get_weights()
                minimo = min(w[idx1],w[idx2])
                rand = random.uniform(0,minimo)
                w[idx1] += rand
                w[idx2] -= rand
                child.set_weights(w)
        return None
    
    def replacement(self, variable, strategy='generational', num_elites=0):
        """
        Replace the current population based on the specified strategy.
        
        Args:
            variable (str): The objective variable, e.g., 'volatility', 'return', or 'sharpe_ratio'.
            strategy (str): Replacement strategy ('generational' or 'elitism').
            num_elites (int): Number of parent individuals to preserve (if using elitism).
        
        For generational replacement, the entire population is replaced by the offspring.
        For elitism replacement, a specified number of elites are preserved from the parent population,
        and the remaining individuals are randomly selected from the offspring to maintain population size.
        """
        if strategy == 'generational':
            new_population = self.offspring.copy()
        elif strategy == 'elitism':
            # Ensure we have a valid number of elites.
            if num_elites < 0 or num_elites > self.population_size:
                raise ValueError("num_elites must be between 0 and the population size.")
            
            # For elitism, sort parent's population by fitness.
            # For 'volatility', lower is better (ascending); for others, higher is better (descending).
            if variable == 'volatility':
                ascending = True
            else:
                ascending = False
            sorted_parents = self.population_df.sort_values(by=variable, ascending=ascending)
            
            # Randomly select num_elites from the top portion (elites) of the parent population.
            # (Here, we take the top half as elites and then randomly choose from that half.)
            elite_pool = list(sorted_parents.head(int(self.population_size/2)).index)
            parent_elites = random.sample(elite_pool, min(num_elites, len(elite_pool)))
            
            # Now, for offspring, sort their fitness similarly.
            offspring_df = self.to_table(self.offspring)
            sorted_offspring = offspring_df.sort_values(by=variable, ascending=ascending)
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
        self.population_df = self.to_table(self.population)
        
        # Record analysis metrics.
        mean_fit = self.population_df[variable].mean()
        if variable == 'volatility':
            best_fit = self.population_df.sort_values(by=variable, ascending=True).head(1)[variable].iloc[0]
        else:
            best_fit = self.population_df.sort_values(by=variable, ascending=False).head(1)[variable].iloc[0]
        # Store the best solution; here we simply append the first one, which is the best (sorted).
        self.population_best.append(self.population[0])
        self.population_fitness.append(best_fit)
        self.population_mean.append(mean_fit)
        return None
    
    def to_table(self, array):
        exp_returns = [s.get_expected_return() for s in array]
        volatilities = [s.get_volatility() for s in array]
        sharpe_ratios = [s.get_sharpe_ratio() for s in array]
        d = {'return': exp_returns, 'volatility': volatilities, 'sharpe_ratio' : sharpe_ratios}
        df = pd.DataFrame(data=d) 
        return df
    
    def get_population(self,variable):
        self.population_df.sort_values(by=variable, inplace=True, ascending=False)
        idx = self.population_df.head(1).index.values[0]
        return self.population[idx]

    def get_analysis(self):
        d = {'best_fitness':self.population_fitness,'fitness_mean':self.population_mean}
        output = pd.DataFrame(data=d) 
        return output
    
        