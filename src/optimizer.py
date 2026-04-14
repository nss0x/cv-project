"""
Evolutionary optimization algorithms (PSO and GA) for hyperparameter tuning
"""
import numpy as np
from copy import deepcopy
import torch
from tqdm import tqdm
from trainer import Trainer
from config import HYPERPARAMETER_RANGES

class ParticleSwarmOptimizer:
    """Particle Swarm Optimization for hyperparameter tuning"""
    
    def __init__(self, population_size=20, generations=30, w=0.7, c1=1.5, c2=1.5):
        """
        Initialize PSO
        
        Args:
            population_size: Number of particles
            generations: Number of generations
            w: Inertia weight
            c1: Cognitive parameter
            c2: Social parameter
        """
        self.population_size = population_size
        self.generations = generations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        # Hyperparameter bounds
        self.bounds = HYPERPARAMETER_RANGES
        self.param_names = list(self.bounds.keys())
        self.num_params = len(self.param_names)
        
        # Initialize particles
        self.particles = self._initialize_particles()
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.num_params))
        
        self.best_position = None
        self.best_score = -np.inf
        self.best_scores = []
        
    def _initialize_particles(self):
        """Initialize particle positions"""
        particles = np.zeros((self.population_size, self.num_params))
        for i, param_name in enumerate(self.param_names):
            lower, upper = self.bounds[param_name]
            particles[:, i] = np.random.uniform(lower, upper, self.population_size)
        return particles
    
    def _particles_to_config(self, particle):
        """Convert particle to training config"""
        config = {}
        for i, param_name in enumerate(self.param_names):
            lower, upper = self.bounds[param_name]
            # Normalize and apply bounds
            value = particle[i]
            if param_name in ['learning_rate', 'weight_decay']:
                # Log scale for learning rate and weight decay
                log_lower = np.log10(lower)
                log_upper = np.log10(upper)
                log_value = np.clip(value, log_lower, log_upper)
                config[param_name] = 10 ** log_value
            elif param_name == 'batch_size':
                # Integer value
                config[param_name] = int(np.clip(value, lower, upper))
            else:
                # Regular bounds
                config[param_name] = np.clip(value, lower, upper)
        return config
    
    def _clip_particles(self):
        """Ensure particles are within bounds"""
        for i, param_name in enumerate(self.param_names):
            lower, upper = self.bounds[param_name]
            self.particles[:, i] = np.clip(self.particles[:, i], lower, upper)
    
    def optimize(self, objective_function):
        """
        Run PSO optimization
        
        Args:
            objective_function: Function that takes config and returns score
        
        Returns:
            Best configuration and best score
        """
        print("\n" + "="*60)
        print("  Particle Swarm Optimization")
        print("="*60)
        
        # Evaluate initial particles
        scores = np.zeros(self.population_size)
        best_positions = deepcopy(self.particles)
        best_scores = best_scores_iter = np.zeros(self.population_size)
        
        print("Evaluating initial population...")
        for i in range(self.population_size):
            config = self._particles_to_config(self.particles[i])
            scores[i] = objective_function(config)
            best_scores[i] = scores[i]
            
            if scores[i] > self.best_score:
                self.best_score = scores[i]
                self.best_position = deepcopy(self.particles[i])
        
        print(f"Initial best score: {self.best_score:.4f}\n")
        
        # Optimization loop
        for gen in tqdm(range(self.generations), desc="PSO Generations"):
            # Update velocities and positions
            for i in range(self.population_size):
                r1 = np.random.random(self.num_params)
                r2 = np.random.random(self.num_params)
                
                self.velocities[i] = (
                    self.w * self.velocities[i] +
                    self.c1 * r1 * (best_positions[i] - self.particles[i]) +
                    self.c2 * r2 * (self.best_position - self.particles[i])
                )
                
                self.particles[i] += self.velocities[i]
            
            # Clip particles to bounds
            self._clip_particles()
            
            # Evaluate particles
            for i in range(self.population_size):
                config = self._particles_to_config(self.particles[i])
                scores[i] = objective_function(config)
                
                if scores[i] > best_scores[i]:
                    best_scores[i] = scores[i]
                    best_positions[i] = deepcopy(self.particles[i])
                
                if scores[i] > self.best_score:
                    self.best_score = scores[i]
                    self.best_position = deepcopy(self.particles[i])
            
            self.best_scores.append(self.best_score)
        
        # Convert best position to config
        best_config = self._particles_to_config(self.best_position)
        return best_config, self.best_score

class GeneticAlgorithm:
    """Genetic Algorithm for hyperparameter tuning"""
    
    def __init__(self, population_size=20, generations=30, mutation_rate=0.1, crossover_rate=0.8):
        """
        Initialize GA
        
        Args:
            population_size: Population size
            generations: Number of generations
            mutation_rate: Mutation probability
            crossover_rate: Crossover probability
        """
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Hyperparameter bounds
        self.bounds = HYPERPARAMETER_RANGES
        self.param_names = list(self.bounds.keys())
        self.num_params = len(self.param_names)
        
        self.best_fitness = -np.inf
        self.best_individual = None
        self.best_scores = []
    
    def _initialize_population(self):
        """Initialize population"""
        population = np.zeros((self.population_size, self.num_params))
        for i, param_name in enumerate(self.param_names):
            lower, upper = self.bounds[param_name]
            population[:, i] = np.random.uniform(lower, upper, self.population_size)
        return population
    
    def _individual_to_config(self, individual):
        """Convert individual to training config"""
        config = {}
        for i, param_name in enumerate(self.param_names):
            lower, upper = self.bounds[param_name]
            value = individual[i]
            if param_name in ['learning_rate', 'weight_decay']:
                log_lower = np.log10(lower)
                log_upper = np.log10(upper)
                log_value = np.clip(value, log_lower, log_upper)
                config[param_name] = 10 ** log_value
            elif param_name == 'batch_size':
                config[param_name] = int(np.clip(value, lower, upper))
            else:
                config[param_name] = np.clip(value, lower, upper)
        return config
    
    def _crossover(self, parent1, parent2):
        """Single-point crossover"""
        if np.random.random() < self.crossover_rate:
            crossover_point = np.random.randint(1, self.num_params)
            offspring = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            return offspring
        return np.copy(parent1)
    
    def _mutate(self, individual):
        """Gaussian mutation"""
        if np.random.random() < self.mutation_rate:
            for i, param_name in enumerate(self.param_names):
                lower, upper = self.bounds[param_name]
                mutation = np.random.normal(0, 0.1 * (upper - lower))
                individual[i] += mutation
                individual[i] = np.clip(individual[i], lower, upper)
        return individual
    
    def optimize(self, objective_function):
        """
        Run GA optimization
        
        Args:
            objective_function: Function that takes config and returns score
        
        Returns:
            Best configuration and best score
        """
        print("\n" + "="*60)
        print("  Genetic Algorithm")
        print("="*60)
        
        population = self._initialize_population()
        
        print("Evaluating initial population...")
        fitness = np.zeros(self.population_size)
        for i in range(self.population_size):
            config = self._individual_to_config(population[i])
            fitness[i] = objective_function(config)
            
            if fitness[i] > self.best_fitness:
                self.best_fitness = fitness[i]
                self.best_individual = deepcopy(population[i])
        
        print(f"Initial best fitness: {self.best_fitness:.4f}\n")
        
        # Optimization loop
        for gen in tqdm(range(self.generations), desc="GA Generations"):
            # Selection (tournament selection)
            selected_indices = []
            for _ in range(self.population_size):
                tournament_idx = np.random.choice(self.population_size, 3, replace=False)
                tournament_fitness = fitness[tournament_idx]
                selected_indices.append(tournament_idx[np.argmax(tournament_fitness)])
            
            # Crossover and mutation
            new_population = []
            for i in range(0, self.population_size, 2):
                parent1 = population[selected_indices[i]]
                parent2 = population[selected_indices[i+1]] if i+1 < self.population_size else parent1
                
                offspring1 = self._crossover(parent1, parent2)
                offspring2 = self._crossover(parent2, parent1)
                
                offspring1 = self._mutate(offspring1)
                offspring2 = self._mutate(offspring2)
                
                new_population.extend([offspring1, offspring2])
            
            population = np.array(new_population[:self.population_size])
            
            # Evaluate new population
            for i in range(self.population_size):
                config = self._individual_to_config(population[i])
                fitness[i] = objective_function(config)
                
                if fitness[i] > self.best_fitness:
                    self.best_fitness = fitness[i]
                    self.best_individual = deepcopy(population[i])
            
            self.best_scores.append(self.best_fitness)
        
        # Convert best individual to config
        best_config = self._individual_to_config(self.best_individual)
        return best_config, self.best_fitness
