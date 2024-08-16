import random
import numpy as np

@dataclass
class Hyperparameters:
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    bias: bool

def initialize_population(pop_size, config):
    population = []
    for _ in range(pop_size):
        n_layer = random.randint(1, config['max_layers'])
        n_head = random.randint(1, config['max_heads'])
        n_embd = random.randint(config['min_embd'], config['max_embd'])
        dropout = random.uniform(0.0, config['max_dropout'])
        bias = random.choice([True, False])
        population.append(Hyperparameters(n_layer, n_head, n_embd, dropout, bias))
    return population

def fitness_function(hyperparameters, train_func):
    # Train the model with the given hyperparameters and return the validation loss
    _, _, val_log_info = train_func(hyperparameters)
    return val_log_info[-1]['val/loss']

def select_parents(population, fitnesses, num_parents):
    parents = random.choices(population, weights=fitnesses, k=num_parents)
    return parents

def crossover(parent1, parent2):
    child1 = Hyperparameters(
        n_layer=random.choice([parent1.n_layer, parent2.n_layer]),
        n_head=random.choice([parent1.n_head, parent2.n_head]),
        n_embd=random.choice([parent1.n_embd, parent2.n_embd]),
        dropout=random.choice([parent1.dropout, parent2.dropout]),
        bias=random.choice([parent1.bias, parent2.bias])
    )
    child2 = Hyperparameters(
        n_layer=random.choice([parent1.n_layer, parent2.n_layer]),
        n_head=random.choice([parent1.n_head, parent2.n_head]),
        n_embd=random.choice([parent1.n_embd, parent2.n_embd]),
        dropout=random.choice([parent1.dropout, parent2.dropout]),
        bias=random.choice([parent1.bias, parent2.bias])
    )
    return child1, child2

def mutate(hyperparameters, config):
    if random.random() < 0.1:
        hyperparameters.n_layer = random.randint(1, config['max_layers'])
    if random.random() < 0.1:
        hyperparameters.n_head = random.randint(1, config['max_heads'])
    if random.random() < 0.1:
        hyperparameters.n_embd = random.randint(config['min_embd'], config['max_embd'])
    if random.random() < 0.1:
        hyperparameters.dropout = random.uniform(0.0, config['max_dropout'])
    if random.random() < 0.1:
        hyperparameters.bias = random.choice([True, False])
    return hyperparameters

def genetic_algorithm(config, train_func, generations=10, pop_size=20, num_parents=10):
    population = initialize_population(pop_size, config)
    for generation in range(generations):
        fitnesses = [fitness_function(hp, train_func) for hp in population]
        parents = select_parents(population, fitnesses, num_parents)
        next_population = []
        for i in range(0, num_parents, 2):
            parent1, parent2 = parents[i], parents[i+1]
            child1, child2 = crossover(parent1, parent2)
            next_population.extend([mutate(child1, config), mutate(child2, config)])
        population = next_population
    best_hyperparameters = population[np.argmin(fitnesses)]
    return best_hyperparameters
