import numpy as np
from operator import itemgetter
from search.search import Search
from random import randint, random
from typing import Dict, List, Union


class GeneticSearch(Search):
    def __init__(self, params: Dict[str, List[Union[int, float]]], n: int = 50):
        super().__init__(params, n)
        self.population_size = n
        self.population = self._init_population()
        
    def _init_population(self):
        return [{param: np.random.choice(values) for param, values in self.params.items()} for _ in range(self.population_size)]
        
    def _mutate(self, individual):
        param_to_mutate = np.random.choice(list(self.params.keys()))
        individual[param_to_mutate] = np.random.choice(self.params[param_to_mutate])
        return individual

    def _crossover(self, parent1, parent2):
        crossover_point = randint(1, len(self.params) - 1)
        child1 = dict(list(parent1.items())[:crossover_point] + list(parent2.items())[crossover_point:])
        child2 = dict(list(parent2.items())[:crossover_point] + list(parent1.items())[crossover_point:])
        return child1, child2

    def _select(self, population_scores):
        sorted_population = sorted(population_scores, key=itemgetter(1), reverse=True)
        return [individual for individual, _ in sorted_population[:self.population_size]]

    def __iter__(self):
        for individual in self.population:
            self._current_params = individual
            yield individual
            
    def receive_results(self, result: Dict[str, float]):
        scores = []
        score = np.mean(result["f1"])
        scores.append((self._current_params, score))
        if score > self.best_score:
            self.best_score = score
            self.best_params = self._current_params

        self.population = self._select(scores + [(i, np.random.random()) for i in self.population])

        for i in range(len(self.population) - 1):
            if random() < 0.5:  # 50% chance to apply crossover
                self.population[i], self.population[i+1] = self._crossover(self.population[i], self.population[i+1])
            if random() < 0.1:  # 10% chance to apply mutation
                self.population[i] = self._mutate(self.population[i])