from itertools import product
from search.search import Search
from typing import Dict, List, Union


class GridSearch(Search):
    def __init__(self, params: Dict[str, List[Union[int, float]]], n: int = 0):
        super().__init__(params, n)
        self._all_params = self._generate_all_params()
        
    def _generate_all_params(self):
        # Generate all combinations of parameter values
        param_values = [vals for vals in self.params.values()]
        all_params = list(product(*param_values))
        return [dict(zip(self.params.keys(), param)) for param in all_params]
        
    def __iter__(self):
        print(self._all_params)
        for params in self._all_params:
            self._current_params = params
            yield params
            