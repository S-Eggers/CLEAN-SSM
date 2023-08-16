import numpy as np
from search.search import Search
from typing import Dict, List, Union


class RandomSearch(Search):
    def __init__(self, params: Dict[str, List[Union[int, float]]], n: int = 12):
        super().__init__(params, n)
        self.max_iter = n
        
    def __iter__(self):
        for _ in range(self.max_iter):
            params = {param: np.random.choice(values) for param, values in self.params.items()}
            self._current_params = params
            yield params
            