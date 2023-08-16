from abc import ABC, abstractmethod
from multiprocessing import Queue
from typing import Tuple, Dict, Any


class Experiment(ABC):
    def __init__(self, error_rate: float, method_name: str, dataset: str, error_generator: int = 1):
        self.error_rate = error_rate
        self.method_name = method_name
        self.dataset = dataset
        self.error_generator = error_generator
    
    @abstractmethod
    def run(self, **kwargs):
        pass
    
    @staticmethod
    @abstractmethod
    def worker(queue: Queue, error_rate: float = 0.1, insert_errors: bool = True, kwargs: Dict[str, Any] = dict()):
        pass

    @abstractmethod
    def result(self) -> Tuple[float, float, float, float]:
        pass
