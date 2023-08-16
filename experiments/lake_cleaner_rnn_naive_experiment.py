import os
from typing import Tuple, Dict, Any
from .experiment import Experiment
from methods.lake_cleaner_rnn_naive.main import main as lake_cleaner_naive


class LakeCleanerRNNNaiveExperiment(Experiment):
    def __init__(self, error_rate: float, method_name: str, dataset: str, error_generator: int = 1):
        super().__init__(error_rate, method_name, dataset, error_generator)
        self.base_dir = os.path.join(os.getcwd(), "methods", "lake_cleaner_rnn_naive")
        self.lake_dataset_path = os.path.join(self.base_dir, "datasets", "artificial_lake")

    def run(self, **kwargs):
        results = lake_cleaner_naive(self.base_dir, self.lake_dataset_path)
        self.results = results
    
    @staticmethod
    def worker(kwargs: Dict[str, Any] = dict()):
        pass
    
    def result(self) -> Tuple[float, float, float, float]:
        return self.results

    
