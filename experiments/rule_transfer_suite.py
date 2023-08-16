import numpy as np
import pandas as pd
from .suite import Suite
from factory import Factory
from .rule_transfer_experiment import RuleTransferExperiment


class RuleTransferSuite(Suite):
    def __init__(self, dataset_name: str, error_generator: int = 0):
        super().__init__("RuleTransfer", "rule_coverage", dataset_name, error_generator)
        self.suppress_plotting = True

    def run(self,
            max_error_rate: float,
            min_error_rate: float = 0.1, 
            error_step_size: float = 0.1, 
            runs_per_error_rate: int = 5, 
            error_intervals: int = 0, 
            **kwargs):
        experiment = RuleTransferExperiment(0.1, "RuleTransfer", self.dataset)
        experiment.run()
        self.results = experiment.result()
        return self