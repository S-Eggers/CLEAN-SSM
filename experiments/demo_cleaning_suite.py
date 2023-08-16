import numpy as np
import pandas as pd
from .suite import Suite
from factory import Factory
from .demo_cleaning import DemoCleaning


class DemoCleaningSuite(Suite):
    def __init__(self, dataset_name: str, error_generator: int = 0):
        super().__init__("GARF-HW", "error_correction", "HW", error_generator)

    def run(self,
            max_error_rate: float,
            min_error_rate: float = 0.1, 
            error_step_size: float = 0.1, 
            runs_per_error_rate: int = 5, 
            error_intervals: int = 0, 
            **kwargs):                   
        factory = Factory("demo-claening", self.dataset, self.error_generator)
        factory.prepare(limit=10000)
        
        experiment = DemoCleaning(
            error_rate=0.1, 
            method_name=f"garf_{self.dataset}_{0.1}", 
            dataset=self.dataset,
            error_generator=self.error_generator
        )
        experiment.run(remove_amount_of_error_tuples=0)

        self.results = pd.DataFrame()
        
        return self