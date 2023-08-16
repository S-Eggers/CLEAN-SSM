import numpy as np
import pandas as pd
from .suite import Suite
from .transfer_learning_garf_experiment import TransferLearningGARFExperiment


class TransferLearningGARFSuite(Suite):
    def __init__(self, dataset_name: str, error_generator: int = 0):
        super().__init__("Transfer-Learning-GARF", "error_correction", dataset_name, error_generator)

    def run(self, 
            max_error_rate: float, 
            min_error_rate: float = 0.1, 
            error_step_size: float = 0.1, 
            runs_per_error_rate: int = 5, 
            error_intervals: int = 0, 
            **kwargs):        
        for i in self.generate_error_rates(max_error_rate, min_error_rate, error_step_size):
            for j in np.linspace(0, 1, error_intervals + 1):
                # j = round(j / 10, 1) # round to 1 decimal place
                for k in range(runs_per_error_rate):
                    print(f"Running experiment {k + 1}/{runs_per_error_rate} for error rate {i}/{max_error_rate} with {j} defective tuples not in train set...")
                    experiment = TransferLearningGARFExperiment(
                        error_rate=i, 
                        method_name=f"garf_{i}_{j}_{k}", 
                        dataset=self.dataset,
                        error_generator=self.error_generator
                    )
                    experiment.run(remove_amount_of_error_tuples=j)
                    self.results["error_rate"].append(i)
                    self.results["precision"].append(experiment.precision)
                    self.results["recall"].append(experiment.recall)
                    self.results["f1"].append(experiment.f1)
                    self.results["runtime"].append(experiment.runtime)
                    self.results["dataset"].append(self.dataset)
                    self.results["removed_error_tuples"].append(j)

        self.results = pd.DataFrame(self.results)
        return self