import numpy as np
import pandas as pd
from .suite import Suite
from .garf_experiment import GARFExperiment
from search.random_search import RandomSearch
from search.grid_search import GridSearch
from search.genetic_search import GeneticSearch
import os


class HypeparameterSearchGARFSuite(Suite):
    def __init__(self, dataset_name: str, error_generator: int = 1):
        super().__init__("GARF", "error_correction", dataset_name, error_generator)

    def run(self, 
            max_error_rate: float,
            min_error_rate: float = 0.1, 
            error_step_size: float = 0.1, 
            runs_per_error_rate: int = 5, 
            error_intervals: int = 0, 
            **kwargs):      
        if "mode" in kwargs and kwargs["mode"] == "random":
            hyperparameter_search = RandomSearch({
                "g_e": [8, 16, 32, 64],
                "g_h": [8, 16, 32, 64],
            }, n=12)
        elif "mode" in kwargs and kwargs["mode"] == "genetic":
            hyperparameter_search = GeneticSearch({
                "g_e": [8, 16, 32, 64],
                "g_h": [8, 16, 32, 64],
            }, n=50)
        else:
            hyperparameter_search = GridSearch({
                "g_e": [16, 32, 64],
                "g_h": [16, 32, 64],
            })
        
        for params in hyperparameter_search:
            print(f"Running experiment with params {params}")
            for i in self.generate_error_rates(max_error_rate, min_error_rate, error_step_size):
                for j in np.linspace(0, 1, error_intervals + 1):
                    # j = round(j / 10, 1) # round to 1 decimal place
                    for k in range(runs_per_error_rate):
                        print(f"Running experiment {k + 1}/{runs_per_error_rate} for error rate {i}/{max_error_rate} with {j} defective tuples not in train set...")
                        experiment = GARFExperiment(
                            error_rate=i, 
                            method_name=f"garf_{i}_{j}_{k}", 
                            dataset=self.dataset,
                            error_generator=self.error_generator
                        )
                        experiment.run(remove_amount_of_error_tuples=j, **params)
                        self.results["error_rate"].append(i)
                        self.results["precision"].append(experiment.precision)
                        self.results["recall"].append(experiment.recall)
                        self.results["f1"].append(experiment.f1)
                        self.results["runtime"].append(experiment.runtime)
                        self.results["dataset"].append(self.dataset)
                        self.results["removed_error_tuples"].append(j)
            
            hyperparameter_search.receive_results(self.results)
            self.reset_results()

        print(f"Finished hyperparameter search, best params: {hyperparameter_search.get_best_params()}, best f1 score: {hyperparameter_search.get_best_score()}")
        hyperparameter_search.plot(os.path.join("results",self.task, self.name))
        self.results = pd.DataFrame(self.results)
        return self
    
    def reset_results(self):
        self.results = {
            "error_rate": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "runtime": [],
            "dataset": [],
            "removed_error_tuples": [],
        }