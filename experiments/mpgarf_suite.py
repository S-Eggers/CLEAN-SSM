import numpy as np
import pandas as pd
import os
import sys
import signal
from concurrent import futures
from .suite import Suite
from .mpgarf_experiment import MPGARFExperiment


class MPGARFSuite(Suite):
    def __init__(self, dataset_name: str, error_generator: int = 0):
        super().__init__("GARF", "error_correction", dataset_name, error_generator)
        self.futures_list = []
        self.executor = None

    @staticmethod
    def _run_single(i, j, k, dataset, directory_id, error_generator):
        print(f"Running experiment {k + 1} with error rate {i} removing {j * 100}% of defective tuples from training set")
        experiment = MPGARFExperiment(
            error_rate=i, 
            method_name=f"garf_{i}_{j}_{k}", 
            dataset=dataset,
            error_generator=error_generator
        )
        experiment.run(remove_amount_of_error_tuples=j, directory_id=directory_id)
        return i, experiment.precision, experiment.recall, experiment.f1, experiment.runtime, dataset, j

    def signal_handler(self, sig, frame):
        print("Aborting all processes")
        for future in self.futures_list:
            future.cancel()
        self.executor.shutdown(wait=False)
        sys.exit(0)
    
    def run(self, 
            max_error_rate: float,
            min_error_rate: float = 0.1, 
            error_step_size: float = 0.1,
            runs_per_error_rate: int = 5, 
            error_intervals: int = 0, 
            **kwargs):
        signal.signal(signal.SIGINT, self.signal_handler)        
        if not os.path.exists(os.path.join(os.getcwd(), "log")):
            os.mkdir(os.path.join(os.getcwd(), "log"))

        n_parallel_jobs = kwargs["n_parallel_jobs"] if "n_parallel_jobs" in kwargs else 1
        max_workers = n_parallel_jobs if n_parallel_jobs < os.cpu_count() else os.cpu_count()
        print(f"Running multiprocessing GARF with {max_workers} parallel jobs")
        l = 0
        
        self.executor = futures.ProcessPoolExecutor(max_workers=max_workers)
        self.futures_list = []
        for i in self.generate_error_rates(max_error_rate, min_error_rate, error_step_size):
            for j in np.linspace(0, 1, error_intervals + 1):
                # j = round(j / 10, 1) # round to 1 decimal place
                for k in range(runs_per_error_rate):
                    future = self.executor.submit(self._run_single, i, j, k, self.dataset, str(l), self.error_generator)
                    self.futures_list.append(future)
                    l += 1

        for future in futures.as_completed(self.futures_list):
            i, precision, recall, f1, runtime, dataset, j = future.result()
            self.results["error_rate"].append(i)
            self.results["precision"].append(precision)
            self.results["recall"].append(recall)
            self.results["f1"].append(f1)
            self.results["runtime"].append(runtime)
            self.results["dataset"].append(dataset)
            self.results["removed_error_tuples"].append(j)

        self.results = pd.DataFrame(self.results)
        return self