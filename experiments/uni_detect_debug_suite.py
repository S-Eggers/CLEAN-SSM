import os
import numpy as np
import pandas as pd
from .suite import Suite
from .uni_detect_experiment import UniDetectExperiment


class UniDetectDebugSuite(Suite):
    def __init__(self, dataset_name: str, error_generator: int = 0):
        super().__init__("UniDetect", "error_detection", dataset_name, error_generator)

    def run(self, 
            max_error_rate: float, 
            min_error_rate: float = 0.1, 
            error_step_size: float = 0.1, 
            runs_per_error_rate: int = 5, 
            error_intervals: int = 0, 
            **kwargs):
        
        dataset = pd.read_csv(os.path.join(os.getcwd(), "datasets", "toy", "clean.csv"))
        dataset = dataset.drop(columns=["ID"])
        # dataset.iloc[1, 1] = np.nan
        print(dataset.head(6))
        self.base_dir = os.path.join(os.getcwd(), "methods", "uni_detect_fatemeh")
        dataset_name = "debug_dataset"
        dataset_base_path = os.path.join(self.base_dir, "datasets", "artificial_lake", dataset_name)
        os.makedirs(dataset_base_path, exist_ok=True)
        dataset_path = os.path.join(dataset_base_path, "clean.csv")
        dataset.to_csv(dataset_path, index=False)
        
        experiment = UniDetectExperiment(
            error_rate=0, 
            method_name=f"uni_detect", 
            dataset=dataset_name,
            error_generator=self.error_generator
        )
        experiment.run(remove_amount_of_error_tuples=0)
        
        dataset.iloc[2, 1] = dataset.iloc[1, 1]
        # dataset.iloc[1, 1] = np.nan
        print(dataset.head(6))
        self.base_dir = os.path.join(os.getcwd(), "methods", "uni_detect_fatemeh")
        dataset_name = "debug_dataset"
        dataset_base_path = os.path.join(self.base_dir, "datasets", "artificial_lake", dataset_name)
        os.makedirs(dataset_base_path, exist_ok=True)
        dataset_path = os.path.join(dataset_base_path, "clean.csv")
        dataset.to_csv(dataset_path, index=False)
        
        experiment = UniDetectExperiment(
            error_rate=0, 
            method_name=f"uni_detect", 
            dataset=dataset_name,
            error_generator=self.error_generator
        )
        experiment.run(remove_amount_of_error_tuples=0)
        
        dataset = pd.read_csv(os.path.join(os.getcwd(), "datasets", "toy", "clean.csv"))
        dataset = dataset.drop(columns=["ID"])
        dataset.iloc[1, 1] = np.nan
        print(dataset.head(6))
        self.base_dir = os.path.join(os.getcwd(), "methods", "uni_detect_fatemeh")
        dataset_name = "debug_dataset"
        dataset_base_path = os.path.join(self.base_dir, "datasets", "artificial_lake", dataset_name)
        os.makedirs(dataset_base_path, exist_ok=True)
        dataset_path = os.path.join(dataset_base_path, "clean.csv")
        dataset.to_csv(dataset_path, index=False)
        
        experiment = UniDetectExperiment(
            error_rate=0, 
            method_name=f"uni_detect", 
            dataset=dataset_name,
            error_generator=self.error_generator
        )
        experiment.run(remove_amount_of_error_tuples=0)
        
        
        dataset = pd.read_csv(os.path.join(os.getcwd(), "datasets", "toy", "clean.csv"))
        dataset = dataset.drop(columns=["ID"])
        dataset.iloc[1, 1] = "Mooooooooooooordoooooooooooooooooooooooor"
        print(dataset.head(6))
        self.base_dir = os.path.join(os.getcwd(), "methods", "uni_detect_fatemeh")
        dataset_name = "debug_dataset"
        dataset_base_path = os.path.join(self.base_dir, "datasets", "artificial_lake", dataset_name)
        os.makedirs(dataset_base_path, exist_ok=True)
        dataset_path = os.path.join(dataset_base_path, "clean.csv")
        dataset.to_csv(dataset_path, index=False)
        
        experiment = UniDetectExperiment(
            error_rate=0, 
            method_name=f"uni_detect", 
            dataset=dataset_name,
            error_generator=self.error_generator
        )
        experiment.run(remove_amount_of_error_tuples=0)

        self.results = pd.DataFrame(self.results)
        return self