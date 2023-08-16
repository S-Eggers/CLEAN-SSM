import os
import numpy as np
import pandas as pd
import seaborn as sns
from .suite import Suite
from factory import Factory
from create_db import datasets
import matplotlib.pyplot as plt
from insert_error import insert_specific_error
from methods.lake_cleaner.utils import type_token_ratio, uniqueness_score
from .garf_enhancement_experiment import GARFEnhancementExperiment as GARFExperiment


class GARFErrorTypeSuite(Suite):
    def __init__(self, dataset_name: str, error_generator: int = 0):
        super().__init__("GARFErrorTypes", "error_correction", dataset_name, error_generator)
        self.garf_base_dir = os.path.join(os.getcwd(), "methods", "garf_original")
        self._create_directory()
        
    def plot(self):        
        self._check_results()
        print(self.base_dir)
        
        datasets = self.results["dataset"].unique()
        all_results = self.results.copy()
        for dataset in datasets:
            # filter results for each dataset
            self.results = self.results[self.results["dataset"] == dataset]
            # plot results for each dataset
            self._aggregated_combined_plots(dataset)
            # reset results dataframe
            self.results = all_results.copy()
        
        return self
    
    def _aggregated_combined_plots(self, dataset: str):
        results_grouped = self.results # self.results.groupby(["error_rate", "error_type"])
        mean_results = results_grouped # results_grouped.agg({"precision": "mean", "recall": "mean", "f1": "mean", "runtime": "mean"})
        # mean_results = mean_results.reset_index()
        mean_results = mean_results.rename(columns={"error_type": "Error Type"})
                
        mean_results["Error Type"] = mean_results["Error Type"].replace({
            "missing": "Missing Value", 
            "cell_swap": "Cell Swap (L+R)", 
            "spelling": "Misspelling", 
            "functional_dependency": "FD Violation", 
            "outlier": "Outlier",
        })
        colors = sns.color_palette("tab10")
        palette = {
            "Missing Value": colors[0], 
            "Cell Swap (L+R)": colors[1],
            "Misspelling": colors[2],
            "FD Violation": colors[3],
            "Outlier": colors[4],
        }
     
        x_ticks = list(mean_results["error_rate"].unique())
        self._plot_combined_avg_metric(mean_results, "recall", f"Recall on {dataset}", "Error Rate", "Recall", x_ticks, palette, hue="Error Type")
        self._save_plot(os.path.join(self.base_dir, f"recall_combined_{dataset}.png"))
        self._plot_combined_avg_metric(mean_results, "precision", f"Precision on {dataset}", "Error Rate", "Precision", x_ticks, palette, hue="Error Type")
        self._save_plot(os.path.join(self.base_dir, f"precision_combined_{dataset}.png"))
        self._plot_combined_avg_metric(mean_results, "f1", f"F1 on {dataset}", "Error Rate", "F1", x_ticks, palette, hue="Error Type")
        self._save_plot(os.path.join(self.base_dir, f"f1_combined_{dataset}.png"))
        self._plot_combined_avg_runtime(mean_results, "runtime", f"Runtime on {dataset}", "Error Rate", "Runtime (in Seconds)", x_ticks, palette, hue="Error Type")
        self._save_plot(os.path.join(self.base_dir, f"runtime_combined_{dataset}.png"))        
            
    def count_nonzero_values(self, arr: np.ndarray) -> dict:
        if len(arr.shape) != 2:
            raise ValueError("Input array must be 2-dimensional")

        non_zero_counts = np.count_nonzero(arr, axis=0)
        result = {idx: count for idx, count in enumerate(non_zero_counts) if count > 0}

        return result
            
    def run(self,
            max_error_rate: float,
            min_error_rate: float = 0.1, 
            error_step_size: float = 0.1, 
            runs_per_error_rate: int = 5, 
            error_intervals: int = 0, 
            **kwargs):

        error_types = ["missing", "cell_swap", "spelling", "functional_dependency", "outlier"]
        
        dataset = pd.read_csv(datasets[self.dataset][0])
        #dataset = dataset.loc[:500]
        if "labelvalue" in dataset.columns:
            dataset = dataset.drop(columns=["labelvalue"])
        dataset["Label"] = ""
        n_tuples = len(dataset)
        
        self.results["error_type"] = []
        self.results["dirty_cells"] = []
        self.results["clean_cells"] = []
        self.results["affected_columns"] = []
        
        for error_type in error_types:
            if (error_type == "outlier" and self.dataset.lower() != "beers") or (error_type == "transformation" and self.dataset.lower() not in ["hospital", "rayyan", "food"]):
                continue

            for i in self.generate_error_rates(max_error_rate, min_error_rate, error_step_size):
                dataset_dirty, dataset_clean, errors = insert_specific_error(dataset.copy(), i, self.garf_base_dir, error_type)
                dataset_dirty = dataset_dirty.astype(str)
                dataset_clean = dataset_clean.astype(str)
                
                for j in range(runs_per_error_rate):
                    print(f"Running experiment {j + 1}/{runs_per_error_rate} for error rate {i} and error type {error_type}...")
                    
                    factory = Factory("garf-enhancement", self.dataset, self.error_generator)
                    factory.prepare(dataset=dataset_dirty)
                    
                    experiment = GARFExperiment(
                        error_rate=i, 
                        method_name=f"garf_{i}_{j}", 
                        dataset=f"{self.dataset}_enhancement",
                        error_generator=self.error_generator
                    )
                    experiment.run(remove_amount_of_error_tuples=0, n_tuples=n_tuples, errors=errors, dataset_name=self.dataset, dataset_clean=dataset_clean, dataset_dirty_ori=dataset_dirty)
                    self.results["error_rate"].append(i)
                    self.results["precision"].append(experiment.precision)
                    self.results["recall"].append(experiment.recall)
                    self.results["f1"].append(experiment.f1)
                    self.results["dataset"].append(self.dataset)
                    self.results["runtime"].append(experiment.runtime)
                    self.results["removed_error_tuples"].append(0)
                    self.results["error_type"].append(error_type)
                    self.results["dirty_cells"].append(np.count_nonzero(errors))
                    self.results["clean_cells"].append((errors.shape[0] *(errors.shape[1] - 1)) - np.count_nonzero(errors))
                    self.results["affected_columns"].append(str(self.count_nonzero_values(errors)))
                    
                    # save temp results
                    pd.DataFrame(self.results).to_csv(os.path.join(self.base_dir, f"results.csv"), index=False)

        self.results = pd.DataFrame(self.results)
        return self
