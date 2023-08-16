import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .suite import Suite
from typing import Dict, Union
from .rule_reusage_experiment import RuleReusageExperiment


class RuleReusageSuite(Suite):
    def __init__(self, dataset_name: str, error_generator: int = 0):
        super().__init__("GARF", "rule_coverage", dataset_name, error_generator)

    def run(self,
            max_error_rate: float,
            min_error_rate: float = 0.1, 
            error_step_size: float = 0.1, 
            runs_per_error_rate: int = 5, 
            error_intervals: int = 0, 
            **kwargs):
        experiment = RuleReusageExperiment(np.inf, "rule_coverage", self.dataset)
        experiment.run()
        self.results = experiment.result()
        return self
    
    def merge(self, suite: Suite):
        self._check_results()
        suite._check_results()
        
        row_results, random_nan_results, col_results = self.results
        suite_row_results, suite_random_nan_results, suite_col_results = suite.results
        row_results = pd.concat([row_results, suite_row_results])
        random_nan_results = pd.concat([random_nan_results, suite_random_nan_results])
        col_results = pd.concat([col_results, suite_col_results])
        self.results = row_results, random_nan_results, col_results
        return self
    
    def save(self, arguments: Dict) -> pd.DataFrame:
        self._check_results()
        self._create_directory()
        
        row_results, random_nan_results, col_results = self.results
        row_results.to_csv(os.path.join(self.base_dir, "row_results.csv"))
        random_nan_results.to_csv(os.path.join(self.base_dir, "random_nan_results.csv"))
        col_results.to_csv(os.path.join(self.base_dir, "col_results.csv"))
        
        with open(os.path.join(self.base_dir, "konfiguration.json"), "w") as file:
            json.dump(arguments, file)
        return self.results        
    
    def debug_plot(self, time: Union[str, int]):
        self.set_result_path(time)
        col_results = pd.read_csv(os.path.join(self.base_dir, "col_results.csv"), index_col=0)
        row_results = pd.read_csv(os.path.join(self.base_dir, "row_results.csv"), index_col=0)
        random_nan_results = pd.read_csv(os.path.join(self.base_dir, "random_nan_results.csv"), index_col=0)
        self.results = row_results, random_nan_results, col_results
        self.plot()
    
    @staticmethod
    def reverse_remove(df):
        max_value = df['remove'].max() + 1
        df['remove'] = abs(df['remove'] - max_value)
        return df
    
    def plot(self):
        self._check_results()
        self._create_directory()
        sns.color_palette("tab10")
        row_results, random_nan_results, col_results = self.results
        
        max_row = pd.DataFrame({
            "dataset": row_results["dataset"].unique(),
            "remove": [1.0] * len(row_results["dataset"].unique()),
            "coverage": [0.0] * len(row_results["dataset"].unique())
        })
        row_results = row_results.append(max_row)
        random_nan_results = random_nan_results.append(max_row)

        col_results = col_results.groupby(["remove", "dataset"])["coverage"].mean().reset_index()
        col_results = col_results.groupby("dataset").apply(self.reverse_remove)
        min_row = pd.DataFrame({
            "dataset": row_results["dataset"].unique(),
            "remove": [0] * len(row_results["dataset"].unique()),
            "coverage": [1.0] * len(row_results["dataset"].unique())
        })
        col_results = col_results.append(min_row)
        col_results = col_results.rename(columns={"coverage": "Rule Coverage", "remove": "Removed Columns", "dataset": "Dataset"})
        sns.lineplot(
            data=col_results,
            x="Removed Columns",
            y="Rule Coverage",
            hue="Dataset",
            marker="o"
        )
        plt.ylim(0, 1.1)
        self._save_plot(os.path.join(self.base_dir, f"rule_coverage_col_removal.png"))
        
        row_result = row_results.groupby(["remove", "dataset"])["coverage"].mean().reset_index()
        row_result = row_result.rename(columns={"coverage": "Rule Coverage", "remove": "Removed Tuples (in \%)", "dataset": "Dataset"})
        sns.lineplot(
            data=row_result,
            x="Removed Tuples (in \%)",
            y="Rule Coverage",
            hue="Dataset",
            marker="o"
        )
        plt.ylim(0, 1.1)
        self._save_plot(os.path.join(self.base_dir, f"rule_coverage_tuple_removal.png"))
        
        random_nan_results = random_nan_results.groupby(["remove", "dataset"])["coverage"].mean().reset_index()
        random_nan_results = random_nan_results.rename(columns={"coverage": "Rule Coverage", "remove": "nan-Cells (in \%)", "dataset": "Dataset"})
        sns.lineplot(
            data=random_nan_results,
            x="nan-Cells (in \%)",
            y="Rule Coverage",
            hue="Dataset",
            marker="o"
        )
        plt.ylim(0, 1.1)
        self._save_plot(os.path.join(self.base_dir, f"rule_coverage_cell_removal.png"))