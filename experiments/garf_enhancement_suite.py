import os
import numpy as np
import pandas as pd
import seaborn as sns
from .suite import Suite
from factory import Factory
from create_db import datasets
import matplotlib.pyplot as plt
from insert_error import insert_error_unidetect
from methods.lake_cleaner.utils import type_token_ratio, uniqueness_score
from .garf_enhancement_experiment import GARFEnhancementExperiment as GARFExperiment


class GARFEnhancementSuite(Suite):
    def __init__(self, dataset_name: str, error_generator: int = 0):
        super().__init__("GARFEnhancement", "error_correction", dataset_name, error_generator)
        self.garf_base_dir = os.path.join(os.getcwd(), "methods", "garf_original")
        self._create_directory()
    
    def enhance_dataset(self, data: pd.DataFrame, data_clean: pd.DataFrame, errors: np.ndarray, n: int) -> pd.DataFrame:
        result_data = data.copy(deep=True)
        result_data_clean = data_clean.copy(deep=True)
        result_errors = errors.copy()
      
        result_data = pd.concat([result_data] * (n + 1), axis=0)
        result_data_clean = pd.concat([result_data_clean] * (n + 1), axis=0)
        result_errors = np.concatenate([result_errors] * (n + 1), axis=0)
            
        print("Dataset Lenghts: ", len(result_data), len(result_data_clean), len(result_errors))
            
        return result_data, result_data_clean, result_errors
    
    def plot(self):
        self._check_results()
        self._create_directory()
        sns.color_palette("tab10")
            
        data = self.results.copy()
        data = data.groupby(["enhancement_factor", "dataset"]).mean().reset_index()
        data["uniqueness_score"] = data["uniqueness_score"].round(4)
        data["type_token_ratio"] = data["type_token_ratio"].round(2)
        
        ax = sns.barplot(data=data, x="uniqueness_score", y="rule_count", hue="dataset", log=True)
        ax.set_title("Dataset Uniqueness Score vs. Number of Rules")
        ax.set_xlabel("Uniqueness Score")
        ax.set_ylabel("Rules Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_dir, "uniqueness_vs_rules_count.png"))
        plt.clf()
        
        ax = sns.barplot(data=data, x="type_token_ratio", y="rule_count", hue="dataset", log=True)
        ax.set_title("Dataset Type Token Ratio vs. Number of Rules")
        ax.set_xlabel("TTR")
        ax.set_ylabel("Rules Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_dir, "ttr_vs_rules_count.png"))
        plt.clf()
                
        ax = sns.lineplot(data=data, x="uniqueness_score", y="f1", hue="dataset", marker="o")
        ax.set_title("Dataset Uniqueness Score vs. F1")
        ax.set_xlabel("Uniqueness Score")
        ax.set_ylabel("F1")
        ax.set_ylim(0, 1)
        plt.savefig(os.path.join(self.base_dir, "uniqueness_vs_f1.png"))
        plt.clf()
        
        ax = sns.lineplot(data=data, x="type_token_ratio", y="f1", hue="dataset", marker="o")
        ax.set_title("Dataset Type Token Ratio vs. F1")
        ax.set_xlabel("TTR")
        ax.set_ylabel("F1")
        ax.set_ylim(0, 1)
        plt.savefig(os.path.join(self.base_dir, "ttr_vs_f1.png"))
        plt.clf()
        
        ax = sns.lineplot(data=data, x="enhancement_factor", y="f1", hue="dataset", marker="o")
        ax.set_title("Enhancement Factor vs. F1")
        ax.set_xlabel("Enhancement Factor")
        ax.set_ylabel("F1")
        ax.set_ylim(0, 1)
        plt.savefig(os.path.join(self.base_dir, "enhancement_factor_vs_f1.png"))
        plt.clf()
        
        ax = sns.lineplot(data=data, x="enhancement_factor", y="uniqueness_score", hue="dataset", marker="o")
        ax.set_title("Enhancement Factor vs. Uniqueness Score")
        ax.set_xlabel("Enhancement Factor")
        ax.set_ylabel("Uniqueness Score")
        ax.set_ylim(0, 1)
        plt.savefig(os.path.join(self.base_dir, "enhancement_factor_vs_uniqueness_score.png"))
        plt.clf()
    
        ax = sns.lineplot(data=data, x="enhancement_factor", y="type_token_ratio", hue="dataset", marker="o")
        ax.set_title("Enhancement Factor vs. Type Token Ratio")
        ax.set_xlabel("Enhancement Factor")
        ax.set_ylabel("TTR")
        plt.savefig(os.path.join(self.base_dir, "enhancement_factor_vs_type_token_ratio.png"))
        plt.clf()
        
    def run(self,
            max_error_rate: float,
            min_error_rate: float = 0.1, 
            error_step_size: float = 0.1, 
            runs_per_error_rate: int = 5, 
            error_intervals: int = 0, 
            **kwargs):
        N = 3
        dataset = pd.read_csv(datasets[self.dataset][0], dtype=str)
        #dataset = dataset.loc[:500]
        if "labelvalue" in dataset.columns:
            dataset = dataset.drop(columns=["labelvalue"])
        dataset["Label"] = ""
        n_tuples = len(dataset)
        dataset_dirty_ori, dataset_clean_ori, errors_ori = insert_error_unidetect(dataset, 0.01, self.garf_base_dir)
        
        self.results["uniqueness_score"] = []
        self.results["type_token_ratio"] = []
        self.results["rule_count"] = []
        self.results["enhancement_factor"] = []
        
        for i in range(0, N):
            dataset_dirty, dataset_clean, errors = self.enhance_dataset(dataset_dirty_ori, dataset_clean_ori, errors_ori, i)
            u_score = uniqueness_score(dataset_dirty)
            ttr = type_token_ratio(dataset_dirty)
            
            for j in range(runs_per_error_rate):
                print(f"Running experiment {j + 1}/{runs_per_error_rate} for decrease of uniqueness and ttr 0/{N-1}...")
                
                factory = Factory("garf-enhancement", self.dataset, self.error_generator)
                factory.prepare(dataset=dataset_dirty)
                
                experiment = GARFExperiment(
                    error_rate=i, 
                    method_name=f"garf_{i}_{j}", 
                    dataset=f"{self.dataset}_enhancement",
                    error_generator=self.error_generator
                )
                experiment.run(remove_amount_of_error_tuples=0, n_tuples=n_tuples, errors=errors, dataset_name=self.dataset, dataset_clean=dataset_clean, dataset_dirty_ori=dataset_dirty)
                self.results["error_rate"].append(0.01)
                self.results["precision"].append(experiment.precision)
                self.results["recall"].append(experiment.recall)
                self.results["f1"].append(experiment.f1)
                self.results["dataset"].append(self.dataset)
                self.results["runtime"].append(experiment.runtime)
                self.results["removed_error_tuples"].append(0)
                self.results["uniqueness_score"].append(u_score)
                self.results["type_token_ratio"].append(ttr)
                self.results["rule_count"].append(experiment.number_of_rules)
                self.results["enhancement_factor"].append(i)
                
                # save temp results
                pd.DataFrame(self.results).to_csv(os.path.join(self.base_dir, f"results.csv"), index=False)

        self.results = pd.DataFrame(self.results)
        return self