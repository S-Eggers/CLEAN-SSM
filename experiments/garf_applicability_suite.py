import os
import numpy as np
import pandas as pd
import seaborn as sns
from .suite import Suite
from factory import Factory
from create_db import datasets
import matplotlib.pyplot as plt
from methods.lake_cleaner.utils import type_token_ratio, uniqueness_score
from .garf_applicability_experiment import GARFApplicabilityExperiment as GARFExperiment


class GARFApplicabilitySuite(Suite):
    def __init__(self, dataset_name: str, error_generator: int = 0):
        super().__init__("GARFApplicability", "error_correction", dataset_name, error_generator)
        self._create_directory()
    
    def decrease_uniqueness(self, data: pd.DataFrame, n: int, id_column: str) -> pd.DataFrame:
        return data.groupby(id_column).apply(lambda x: x.sample(min(len(x), n))).reset_index(drop=True)
    
    def plot(self):
        self._check_results()
        self._create_directory()
        colors = sns.color_palette("tab10")
        palette = {
            "Beers": colors[0], 
            "Flights": colors[1],
            "Food": colors[2],
            "Hospital": colors[3],
            "Rayyan": colors[4],
        }
            
        data = self.results.copy()
        data = data.groupby(["uniqueness_score", "dataset"]).mean().reset_index()
        data["uniqueness_score"] = data["uniqueness_score"].round(4)
        data["type_token_ratio"] = data["type_token_ratio"].round(2)
        
        plt.figure(figsize=(8, 5))
        ax = sns.barplot(data=data, x="uniqueness_score", y="rule_count", hue="dataset", palette=palette)
        ax.set_title("Dataset Uniqueness Score vs. Number of Rules")
        ax.set_xlabel("Uniqueness Score")
        ax.set_ylabel("Rules Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_dir, "uniqueness_vs_rules_count.png"))
        plt.clf()
        plt.close()
        
        plt.figure(figsize=(8, 5))
        ax = sns.barplot(data=data, x="type_token_ratio", y="rule_count", hue="dataset", palette=palette)
        ax.set_title("Dataset Type Token Ratio vs. Number of Rules")
        ax.set_xlabel("TTR")
        ax.set_ylabel("Rules Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_dir, "ttr_vs_rules_count.png"))
        plt.clf()
        plt.close()
        
        ax = sns.lineplot(data=data, x="uniqueness_score", y="f1", hue="dataset", marker="o", palette=palette)
        ax.set_title("Dataset Uniqueness Score vs. F1")
        ax.set_xlabel("Uniqueness Score")
        ax.set_ylabel("F1")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_dir, "uniqueness_vs_f1.png"))
        plt.clf()
        
        ax = sns.lineplot(data=data, x="type_token_ratio", y="f1", hue="dataset", marker="o", palette=palette)
        ax.set_title("Dataset Type Token Ratio vs. F1")
        ax.set_xlabel("TTR")
        ax.set_ylabel("F1")
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.base_dir, "ttr_vs_f1.png"))
        plt.clf()
        
    def run(self,
            max_error_rate: float,
            min_error_rate: float = 0.1, 
            error_step_size: float = 0.1, 
            runs_per_error_rate: int = 5, 
            error_intervals: int = 0, 
            **kwargs):
        N = 10
        keys = {
            "hospital": "providerid",
            "food": "license",
            "flights": "flight",
            "beers": "breweryid",
            "rayyan": "id",
        }
        self.results["uniqueness_score"] = []
        self.results["type_token_ratio"] = []
        self.results["rule_count"] = []
        dataset_key = self.dataset.lower()
        dataset = pd.read_csv(datasets[self.dataset][0], dtype=str)
        max_key_count = dataset[keys[dataset_key]].value_counts().max()
        int_numbers = []
        numbers = np.linspace(1, max_key_count, N)
        for number in numbers:
            rounded = int(np.round(number))
            if rounded not in int_numbers:
                int_numbers.append(rounded)
            else:
                if rounded + 1 <= max_key_count and rounded + 1 not in int_numbers:
                    int_numbers.append(rounded + 1)
                elif rounded - 1 >= 1 and rounded - 1 not in int_numbers:
                    int_numbers.append(rounded - 1)
        
        int_numbers = reversed(sorted(int_numbers))
        
        for i in int_numbers:
            dataset = self.decrease_uniqueness(dataset, i, keys[dataset_key])
            u_score = uniqueness_score(dataset)
            ttr = type_token_ratio(dataset)
            
            for j in range(runs_per_error_rate):
                print(f"Running experiment {j + 1}/{runs_per_error_rate} for decrease of uniqueness and ttr {max_key_count - i}/{max_key_count - 1}...")
                
                factory = Factory("garf-applicability", self.dataset, self.error_generator)
                factory.prepare(dataset=dataset)
                
                experiment = GARFExperiment(
                    error_rate=i, 
                    method_name=f"garf_{i}_{j}", 
                    dataset=f"{self.dataset}_applicability",
                    error_generator=self.error_generator
                )
                experiment.run(remove_amount_of_error_tuples=0)
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
                # save temp results
                pd.DataFrame(self.results).to_csv(os.path.join(self.base_dir, f"results.csv"), index=False)

        self.results = pd.DataFrame(self.results)
        return self