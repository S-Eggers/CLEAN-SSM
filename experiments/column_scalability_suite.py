import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .suite import Suite
from factory import Factory
from .garf_extra_measurements_experiment import GARFExperiment


class ColumnScalabilitySuite(Suite):
    def __init__(self, dataset_name: str, error_generator: int = 0):
        super().__init__("ColumnScalability", "error_correction", "Tax", error_generator)
        self._create_directory()
        self.suppress_plotting = True
        
    def plot(self):
        self._check_results()
        self._create_directory()
        sns.color_palette("tab10")
        
        # calculate mean
        self.results = self.results.groupby(["num_columns"]).mean().reset_index()
        ax = sns.lineplot(
            data=self.results,
            x="v",
            y="runtime",
            marker="o"
        )
        ax.set_title("Vocabulary Size vs. Runtime (Tax)")
        ax.set_xlabel("Vocabulary Size")
        ax.set_ylabel("Runtime (s)")
        plt.savefig(os.path.join(self.base_dir, "v_runtime.png"))
        plt.clf()
        
        ax = sns.lineplot(
            data=self.results,
            x="lstm_weights",
            y="runtime",
            marker="o"
        )
        ax.set_title("LSTM Size vs. Runtime (Tax)")
        ax.set_xlabel("LSTM Size")
        ax.set_ylabel("Runtime (s)")
        plt.savefig(os.path.join(self.base_dir, "lstm_runtime.png"))
        plt.clf()
        
        ax = sns.lineplot(
            data=self.results,
            x="num_columns",
            y="runtime",
            marker="o"
        )
        ax.set_title("Columns vs. Runtime (Tax)")
        ax.set_xlabel("Columns")
        ax.set_ylabel("Runtime (s)")
        plt.savefig(os.path.join(self.base_dir, "columns_runtime.png"))
        plt.clf()
        
        ax = sns.barplot(
            data=self.results,
            x="num_columns",
            y="v",
            # hue="dataset_length"
            # marker="o"
        )
        ax.set_title("Columns vs. Vocabulary Size (Tax)")
        ax.set_xlabel("Columns")
        ax.set_ylabel("Vocabulary Size")
        plt.savefig(os.path.join(self.base_dir, "columns_v.png"))
        plt.clf()
                       
        return self

    def run(self,
            max_error_rate: float,
            min_error_rate: float = 0.1, 
            error_step_size: float = 0.1, 
            runs_per_error_rate: int = 5, 
            error_intervals: int = 0, 
            **kwargs):   
        self.results["num_columns"] = []
        self.results["pretraining"] = []
        self.results["adversarial"] = []
        self.results["rule"] = []
        self.results["rule_filter"] = []
        self.results["repair"] = []
        self.results["v"] = []
        self.results["lstm_weights"] = []
        
        for num_columns in range(2, 16):
            for i in range(runs_per_error_rate):
                factory = Factory("column-scalability", self.dataset, self.error_generator)
                factory.prepare(limit=num_columns)
                
                experiment = GARFExperiment(
                    error_rate=0.1, 
                    method_name=f"garf_{self.dataset}_{0.1}_{num_columns}", 
                    dataset=f"{self.dataset} ({num_columns} Columns)",
                    error_generator=self.error_generator
                )
                experiment.run(remove_amount_of_error_tuples=0)
                self.results["error_rate"].append(0.1)
                self.results["precision"].append(experiment.precision)
                self.results["recall"].append(experiment.recall)
                self.results["f1"].append(experiment.f1)
                self.results["runtime"].append(experiment.runtime)
                self.results["dataset"].append(self.dataset)
                self.results["removed_error_tuples"].append(0)
                self.results["num_columns"].append(num_columns)
                self.results["pretraining"].append(experiment.pretraining)
                self.results["adversarial"].append(experiment.adversarial)
                self.results["rule"].append(experiment.rule)
                self.results["rule_filter"].append(experiment.rule_filter)
                self.results["repair"].append(experiment.repair)
                self.results["v"].append(experiment.v)
                self.results["lstm_weights"].append(experiment.lstm_weights)
            
            part_result = pd.DataFrame(self.results)
            part_result.to_csv(os.path.join(self.base_dir, f"results_{num_columns}.csv"), index=False)
        
        self.results = pd.DataFrame(self.results)
        return self