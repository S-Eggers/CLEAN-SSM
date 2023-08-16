import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from .suite import Suite
from factory import Factory
from .garf_extra_measurements_experiment import GARFExperiment


class TaxGARFSuite(Suite):
    def __init__(self, dataset_name: str, error_generator: int = 0):
        super().__init__("TaxGARF", "error_correction", "Tax", error_generator)
        self._create_directory()
        self.suppress_plotting = True
        
    def plot(self):
        self._check_results()
        self._create_directory()
        sns.color_palette("tab10")
        
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
            x="dataset_length",
            y="runtime",
            marker="o"
        )
        ax.set_title("Tuples vs. Runtime (Tax)")
        ax.set_xlabel("Tuples")
        ax.set_ylabel("Runtime (s)")
        plt.savefig(os.path.join(self.base_dir, "tuples_runtime.png"))
        plt.clf()
        
        ax = sns.barplot(
            data=self.results,
            x="dataset_length",
            y="v",
            # hue="dataset_length"
            # marker="o"
        )
        ax.set_title("Tuples vs. Vocabulary Size (Tax)")
        ax.set_xlabel("Tuples")
        ax.set_ylabel("Vocabulary Size")
        plt.savefig(os.path.join(self.base_dir, "tuples_v.png"))
        plt.clf()
                       
        return self

    def run(self,
            max_error_rate: float,
            min_error_rate: float = 0.1, 
            error_step_size: float = 0.1, 
            runs_per_error_rate: int = 5, 
            error_intervals: int = 0, 
            **kwargs):   
        self.results["dataset_length"] = []
        self.results["pretraining"] = []
        self.results["adversarial"] = []
        self.results["rule"] = []
        self.results["rule_filter"] = []
        self.results["repair"] = []
        self.results["v"] = []
        self.results["lstm_weights"] = []
        
        for dataset_length in [1000, 2500, 5000, 10000, 25000, 50000, 100000, -1]:
            factory = Factory("garf-original", self.dataset, self.error_generator)
            factory.prepare(limit=dataset_length)
            
            experiment = GARFExperiment(
                error_rate=0.1, 
                method_name=f"garf_{self.dataset}_{0.1}_{dataset_length}", 
                dataset=self.dataset,
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
            self.results["dataset_length"].append(dataset_length)
            self.results["pretraining"].append(experiment.pretraining)
            self.results["adversarial"].append(experiment.adversarial)
            self.results["rule"].append(experiment.rule)
            self.results["rule_filter"].append(experiment.rule_filter)
            self.results["repair"].append(experiment.repair)
            self.results["v"].append(experiment.v)
            self.results["lstm_weights"].append(experiment.lstm_weights)
            
            part_result = pd.DataFrame(self.results)
            part_result.to_csv(os.path.join(self.base_dir, f"results_{dataset_length}.csv"), index=False)
        
        self.results = pd.DataFrame(self.results)
        return self