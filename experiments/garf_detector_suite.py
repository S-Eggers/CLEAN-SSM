import os
import numpy as np
import pandas as pd
from .suite import Suite
from .garf_experiment import GARFExperiment
from insert_error import DETECTION
import seaborn as sns


class GARFDetectorSuite(Suite):
    def __init__(self, dataset_name: str, error_generator: int = DETECTION):
        super().__init__("GARF", "error_detection", dataset_name, error_generator)

    def combined_plot(self):
        if self.suppress_plotting:
            return self
        self._check_results()
        print(self.base_dir)
        self.results["dataset"] = self.results["dataset"].str.capitalize()
        self._aggregated_combined_plots()
        
        return self
    
    def _aggregated_combined_plots(self):
        
        colors = sns.color_palette("tab10")
        palette = {
            "Beers": colors[0], 
            "Flights": colors[1],
            "Food": colors[2],
            "Hospital": colors[3],
            "Rayyan": colors[4],
        }
        x_ticks = list(self.results["error_rate"].unique())
        self._plot_combined_avg_metric(self.results, "recall", f"Recall (GARF)", "Error Rate", "Recall", x_ticks, palette, hue="dataset")
        self._save_plot(os.path.join(self.base_dir, f"recall_combined.png"))
        self._plot_combined_avg_metric(self.results, "precision", f"Precision (GARF)", "Error Rate", "Precision", x_ticks, palette, hue="dataset")
        self._save_plot(os.path.join(self.base_dir, f"precision_combined.png"))
        self._plot_combined_avg_metric(self.results, "f1", f"F1 (GARF)", "Error Rate", "F1", x_ticks, palette, hue="dataset")
        self._save_plot(os.path.join(self.base_dir, f"f1_combined.png"))
        self._plot_combined_avg_runtime(self.results, "runtime", f"Runtime (GARF)", "Error Rate", "Runtime (in Seconds)", x_ticks, palette, hue="dataset")
        self._save_plot(os.path.join(self.base_dir, f"runtime_combined.png"))   

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
                    experiment = GARFExperiment(
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