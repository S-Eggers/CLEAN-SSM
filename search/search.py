import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Dict, List, Union


class Search(ABC):
    def __init__(self, params: Dict[str, List[Union[int, float]]], n: int):
        self.params = params
        self.best_score = -np.inf
        self.best_params = None
        self._current_params = None
        self._configuration_to_results = {}
        self._current_iter = 0
        
    @abstractmethod
    def __iter__(self):
        pass
            
    def receive_results(self, result: Dict[str, float]):
        # store coniguration and results for plotting
        self._configuration_to_results[self._current_iter] = {
            "params": self._current_params,
            "result": result,
        }
        # check for best score
        score = np.mean(result["f1"])
        if score > self.best_score:
            self.best_score = score
            self.best_params = self._current_params
        # increment iteration
        self._current_iter += 1
    
    def get_best_params(self):
        return self.best_params
    
    def get_best_score(self):
        return self.best_score
    
    def plot(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)

        search_results = {
            "params": [],
            "f1": [],
            "error_rate": [],
            "runtime": [],
        }
                
        for config, results in enumerate(self._configuration_to_results.values()):
            print(f"Configuration: {config}")
            print(f"Results: {results}")
            param_str = ", ".join([f"{param}: {value}" for param, value in results["params"].items()])
            for i, f1 in enumerate(results["result"]["f1"]):
                search_results["params"].append(param_str)
                search_results["f1"].append(f1)
                search_results["error_rate"].append(results["result"]["error_rate"][i])
                search_results["runtime"].append(results["result"]["runtime"][i])
            
        results = pd.DataFrame(search_results)
        print(results)
        results = results.groupby(["params", "error_rate"]).mean()
        results.to_csv(os.path.join(path, "random_search.csv"))
        
        sns.lineplot(
            x="error_rate", 
            y="f1", 
            hue="params", 
            data=results,
            marker="x",
            palette="tab10"
        )
        plt.savefig(os.path.join(path, "random_search_f1.png"))
        plt.clf()
        plt.close()
        sns.lineplot(
            x="error_rate",
            y="runtime",
            hue="params",
            data=results,
            marker="x",
            palette="tab10"
        )
        plt.savefig(os.path.join(path, "random_search_runtime.png"))
        plt.clf()
        plt.close()
        
    @staticmethod
    def debug_plot(path: str):
        results = pd.read_csv(path)
        path = os.path.dirname(path)
        results = results.rename(columns={"f1": "F1", "runtime": "Runtime (in s)", "error_rate": "Error Rate"})
        results = results.groupby(["params", "Error Rate"]).mean()
        sns.lineplot(
            x="Error Rate", 
            y="F1", 
            hue="params", 
            data=results,
            # marker="x",
            palette="tab10"
        )
        plt.savefig(os.path.join(path, "random_search_f1.png"))
        plt.clf()
        plt.close()
        sns.lineplot(
            x="Error Rate",
            y="Runtime (in s)",
            hue="params",
            data=results,
            # marker="x",
            palette="tab10"
        )
        plt.savefig(os.path.join(path, "random_search_runtime.png"))
        plt.clf()
        plt.close()
            
            