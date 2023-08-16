from __future__ import annotations
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import os
from typing import List, Union, Dict
import time
import json
import logging
from scipy.stats import f_oneway
from create_db import dataset_names
import sys


class Suite(ABC):
    def __init__(self, name: str, task: str, dataset: str, error_generator: int = 0):
        self.name = name
        self.task = task
        self.results = {
            "error_rate": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "runtime": [],
            "dataset": [],
            "removed_error_tuples": [],
        }
        self.dataset = dataset
        self.error_generator = error_generator
        self._directory_created = False
        self.suppress_plotting = False
        if self.dataset in dataset_names:
            name = dataset_names[self.dataset]
        else:
            name = self.dataset
        logging.info(f"Created suite {self.name} for task {self.task} on dataset {name}")

    def _create_directory(self):
        if not self._directory_created:
            if not os.path.exists("results"):
                os.mkdir("results")
            if not os.path.exists(os.path.join("results", self.task)):
                os.mkdir(os.path.join("results", self.task))
            self.method_dir = os.path.join("results",self.task, self.name)
            if not os.path.exists(self.method_dir):
                os.mkdir(self.method_dir)
            self.base_dir = os.path.join(self.method_dir, str(round(time.time())))
            if not os.path.exists(self.base_dir):
                os.mkdir(self.base_dir)

        self._directory_created = True

    def _check_results(self):
        if isinstance(self.results, dict):
            raise Exception("Results not yet computed. Please run the suite first.")
        
    def set_result_path(self, input_dir: Union[str, int]) -> str:
         self._create_directory()
         os.rmdir(self.base_dir)
         self.base_dir = os.path.join(self.method_dir, f"{input_dir}")
         return self.base_dir
        
    def merge(self, suite: Suite):
        self._check_results()
        suite._check_results()
        self.results = pd.concat([self.results, suite.results])
        return self
    
    def generate_error_rates(self, max_error_rate: float, min_error_rate: int = 0.01, interval_size: float = 0.05):
        if min_error_rate == 0.01:
            error_rates = [round(x, 2) for x in np.arange(0, max_error_rate + interval_size, interval_size)]
            error_rates[0] = 0.01
        else:
            error_rates = [round(x, 2) for x in np.arange(min_error_rate, max_error_rate + interval_size, interval_size)]
            
        return error_rates

    def debug_plot(self, time: Union[str, int]):
        self._create_directory()
        os.rmdir(self.base_dir)
        self.base_dir = os.path.join(self.method_dir, f"{time}")
        self.results = pd.read_csv(os.path.join(self.base_dir, "results.csv"), index_col=0)
        self.plot()
        
    def combined_debug_plot(self, results: pd.DataFrame):
        self._create_directory()
        self.results = results
        self.combined_plot()
    
    @staticmethod
    def _format_aggregated_results(df: pd.DataFrame, metric: str):
        df = df[[metric]]
        if metric == "runtime":
            new_columns = df.columns.map(lambda x: "Worst" if x[1] == "max" else "Best" if x[1] == "min" else "Mean")
        else:
            new_columns = df.columns.map(lambda x: "Best" if x[1] == "max" else "Worst" if x[1] == "min" else "Mean")
        df.columns = new_columns
        return df
    
    def combined_plot(self):
        if self.suppress_plotting:
            return self
        
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
        results_grouped = self.results.groupby(["error_rate", "method"])
        mean_results = results_grouped.agg({"precision": "mean", "recall": "mean", "f1": "mean", "runtime": "mean"})
        mean_results = mean_results.reset_index()
        mean_results = mean_results.rename(columns={"method": "Model"})
        
        mean_results["Model"] = mean_results["Model"].replace({
            "distilled-garf-Flights-g1": "DistilledGARF", 
            "distilled-garf-Beers-g1": "DistilledGARF", 
            "garf-original-baseline-g1": "GARF (Original)", 
            "rnn-garf-baseline-all-g1": "RNN-GARF", 
            "rnn-garf-final-model-g1": "RNN-GARF (tuned)",
            "rnn-garf-baseline-hospital-g1-64lstm": "RNN-GARF",
            "bilstm-garf-baseline-g1-food-flights": "Bidirectional LSTM-GARF",
            "rnn-garf-final-model-g1-rayyan": "RNN-GARF (tuned)",
            "rnn-garf-baseline-rayyan-g1-64lstm": "RNN-GARF",
            "bilstm-garf-baseline-g1-rayyan": "Bidirectional LSTM-GARF",
            "distilled-garf-Rayyan-g1": "DistilledGARF",
            "garf-original-baseline-rayyan-g1": "GARF (Original)",
            "bilstm-garf-baseline-g1-hospital": "Bidirectional LSTM-GARF",
            "bilstm-garf-baseline-g1-food": "Bidirectional LSTM-GARF",
            "uni-detect-baseline": "UniDetect (pretrained 1m)",
        })
        colors = sns.color_palette("tab10")
        palette = {
            "DistilledGARF": colors[0], 
            "GARF (Original)": colors[1],
            "RNN-GARF": colors[2],
            "RNN-GARF (tuned)": colors[3],
            "Bidirectional LSTM-GARF": colors[4],
            "UniDetect (pretrained 1m)": colors[0],
        }
        x_ticks = list(mean_results["error_rate"].unique())
        self._plot_combined_avg_metric(mean_results, "recall", f"Recall on {dataset}", "Error Rate", "Recall", x_ticks, palette)
        self._save_plot(os.path.join(self.base_dir, f"recall_combined_{dataset}.png"))
        self._plot_combined_avg_metric(mean_results, "precision", f"Precision on {dataset}", "Error Rate", "Precision", x_ticks, palette)
        self._save_plot(os.path.join(self.base_dir, f"precision_combined_{dataset}.png"))
        self._plot_combined_avg_metric(mean_results, "f1", f"F1 on {dataset}", "Error Rate", "F1", x_ticks, palette)
        self._save_plot(os.path.join(self.base_dir, f"f1_combined_{dataset}.png"))
        self._plot_combined_avg_runtime(mean_results, "runtime", f"Runtime on {dataset}", "Error Rate", "Runtime (in Seconds)", x_ticks, palette)
        self._save_plot(os.path.join(self.base_dir, f"runtime_combined_{dataset}.png"))        
        
    @staticmethod
    def _plot_combined_avg_metric(dataframe: pd.DataFrame, metric:str, title: str, x_label: str, y_label: str, x_ticks: List[Union[float, int]] = None, palette = None, hue: str = "Model"):
        sns.lineplot(data=dataframe, x="error_rate", y=metric, hue=hue, marker="o", palette=palette)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xticks(x_ticks if x_ticks is not None else dataframe.index)
        plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        plt.ylim(0, 1)
        plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)
        
    @staticmethod
    def _plot_combined_avg_runtime(dataframe: pd.DataFrame, metric:str, title: str, x_label: str, y_label: str, x_ticks: List[Union[float, int]] = None, palette= None, hue: str = "Model"):
        sns.lineplot(data=dataframe, x="error_rate", y=metric, hue=hue, marker="o", palette=palette)
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.yscale("log")
        plt.xticks(x_ticks if x_ticks is not None else dataframe.index)
        plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=2)

    @staticmethod
    def _plot_avg_metric(dataframe: pd.DataFrame, title: str, x_label: str, y_label: str, x_ticks: List[Union[float, int]] = None):
        dataframe.plot.line(
            title=title,
            xlabel=x_label,
            ylabel=y_label,
            ylim=(0, 1), 
            yticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 
            xticks=x_ticks if x_ticks is not None else dataframe.index,
            marker="o",
        )

        
    def plot(self):
        if self.suppress_plotting:
            return self
        
        self._check_results()
        self._create_directory()
        sns.color_palette("tab10")
        
        datasets = self.results["dataset"].unique()
        all_results = self.results.copy()
        for dataset in datasets:
            # filter results for each dataset
            self.results = self.results[self.results["dataset"] == dataset]
            # plot results for each dataset
            self._plot_dataset_results(dataset)
            # reset results dataframe
            self.results = all_results.copy()
        
        return self
    
    def _plot_dataset_results(self, dataset: str):
        results_grouped = self.results.groupby("error_rate")
        # error rate plots
        for error_rate, indicies in results_grouped.groups.items():
            error_rate_df = self.results.iloc[indicies]
            self._error_rate_plots(error_rate_df, error_rate, dataset)
        # aggregated plots
        self._aggregated_plots(dataset)
        # anova plots
        if len(self.results["removed_error_tuples"].unique()) > 1 and self.results.groupby(["error_rate", "removed_error_tuples"]).size().iloc[0] > 1:
            data = self._calc_anova_data()
            self._anova_plots(data, dataset)
    
    def _error_rate_plots(self, error_rate_df: pd.DataFrame, error_rate: float, dataset: str):
        error_rate_df = error_rate_df.reset_index()
        error_rate_df = error_rate_df.rename(columns={"removed_error_tuples": "Defective Tuples not in Training Set"})
        error_rate_df.index = error_rate_df.index + 1
        title_end = f"for Error Rate {error_rate} ({dataset_names[dataset]})"                
        
        self._plot_metric(
            error_rate_df[["recall", "Defective Tuples not in Training Set"]], 
            "recall",
            f"Recall {title_end}", 
            "Run Number", 
            "Recall")
        self._save_plot(os.path.join(self.base_dir, f"recall_{error_rate}_{dataset}.png"))
        
        self._plot_metric(
            error_rate_df[["precision", "Defective Tuples not in Training Set"]],
            "precision",
            f"Precision {title_end}", 
            "Run Number",
            "Precision"
        )
        self._save_plot(os.path.join(self.base_dir, f"precision_{error_rate}_{dataset}.png"))
        
        self._plot_metric(
            error_rate_df[["f1", "Defective Tuples not in Training Set"]],
            "f1", 
            f"F1 {title_end}",
            "Run Number", 
            "F1"
        )
        self._save_plot(os.path.join(self.base_dir, f"f1_{error_rate}_{dataset}.png"))
        
        self._plot_runtime(
            error_rate_df[["runtime", "Defective Tuples not in Training Set"]],
            f"Runtime {title_end}", 
            os.path.join(self.base_dir, f"runtime_{error_rate}.png"), 
            "Run Number", 
            "Runtime (in Seconds)"
        )
    
    def _aggregated_plots(self, dataset: str):
        results_grouped = self.results.groupby("error_rate")
        mean_results = results_grouped.agg({"precision": ["mean", "max", "min"], "recall": ["mean", "max", "min"], "f1": ["mean", "max", "min"], "runtime": ["mean", "min", "max"]})
        x_ticks = list(mean_results.index)
        self._plot_avg_metric(self._format_aggregated_results(mean_results, "recall"), f"Recall for {self.name} ({dataset})", "Error Rate", "Recall", x_ticks)
        self._save_plot(os.path.join(self.base_dir, f"recall_{dataset}.png"))
        self._plot_avg_metric(self._format_aggregated_results(mean_results, "precision"), f"Precision for {self.name} ({dataset})", "Error Rate", "Precision", x_ticks)
        self._save_plot(os.path.join(self.base_dir, f"precision_{dataset}.png"))
        self._plot_avg_metric(self._format_aggregated_results(mean_results, "f1"), f"F1 for {self.name} ({dataset})", "Error Rate", "F1", x_ticks)
        self._save_plot(os.path.join(self.base_dir, f"f1_{dataset}.png"))
        self._plot_avg_runtime(self._format_aggregated_results(mean_results, "runtime"), f"Runtime for {self.name} ({dataset})", os.path.join(self.base_dir, f"runtime_{dataset}.png"), "Error Rate", "Runtime (in Seconds)", x_ticks)

    def _calc_anova_data(self) -> Dict[str, List[float]]:
        error_groups = self.results.groupby("error_rate")
        data = {
            "error_rate": [],
            "metric": [],
            "p_value": [],
            "f_statistic": [],
        }
        for group in error_groups.groups.keys():
            configuration_group = self.results[self.results["error_rate"] == group].groupby("removed_error_tuples")
            for metric in ["precision", "recall", "f1", "runtime"]:
                result_list = configuration_group[metric].apply(list)
                f_statistic, p_value = f_oneway(*result_list)
                data["error_rate"].append(group)
                data["metric"].append(metric)
                data["p_value"].append(p_value)
                data["f_statistic"].append(f_statistic)
        return data
    
    def _anova_plots(self, data: Dict[str, List[float]], dataset: str):
        anova_df = pd.DataFrame(data)
        anova_df.set_index("error_rate", inplace=True)
        print(anova_df.head())
        self._plot_anova(anova_df[anova_df["metric"] == "precision"], "p_value", f"ANOVA for Precision ({dataset})", "Error Rate", "p-value")
        self._save_plot(os.path.join(self.base_dir, f"anova_precision_{dataset}.png"))
        self._plot_anova(anova_df[anova_df["metric"] == "recall"], "p_value", f"ANOVA for Recall ({dataset})", "Error Rate", "p-value")
        self._save_plot(os.path.join(self.base_dir, f"anova_recall_{dataset}.png"))
        self._plot_anova(anova_df[anova_df["metric"] == "f1"], "p_value", f"ANOVA for F1 ({dataset})", "Error Rate", "p-value")
        self._save_plot(os.path.join(self.base_dir, f"anova_f1_{dataset}.png"))
        self._plot_anova(anova_df[anova_df["metric"] == "runtime"], "p_value", f"ANOVA for Runtime ({dataset})", "Error Rate", "p-value")
        self._save_plot(os.path.join(self.base_dir, f"anova_runtime_{dataset}.png"))
    
    def save(self, arguments: Dict) -> pd.DataFrame:
        self._check_results()
        self._create_directory()
        self.results.to_csv(os.path.join(self.base_dir, "results.csv"))
        with open(os.path.join(self.base_dir, "konfiguration.json"), "w") as file:
            json.dump(arguments, file)
        return self.results
    
    @staticmethod
    def _plot_anova(dataframe: pd.DataFrame, y: str, title: str, x_label: str, y_label: str, x_ticks: List[Union[float, int]] = None):        
        ax = sns.lineplot(data=dataframe, x="error_rate", y=y, marker="o")
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if x_ticks is not None:
            ax.set_xticks(x_ticks)
        else:
            ax.set_xticks(dataframe.index)   
    
    @staticmethod
    def _save_plot(path: str):
        plt.savefig(path)
        plt.clf()
        plt.close()
        
    @staticmethod
    def _plot_avg_metric(dataframe: pd.DataFrame, title: str, x_label: str, y_label: str, x_ticks: List[Union[float, int]] = None):
        dataframe.plot.line(
            title=title,
            xlabel=x_label,
            ylabel=y_label,
            ylim=(0, 1), 
            yticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], 
            xticks=x_ticks if x_ticks is not None else dataframe.index,
            marker="o",
        )
        
    @staticmethod  
    def _plot_avg_runtime(dataframe: pd.DataFrame, title: str, path: str, x_label: str, y_label: str, x_ticks: List[Union[float, int]] = None):
        dataframe.plot.line(
            title=title, 
            marker="o", 
            xlabel=x_label,
            ylabel=y_label,
            xticks=x_ticks if x_ticks is not None else dataframe.index,
        )
        plt.savefig(path)
        plt.clf()
        plt.close()
    
    @staticmethod
    def _plot_metric(dataframe: pd.DataFrame, y: str, title: str, x_label: str, y_label: str, x_ticks: List[Union[float, int]] = None):        
        dataframe = dataframe.copy()
        dataframe["runs"] = dataframe.groupby("Defective Tuples not in Training Set").cumcount() + 1
        
        ax = sns.lineplot(data=dataframe, x="runs", y=y, marker="o", hue="Defective Tuples not in Training Set", palette="tab10")
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        if x_ticks is not None:
            ax.set_xticks(x_ticks)
        else:
            ax.set_xticks(dataframe["runs"].unique().tolist())
    
    @staticmethod  
    def _plot_runtime(dataframe: pd.DataFrame, title: str, path: str, x_label: str, y_label: str, x_ticks: List[Union[float, int]] = None):
        dataframe = dataframe.copy()
        dataframe["runs"] = dataframe.groupby("Defective Tuples not in Training Set").cumcount() + 1
        
        ax = sns.lineplot(data=dataframe, x="runs", y="runtime", marker="o", hue="Defective Tuples not in Training Set", palette="tab10")
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if x_ticks is not None:
            ax.set_xticks(x_ticks)
        else:
            ax.set_xticks(dataframe["runs"].unique().tolist())
        
        plt.savefig(path)
        plt.clf()
        plt.close()
    
    @abstractmethod
    def run(self, max_error_rate: float, runs_per_error_rate: int = 5, error_intervals: int = 0, **kwargs):
        pass