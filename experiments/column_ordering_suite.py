import os
import sqlite3
import numpy as np
import pandas as pd
import seaborn as sns
import random
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from .suite import Suite
from create_db import datasets
from insert_error import insert_error_unidetect
from methods.lake_cleaner.utils import combine, calculate_rule_ratio
from .column_ordering_experiment import ColumnOrderingExperiment as GARFExperiment


class ColumnOrderingSuite(Suite):
    def __init__(self, dataset_name: str, error_generator: int = 0):
        super().__init__("ColumnOrdering", "error_correction", dataset_name, error_generator)
        self._create_directory()
        
    def plot(self):
        df_mean = self.results.groupby("ordering")["f1"].mean().reset_index()
        df_mean = df_mean.rename(columns={"f1": "mean_f1"})
        df_std = self.results.groupby("ordering")["f1"].std().reset_index()
        df_std = df_std.rename(columns={"f1": "std_f1"})
        df = pd.merge(df_mean, df_std, on="ordering")
        
        palette = sns.color_palette("hls", len(df))
        
        order_labels = df["ordering"].tolist()
        new_labels = []
        for label in order_labels:
            label_parts = label.split(" ")
            label_parts = [part[:4] if len(part) > 4 else part for part in label_parts]
            new_labels.append(" \& ".join(label_parts))
        order_labels = new_labels
        
        df["ordering"] = df.index + 1
        df_sorted = df.sort_values("mean_f1")        

        # barplot
        plt.figure(figsize=(5,6))
        sns.barplot(x="mean_f1", y="ordering", data=df_sorted, orient="h", errorbar=None, order=df_sorted["ordering"], palette=palette)
        for i, row in enumerate(df_sorted.itertuples()):
            plt.errorbar(x=row.mean_f1, y=i, xerr=row.std_f1, fmt='none', color='black', capsize=3)

        plt.title("Mean F1 for different column orderings")
        plt.xlabel("F1")
        plt.xlim(0, 1)
        plt.ylabel("Column Ordering")
        patches = [mpatches.Patch(color=palette[i], label=order_labels[i]) for i in range(len(order_labels))]
        plt.legend(handles=patches, title='Ordering', bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=1, fancybox=True, shadow=True)
        #plt.tight_layout()
        self._save_plot(os.path.join(self.base_dir, f"column_ordering_bar_{self.dataset}.png"))
        
        # lineplot
        plt.figure(figsize=(8,5))
        sns.lineplot(x=np.arange(len(df_sorted)), y="mean_f1", data=df_sorted, marker='o', errorbar=None)
        plt.title("Mean F1 for different column orderings")
        plt.xlabel("Column Ordering")
        plt.xticks(list(df.index))
        plt.ylabel("F1")
        plt.ylim(0, 1)
        self._save_plot(os.path.join(self.base_dir, f"column_ordering_line_{self.dataset}.png"))
        
        # rule similarity heatmap
        rule_folder = [folder for folder in os.listdir(self.base_dir) if os.path.isdir(os.path.join(self.base_dir, folder))]
        rule_to_index = {rule_folder[i]: i for i in range(len(rule_folder))}
        combinations_to_check = list(combinations(rule_folder, 2))
        data = pd.DataFrame(index=list(range(len(rule_folder))), columns=list(range(len(rule_folder))), dtype=float).fillna(0)
        np.fill_diagonal(data.values, 1)
        
        for folder in rule_folder:        
            forward_rules_first = eval(open(os.path.join(self.base_dir, folder, "rules_forward.txt"), "r").read())            
            backward_rules_first = eval(open(os.path.join(self.base_dir, folder, "rules_backward.txt"), "r").read())
            ruleset = combine(forward_rules_first, backward_rules_first)
            ratio = calculate_rule_ratio(ruleset, ruleset)
            assert ratio == 1, "Ratio should be 1"  
        
        for combination in combinations_to_check:
            first_ruleset_name = combination[0]
            second_ruleset_name = combination[1]
                        
            forward_rules_first = eval(open(os.path.join(self.base_dir, first_ruleset_name, "rules_forward.txt"), "r").read())            
            backward_rules_first = eval(open(os.path.join(self.base_dir, first_ruleset_name, "rules_backward.txt"), "r").read())
            first_ruleset = combine(forward_rules_first, backward_rules_first)
                        
            forward_rules_second = eval(open(os.path.join(self.base_dir, second_ruleset_name, "rules_forward.txt"), "r").read())
            backward_rules_second = eval(open(os.path.join(self.base_dir, second_ruleset_name, "rules_backward.txt"), "r").read())
            second_ruleset = combine(forward_rules_second, backward_rules_second)
            
            ratio = calculate_rule_ratio(first_ruleset, second_ruleset)
            data.loc[rule_to_index[first_ruleset_name], rule_to_index[second_ruleset_name]] = ratio
            data.loc[rule_to_index[second_ruleset_name], rule_to_index[first_ruleset_name]] = ratio
        
        plt.figure(figsize=(12,8))
        heatmap = sns.heatmap(data, annot=True)
        plt.tight_layout()
        plt.title("Similarities between found rules")
        text = "\n".join([f"{i}: {order_labels[i]}" for i in range(len(order_labels))])
        plt.text(0.5, -0.35, text, horizontalalignment='center', verticalalignment='center', transform=heatmap.transAxes)
        self._save_plot(os.path.join(self.base_dir, f"column_ordering_similarities_{self.dataset}.png"))
        
        return self


    def run(self,
            max_error_rate: float,
            min_error_rate: float = 0.1, 
            error_step_size: float = 0.1, 
            runs_per_error_rate: int = 5, 
            error_intervals: int = 0, 
            **kwargs):
        dataset_url = os.path.join(os.getcwd(), datasets[self.dataset][0])
        dataset = pd.read_csv(dataset_url, dtype=str)
                    
        # remove label columns
        if "Label" in dataset.columns:
            dataset = dataset.drop(columns=["Label"])
        if "labelvalue" in dataset.columns:
            dataset = dataset.drop(columns=["labelvalue"])
        dataset["Label"] = 0
        # insert errors
        dataset_dirty, dataset_clean, errors = insert_error_unidetect(dataset, 0.1, self.base_dir)
        dataset = dataset_dirty
        dataset_label_column = dataset["Label"].copy()
        dataset = dataset.drop(columns=["Label"])
        
        columns_list = list(dataset.columns)
        orderings = []
        ordering_strings = []
        while len(orderings) < 30:
            random.shuffle(columns_list)
            if str(columns_list) not in ordering_strings:
                orderings.append(list(columns_list))
                ordering_strings.append(str(columns_list))
        
        self.results["ordering"] = []
        for index, ordering in enumerate(orderings):
            path = f"{self.dataset} ({index}-Ordering)"
            
            # create database
            dataset_dirty_ordering = dataset_dirty.copy()
            dataset_dirty_ordering = dataset_dirty_ordering[ordering]
            dataset_dirty_ordering["Label"] = dataset_label_column
            dataset_clean_ordering = dataset_clean.copy()
            dataset_clean_ordering = dataset_clean_ordering[ordering]
            dataset_clean_ordering["Label"] = None
            
            # clean database
            for column in dataset_clean_ordering.columns:
                if column == "Label":
                    continue
                dataset_dirty_ordering[column] = dataset_dirty_ordering[column].fillna("")
                dataset_clean_ordering[column] = dataset_clean_ordering[column].fillna("")
                
                dataset_dirty_ordering[column] = dataset_dirty_ordering[column].str.replace(',', ' ')
                dataset_clean_ordering[column] = dataset_clean_ordering[column].str.replace(',', ' ')

            # save database
            garf_method_dir = os.path.join(os.getcwd(), "methods", "garf_original")
            connection = sqlite3.connect(os.path.join(garf_method_dir, "database.db"))
            dataset_clean_ordering.to_sql(f"{path}", connection, if_exists="replace", index=False)
            dataset_dirty_ordering.to_sql(f"{path}_copy", connection, if_exists="replace", index=False)
            dataset_dirty_ordering.to_sql(f"{path}_err_ori", connection, if_exists="replace", index=False)
            
            # save attribute names
            if not os.path.exists(os.path.join(garf_method_dir, "data", "save")):
                os.makedirs(os.path.join(garf_method_dir, "data", "save"))
            _dict = {i: column for i, column in enumerate(dataset_clean_ordering.loc[:, ~dataset_clean_ordering.columns.isin(["Label"])].columns)} 
            with open(os.path.join(garf_method_dir, "data", "save", "att_name.txt"), "w") as f:
                f.write(str(_dict))

            # run experiments
            for i in range(runs_per_error_rate):
                experiment = GARFExperiment(
                    error_rate=0.1, 
                    method_name=f"garf_{self.dataset}_{0.1}_{index}_{i}", 
                    dataset=f"{self.dataset} ({index}-Ordering)",
                    error_generator=self.error_generator,
                    rules_path=os.path.join(self.base_dir, "_".join(ordering))
                )
                experiment.run(remove_amount_of_error_tuples=0, errors=errors)
                self.results["error_rate"].append(0.1)
                self.results["precision"].append(experiment.precision)
                self.results["recall"].append(experiment.recall)
                self.results["f1"].append(experiment.f1)
                self.results["runtime"].append(experiment.runtime)
                self.results["dataset"].append(self.dataset)
                self.results["removed_error_tuples"].append(0)
                self.results["ordering"].append(" ".join(ordering))
                pd.DataFrame(self.results).to_csv(os.path.join(self.base_dir, "column_ordering_temp_results.csv"), index=False)
            
        self.results = pd.DataFrame(self.results)
        return self