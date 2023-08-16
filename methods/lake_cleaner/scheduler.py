import os
import sys
import tqdm
import shutil
import sqlite3
import logging
import numpy as np
import pandas as pd
import seaborn as sns
# from textblob import TextBlob
from typing import List, Tuple
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
# import scienceplots
from methods.lake_cleaner.utils import combine, get_rule_vocab, get_dataframe_vocab, get_assignments, get_applicable_rules, neg_log_avg_score, vocab_overlap, jaccard_similarity

DEBUG = False

class Scheduler:
    GARF_EXECUTION = 0
    RULE_APPLICATION = 1
    IGNORED = 2
    
    def __init__(self, 
                 lake_base_path: str, 
                 max_dataset_size: int = 25000, 
                 min_dataset_size: int = 500, 
                 max_uniqueness_score: float = 0.3, 
                 min_lhs_coverage: float = 0.5, 
                 min_rhs_coverage: float = 0.01,
                 start_rule_application: int = -1,  # -1 to disable
        ):
        self.lake_base_path = lake_base_path
        self.datasets = 0
        self.garf_executions = 0
        self.rule_applications = 0
        self.dataset_to_garf_executions = dict()
        self.rule_subdirs = []
        self.ignored_datasets = []
        self.max_dataset_size = max_dataset_size
        self.min_dataset_size = min_dataset_size
        self.max_uniqueness_score = max_uniqueness_score
        self.min_lhs_coverage = min_lhs_coverage
        self.min_rhs_coverage = min_rhs_coverage
        self.start_rule_application = start_rule_application
        self.min_rule_coverage = 0.001
            
    def get_dataset_paths(self) -> List[str]:
        return list(self.dataset_to_garf_executions.keys())
    
    def get_dataset_to_garf_executions(self) -> dict:
        return self.dataset_to_garf_execution
    
    def execution_stats(self):
        return self.garf_executions, self.datasets
    
    def get_results(self, dataset_to_time_consumption: dict, dataset_checkpoint_time: dict):
        runtime = pd.DataFrame(list(dataset_to_time_consumption.items()), columns=["dataset", "runtime"])
        execution = pd.DataFrame(list(self.dataset_to_garf_executions.items()), columns=["dataset", "executed"])
        checkpoint = pd.DataFrame(list(dataset_checkpoint_time.items()), columns=["dataset", "checkpoint"])
        
        results = pd.merge(execution, runtime, on="dataset", how="left")
        results = pd.merge(results, checkpoint, on="dataset", how="left")
        results["runtime"] = results["runtime"].fillna(0)
        return results        
    
    def save_temporary_result(self, dataset_to_time_consumption: dict, dataset_checkpoint_time: dict):
        temp_results = self.get_results(dataset_to_time_consumption, dataset_checkpoint_time)
        
        temp_results.to_csv(os.path.join(self.lake_base_path, "temp_results.csv"), index=False)
        
        return temp_results
    
    def uniqueness_score(self, data: pd.DataFrame) -> float:
        uniqueness_df = pd.DataFrame(index=data.columns, columns=["uniqueness"])
        for column in data.columns:
            uniqueness_df.loc[column] = data[column].nunique() / data.shape[0]
        return uniqueness_df["uniqueness"].mean()
    
    def vocab_stats(self, data: pd.DataFrame) -> Tuple[int, float]:
        vocab = set()
        for column in data.columns:
            vocab.update(data[column].unique())
            
        avg_token_length = sum([len(token) for token in vocab if isinstance(token, str)]) / len(vocab)
        return len(vocab), avg_token_length
    
    def word_lengths(self, data: pd.DataFrame) -> Tuple[int, int, int, int]:
        bins = {'0-2': 0, '3-10': 0, '11-100': 0, '>100': 0}
        for column in data.columns:
            for word in data[column]:
                if isinstance(word, str):
                    length = len(word)
                    if length <= 2:
                        bins['0-2'] += 1
                    elif length <= 10:
                        bins['3-10'] += 1
                    elif length <= 100:
                        bins['11-100'] += 1
                    else:
                        bins['>100'] += 1
        return tuple(bins.values())
    
    def count_dtypes(self, data: pd.DataFrame) -> Tuple[int, int, int]:
        str_columns = int_columns = float_columns = 0

        for column in data.columns:
            inferred_dtype = pd.api.types.infer_dtype(data[column])

            if inferred_dtype == 'string':
                str_columns += 1
            elif inferred_dtype == 'integer':
                int_columns += 1
            elif inferred_dtype == 'floating':
                float_columns += 1

        return str_columns, int_columns, float_columns
    
    def count_nulls(self, data: pd.DataFrame) -> int:
        nulls = data.isnull().sum().sum()
        return nulls
    
    # def sentiment_score(self, data: pd.DataFrame) -> float:
    #    sentiment_scores = data.apply(lambda row: TextBlob(" ".join(str(row))).sentiment.polarity)
    #    return sentiment_scores.mean()
    
    def _check_base_dir(self):
        if not os.path.exists(self.lake_base_path):
            raise ValueError("Data lake base path does not exist")
    
    def mine_features(self) -> pd.DataFrame:
        self._check_base_dir()
        
        file_paths = os.listdir(self.lake_base_path)
        file_paths = list(sorted(file_paths))        
        file_paths = [file_path for file_path in file_paths if os.path.isdir(os.path.join(self.lake_base_path, file_path))]
        
        dataframe_features = {
            "dataset": [],
            "num_tuples": [],
            "num_attributes": [],
            "uniqueness_score": [],
            "vocab_size": [],
            "avg_token_size": [],
            "type_token_ratio": [],
            "word_length<=2": [],
            "word_length<=10": [],
            "word_length<=100": [],
            "word_length>100": [],
            "str_column_ratio": [],
            "int_column_ratio": [],
            "float_column_ratio": [],
            "null_ratio": [],
        }
        progress_bar = tqdm.tqdm(file_paths, desc="Dataset feature mining")
        for subdir in progress_bar:
            self.datasets += 1
            file_path =  os.path.join(self.lake_base_path, subdir, "dirty.csv")
            df = pd.read_csv(file_path)
            
            # heuristic 1: skip datasets that are too small or too large
            if df.shape[0] > self.max_dataset_size or df.shape[0] < self.min_dataset_size or df.shape[1] > 15:
                progress_bar.write(f"Skipping dataset {subdir} due to size ({df.shape[0]}))")
                self.dataset_to_garf_executions[subdir] = self.IGNORED
                self.ignored_datasets.append(subdir)     
                continue
            
            # heuristic 2: skip datasets that are too unique
            uniqueness_score = self.uniqueness_score(df)
            if uniqueness_score > self.max_uniqueness_score:
                progress_bar.write(f"Skipping dataset {subdir} due to uniqueness ({uniqueness_score}))")
                self.dataset_to_garf_executions[subdir] = self.IGNORED
                self.ignored_datasets.append(subdir)     
                continue
            
            vocab_size, avg_token_size = self.vocab_stats(df)
            w2, w10, w100, wg100 = self.word_lengths(df)
            str_dtype_count, int_dtype_count, float_dtype_count = self.count_dtypes(df)
            null_count = self.count_nulls(df)
            
            # heuristic 3: skip only numeric datasets
            if str_dtype_count == 0:
                progress_bar.write(f"Skipping dataset {subdir} due to only numeric columns")
                self.dataset_to_garf_executions[subdir] = self.IGNORED
                self.ignored_datasets.append(subdir)     
                continue
            
            cell_count = df.shape[0] * df.shape[1]
            dataframe_features["dataset"].append(subdir)
            dataframe_features["num_tuples"].append(df.shape[0])
            dataframe_features["num_attributes"].append(df.shape[1])
            dataframe_features["uniqueness_score"].append(uniqueness_score)
            dataframe_features["vocab_size"].append(vocab_size)
            dataframe_features["avg_token_size"].append(avg_token_size)
            dataframe_features["type_token_ratio"].append(vocab_size / cell_count)
            dataframe_features["word_length<=2"].append(w2 / cell_count)
            dataframe_features["word_length<=10"].append(w10 / cell_count)
            dataframe_features["word_length<=100"].append(w100 / cell_count)
            dataframe_features["word_length>100"].append(wg100 / cell_count)
            dataframe_features["str_column_ratio"].append(str_dtype_count / df.shape[1])
            dataframe_features["int_column_ratio"].append(int_dtype_count / df.shape[1])
            dataframe_features["float_column_ratio"].append(float_dtype_count / df.shape[1])
            dataframe_features["null_ratio"].append(null_count / cell_count)
        
        dataframe_stats = pd.DataFrame(dataframe_features)
        dataframe_stats = dataframe_stats.set_index("dataset")
        plt.figure(figsize=(12,10))
        sns.heatmap(dataframe_stats.corr(), annot=True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.lake_base_path, "dataframe_stats_correlation.png"))
        plt.close()
        return dataframe_stats
    
    def plot_dataset_ordering(self, dataframe_stats: pd.DataFrame, scaled_df: pd.DataFrame, n_cluster: int = 4, centroids: np.ndarray = None):
        # plt.style.use("science")
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(scaled_df.to_numpy())
        centroids = pca.transform(centroids)
        palette = sns.color_palette("tab10")
        color_dict = {i: palette[i] for i in range(n_cluster)}
        
        plt.figure(figsize=(10,6))
        ax1 = plt.gca()
        plot = sns.scatterplot(
            x=principal_components[:,0], 
            y=principal_components[:,1], 
            hue=dataframe_stats["Cluster"], 
            palette=color_dict, 
            ax=ax1
        )
        for i, centroid in enumerate(centroids):
            sns.scatterplot(
                x=[centroid[0]], 
                y=[centroid[1]], 
                color=color_dict[i], 
                marker="v", 
                s=100, 
                ax=ax1,
                label=f"Centroid {i}"
            )
        plot.axhline(0, color="black", alpha=0.25, linestyle="--") 
        plot.axvline(0, color="black", alpha=0.25, linestyle="--") 
        x_lim = plot.get_xlim()
        y_lim = plot.get_ylim()
        plt.savefig(os.path.join(self.lake_base_path, "dataframe_stats_clustered.png"))
        plt.close()

        dataframe_demo_datasets = dataframe_stats.reset_index().copy()
        dataframe_demo_datasets = dataframe_demo_datasets[dataframe_demo_datasets["dataset"].isin(["beers", "flights", "hospital", "food", "rayyan"])]
        demo_dataset_indicies = dataframe_demo_datasets.index.tolist()
        highlighted_datasets = dataframe_demo_datasets["dataset"].tolist()
    
        highlighted_points = []
        highlighted_clusters = []

        for i, component in enumerate(principal_components):
            if i in demo_dataset_indicies:
                highlighted_points.append(component)
                highlighted_clusters.append(dataframe_stats.iloc[i]["Cluster"])
    
        highlighted_points = np.asarray(highlighted_points)
        dataframe_demo_datasets.to_csv(os.path.join(self.lake_base_path, "dataframe_demo_datasets.csv"), index=False)
        
        plt.figure(figsize=(10,6))
        ax2 = plt.gca()
        sns.scatterplot(
            x=principal_components[:,0], 
            y=principal_components[:,1], 
            hue=dataframe_stats["Cluster"], 
            palette=color_dict, 
            alpha=0.25, 
            ax=ax2
        )
        for i, centroid in enumerate(centroids):
            sns.scatterplot(
                x=[centroid[0]], 
                y=[centroid[1]], 
                color=color_dict[i],
                marker="v", 
                s=100, 
                ax=ax2,
                label=f"Centroid {i}"
            )
        plot = sns.scatterplot(
            x=highlighted_points[:,0], 
            y=highlighted_points[:,1], 
            hue=highlighted_clusters, 
            palette=color_dict,
            edgecolor="black", 
            ax=ax2
        )
        for i, dataset_name in enumerate(highlighted_datasets):
            plt.text(highlighted_points[i, 0], highlighted_points[i, 1], dataset_name.capitalize(), horizontalalignment="right")

        plot.axhline(0, color="black", alpha=0.25, linestyle="--") 
        plot.axvline(0, color="black", alpha=0.25, linestyle="--")
        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in color_dict.values()]
        ax2.legend(handles=legend_elements, labels=color_dict.keys(), title='Cluster')

        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.savefig(os.path.join(self.lake_base_path, "dataframe_demo_datasets_stats_clustered.png"))
        plt.close()

    def order_datasets(self, dataframe_stats: pd.DataFrame, n_cluster: int = 2) -> List[str]:
        scaler = MinMaxScaler()
        scaled_df = pd.DataFrame(scaler.fit_transform(dataframe_stats), columns=dataframe_stats.columns)     
        kmeans = KMeans(n_clusters=n_cluster, random_state=42)
        dataframe_stats["Cluster"] = kmeans.fit_predict(scaled_df)
        # plot clustering
        self.plot_dataset_ordering(dataframe_stats, scaled_df, n_cluster, kmeans.cluster_centers_)
        dataframe_stats.groupby("Cluster").mean().to_csv(os.path.join(self.lake_base_path, "dataframe_stats_clustered.csv"))
        dataframe_stats = dataframe_stats.sort_values(by=["Cluster", "num_tuples"], ascending=[True, False])
        dataframe_stats["group"] = dataframe_stats.groupby("Cluster").cumcount()
        dataframe_stats = dataframe_stats.sort_values(by=["group", "Cluster"])
        dataframe_stats = dataframe_stats.drop(columns=["group"])
        dataframe_stats = dataframe_stats.reset_index()
        dataframe_stats.to_csv(os.path.join(self.lake_base_path, "dataframe_stats.csv"), index=False)
        
        dataframe_stats_food_hospital_flights = dataframe_stats[dataframe_stats["dataset"].isin(["food", "hospital", "flights"])]
        most_promising_cluster = dataframe_stats_food_hospital_flights["Cluster"].value_counts().idxmax()
        if not DEBUG:
            ignored_datasets = dataframe_stats[dataframe_stats["Cluster"] != most_promising_cluster]["dataset"].tolist()
            self.ignored_datasets += ignored_datasets
            dataframe_stats = dataframe_stats[dataframe_stats["Cluster"] == most_promising_cluster]
        
        return dataframe_stats["dataset"].tolist()

    def _check_rule_applications(self, data: pd.DataFrame, subdir: str) -> Tuple[bool, dict, dict, dict]:
        with open(os.path.join(self.lake_base_path, subdir, "rules_forward.txt"), "r") as f:
            rules_forward = eval(f.read())
        
        with open(os.path.join(self.lake_base_path, subdir, "rules_backward.txt"), "r") as f:
            rules_backward = eval(f.read())

        rules = combine(rules_forward, rules_backward)
        if len(rules) == 0:
            return False, dict(), dict(), dict()
        
        lhs_vocab, rhs_vocab = get_rule_vocab(rules)        
        df_vocab = get_dataframe_vocab(data)
        
        lhs_assignment, _ = get_assignments(lhs_vocab, df_vocab, self.min_lhs_coverage, scoring_fn=neg_log_avg_score, similarity_fn=vocab_overlap)
        # reverse assignment
        lhs_assignment = {v: k for k, v in lhs_assignment.items()}
        rhs_assignment, _ = get_assignments(rhs_vocab, df_vocab, self.min_rhs_coverage, scoring_fn=neg_log_avg_score, similarity_fn=jaccard_similarity)
        # reverse assignment
        rhs_assignment = {v: k for k, v in rhs_assignment.items()}
        
        applicable_rules = get_applicable_rules(data, rules, lhs_assignment, rhs_assignment)
        coverage = len(applicable_rules) / len(rules)

        return coverage > self.min_rule_coverage, applicable_rules, lhs_assignment, rhs_assignment
    
    def receive_rules(self, directory: str):
        self.rule_subdirs.append(directory)

    def apply_rules(self, data: pd.DataFrame, applicable_rules: dict, subdir: str) -> pd.DataFrame:
        data["rules_applied"] = 0
        
        for _, rules in applicable_rules.items():
            
            reason = rules["reason"]
            result = rules["result"]
            mask = [True] * data.shape[0]
            
            for key, value in reason.items():
                mask &= data[key] == value
            
            for key, value in result.items():
                data.loc[mask, key] = value
                data.loc[mask, "rules_applied"] = 1 + data.loc[mask, "rules_applied"]

        print(data["rules_applied"].value_counts())
        db_path = os.path.join(self.lake_base_path, subdir, "database.db")
        connection = sqlite3.connect(db_path)
        data.to_sql(f"{subdir}_copy", connection, if_exists="replace", index=False)
        return data
    
    def clean_ignored_datasets(self):
        for directory in self.ignored_datasets:
            file_path =  os.path.join(self.lake_base_path, directory, "dirty.csv")
            data = pd.read_csv(file_path, dtype=str)
            _, applicable_rules = self.get_applicable_rules(data)
            if len(applicable_rules) == 0:
                continue
            
            data = self.apply_rules(data, applicable_rules, directory)
            db_path = os.path.join(self.lake_base_path, directory, "database.db")
            connection = sqlite3.connect(db_path)
            data.to_sql(f"{directory}_copy", connection, if_exists="replace", index=False)
            shutil.copy2(db_path, os.path.join(self.lake_base_path, directory, "cleaned.db"))
            self.dataset_to_garf_executions[directory] = self.RULE_APPLICATION
            self.rule_applications += 1
               
    def get_applicable_rules(self, data: pd.DataFrame) -> Tuple[bool, dict]:
        rules_applicable = False
        applicable_rules = dict()
        for rule_subdir in self.rule_subdirs:
            current_rules_applicable, new_applicable_rules, lhs_assignment, rhs_assignment = self._check_rule_applications(data, rule_subdir)
            
            for key, value in new_applicable_rules.items():
                if key in applicable_rules and applicable_rules[key]["confidence"] > value["confidence"]:
                        continue

                # assign rules to new dataset
                new_values = {
                    "reason": dict(),
                    "result": dict(),
                    "confidence": -np.inf
                }
                for column_name, attribute_value in value["reason"].items():
                    new_values["reason"][lhs_assignment[column_name]] = attribute_value
                for column_name, attribute_value in value["result"].items():
                    new_values["result"][rhs_assignment[column_name]] = attribute_value
                
                new_values["confidence"] = value["confidence"]
                applicable_rules[key] = new_values
            
            rules_applicable = rules_applicable or current_rules_applicable

        return rules_applicable, applicable_rules
            
    def __iter__(self):
        self._check_base_dir()
        data = self.mine_features()
        file_paths = self.order_datasets(data)
        logging.info(file_paths)
        logging.info(f"Analysing {len(file_paths)} of {len(list(self.dataset_to_garf_executions.keys()))} datasets")

        for subdir in file_paths:
            self.datasets += 1
            file_path =  os.path.join(self.lake_base_path, subdir, "dirty.csv")
            df = pd.read_csv(file_path, dtype=str)
            
            if self.garf_executions > self.start_rule_application:
                rules_applicable, applicable_rules = self.get_applicable_rules(df)
                
                # apply any applicable rules
                if len(applicable_rules) > 0:
                    self.apply_rules(df, applicable_rules, subdir)
                
                # if applicable rules exceed threshold, skip garf execution
                if rules_applicable:
                    self.dataset_to_garf_executions[subdir] = self.RULE_APPLICATION
                    self.rule_applications += 1
                    logging.info(f"Skipping dataset {subdir} due to applicable rules")
                    continue
            
            self.garf_executions += 1
            if DEBUG and os.path.exists(os.path.join(self.lake_base_path, subdir, "rules_backward.txt")):
                self.receive_rules(subdir)
                continue
            
            self.dataset_to_garf_executions[subdir] = self.GARF_EXECUTION        
            yield os.path.join(self.lake_base_path, subdir)