import os
import pickle
import pandas as pd
from typing import List
from .preparation import Preparation
from create_db import datasets


class UniDetectPreparation(Preparation):
    def __init__(self, method: str, datasets: List[str], limit: int = -1):
        self.method = method
        self.base_dir = os.path.join(os.getcwd(), "methods", self.method)
        self.datasets = datasets
        self.limit = limit
    
    def run(self):
        base_dir = os.path.join(os.getcwd(), "methods", self.method)
        artificial_lake_dir = os.path.join(base_dir, "datasets", "artificial_lake")
        os.makedirs(artificial_lake_dir, exist_ok=True)
        
        clean_urls = []
        for dataset in self.datasets:
            clean_url = datasets[dataset][0]
            clean_df = pd.read_csv(clean_url)
            if self.limit > 0:
                clean_df = clean_df.loc[:self.limit - 1]
            dataset_path = os.path.join(artificial_lake_dir, dataset)
            os.makedirs(dataset_path, exist_ok=True)
            path = os.path.join(dataset_path, "clean.csv")
            clean_df.to_csv(path, index=False)
            clean_urls.append(path)
        
        pkl_path = os.path.join(base_dir, "pkl", "test.pkl")
        with open(pkl_path, "wb") as file:
            pickle.dump(clean_urls, file, protocol=pickle.HIGHEST_PROTOCOL)
