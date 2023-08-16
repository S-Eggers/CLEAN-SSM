import os
import pandas as pd
import numpy as np
from .preparation import Preparation
from multiprocessing import Process
from create_db import create_database_original, datasets


class GARFColumnPreparation(Preparation):
    def __init__(self, 
                 directory: str, 
                 dataset: str, 
                 error_generator: int, 
                 limit: int = -1):
        self.directory = directory
        self.dataset = dataset
        self.error_generator = error_generator
        self.limit = limit
    
    def run(self):
        if self.limit < 1:
            return
        
        process = Process(
            target=self._prepare, 
            args=(
                self.dataset, 
                self.limit, 
                self.error_generator,
                self.directory
            ))
        process.start()
        process.join()
    
    @staticmethod
    def _prepare(dataset: str = "Hospital",
                 limit: int = -1, 
                 error_generator: int = 1, 
                 method_dir: str = "garf_original"):
        # read data
        dataset_url = os.path.join(os.getcwd(), datasets[dataset][0])       
        temp_dataset = pd.read_csv(dataset_url)
        
        # we only use the first 10k rows
        temp_dataset = temp_dataset.loc[:10001]
        
        # remove label columns
        if "Label" in temp_dataset.columns:
            temp_dataset = temp_dataset.drop(columns=["Label"])
        if "labelvalue" in temp_dataset.columns:
            temp_dataset = temp_dataset.drop(columns=["labelvalue"])
        
        # create weights
        column_weights = np.array([1 for _ in temp_dataset.columns])
        city_index = temp_dataset.columns.get_loc("city")
        column_weights[city_index] = 2
        state_index = temp_dataset.columns.get_loc("statevalue")
        column_weights[state_index] = 3
        zip_code_index = temp_dataset.columns.get_loc("zip")
        column_weights[zip_code_index] = 3
        area_code_index = temp_dataset.columns.get_loc("areacode")
        column_weights[area_code_index] = 2
        column_weights = column_weights / np.sum(column_weights)
        
        # choose columns
        chosen_columns = np.random.choice(list(temp_dataset.columns), size=limit, replace=False, p=column_weights)
        temp_dataset = temp_dataset[chosen_columns]
        
        # readd label column
        temp_dataset["Label"] = ""
        dataset_url = os.path.join(os.path.dirname(dataset_url), f"{dataset}_{limit}_columns.csv")        
        temp_dataset.to_csv(dataset_url, index=False)
        
        # create database
        base_dir = os.path.join(os.getcwd(), "methods", "garf_original")
        create_database_original(
                base_dir=base_dir, 
                datasets={f"Tax ({limit} Columns)": [dataset_url]}, 
                error_generator=error_generator, 
                limit=-1
        )
