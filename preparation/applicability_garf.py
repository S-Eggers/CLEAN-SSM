import os
import pandas as pd
import numpy as np
from .preparation import Preparation
from multiprocessing import Process
from create_db import create_database_original, datasets


class GARFApplicabilityPreparation(Preparation):
    def __init__(self, 
                 directory: str, 
                 dataset: str, 
                 error_generator: int, 
                 limit: int = -1):
        self.directory = directory
        self.dataset = dataset
        self.error_generator = error_generator
        self.limit = limit
    
    def run(self, dataset: pd.DataFrame):        
        process = Process(
            target=self._prepare, 
            args=(
                self.dataset, 
                self.limit, 
                self.error_generator,
                self.directory,
                dataset
            ))
        process.start()
        process.join()
    
    @staticmethod
    def _prepare(dataset: str = "Hospital",
                 limit: int = -1, 
                 error_generator: int = 1, 
                 method_dir: str = "garf_original",
                 dataframe: pd.DataFrame = None):
        # read data
        dataframe = dataframe.copy()
        if limit > 0:
            dataframe = dataframe.loc[:limit]
        # readd label column
        dataframe["Label"] = ""
        dataset_url = os.path.join(os.path.dirname(os.getcwd()), f"{dataset}_applicability.csv")        
        dataframe.to_csv(dataset_url, index=False)
        # create database
        base_dir = os.path.join(os.getcwd(), "methods", "garf_original")
        create_database_original(
                base_dir=base_dir, 
                datasets={f"{dataset}_applicability": [dataset_url]}, 
                error_generator=error_generator, 
                limit=-1
        )
