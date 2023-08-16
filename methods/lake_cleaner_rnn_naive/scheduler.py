import os
import tqdm
import logging
import pandas as pd
from typing import List


class Scheduler:
    def __init__(self, lake_base_path: str, max_dataset_size: int = 25000, min_dataset_size: int = 500):
        self.lake_base_path = lake_base_path
        self.datasets = 0
        self.garf_executions = 0
        self.dataset_to_garf_executions = dict()
        self.max_dataset_size = max_dataset_size
        self.min_dataset_size = min_dataset_size
            
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
    
    def __iter__(self):
        if not os.path.exists(self.lake_base_path):
            raise ValueError("Data lake base path does not exist")
        
        file_paths = os.listdir(self.lake_base_path)
        file_paths = list(sorted(file_paths))
        file_paths = [file_path for file_path in file_paths if os.path.isdir(os.path.join(self.lake_base_path, file_path))]
        
        for subdir in file_paths:
            self.datasets += 1
            file_path =  os.path.join(self.lake_base_path, subdir, "dirty.csv")
            df = pd.read_csv(file_path)
            
            # heuristic 1: skip datasets that are too small or too large
            if df.shape[0] > self.max_dataset_size or df.shape[0] < self.min_dataset_size or df.shape[1] > 15:
                logging.info(f"Skipping dataset {subdir} due to size")
                self.dataset_to_garf_executions[subdir] = False        
                continue
            
            # heuristic 2: skip datasets that are too unique
            uniqueness_score = self.uniqueness_score(df)
            if uniqueness_score > self.max_uniqueness_score:
                logging.info(f"Skipping dataset {subdir} due to uniqueness ({uniqueness_score}))")
                self.dataset_to_garf_executions[subdir] = False     
                continue
            
            str_dtype_count, _, _ = self.count_dtypes(df)            
            # heuristic 3: skip only numeric datasets
            if str_dtype_count == 0:
                logging.info(f"Skipping dataset {subdir} due to only numeric columns")
                self.dataset_to_garf_executions[subdir] = False 
                continue

            self.garf_executions += 1
            self.dataset_to_garf_executions[subdir] = True        
            yield os.path.join(self.lake_base_path, subdir)
            
    def count_dtypes(self, data: pd.DataFrame):
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
    
    def uniqueness_score(self, data: pd.DataFrame) -> float:
        uniqueness_df = pd.DataFrame(index=data.columns, columns=["uniqueness"])
        for column in data.columns:
            uniqueness_df.loc[column] = data[column].nunique() / data.shape[0]
        return uniqueness_df["uniqueness"].mean()