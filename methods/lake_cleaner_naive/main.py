import os
import sys
import time
import logging
import pandas as pd
from methods.lake_cleaner_naive.garf import GARFRunner
from methods.lake_cleaner_naive.scheduler import Scheduler
from methods.lake_cleaner_naive.evaluation import evaluate


def single(lake_base_path: str, subdir: str):
    return subdir, subdir, subdir

def main(method_base_dir: str, lake_base_path: str) -> pd.DataFrame:
    start_time = time.time()
    
    scheduler = Scheduler(lake_base_path)
    dataset_to_time_consumption = dict()
    dataset_checkpoint_time = dict()
    
    
    
    #results = pd.read_csv(os.path.join(method_base_dir, "results.csv"))
    #results.loc[results["executed"], ["precision", "recall", "f1"]] = 1, 1, 1
    #print(results)
    #results_not_nan = results[results["executed"]]
    #print(results_not_nan)
    #results_not_nan.loc[results["executed"], ["precision", "recall", "f1"]] = results.loc[results["executed"]].apply(lambda row: pd.Series(single(1, row["dataset"])), axis=1).values
    #print(results_not_nan)
    
    
    for dataset_path in scheduler:
        dataset_start_time = time.time()
        # executor
        GARFRunner(dataset_path).run()
        dataset_end_time = time.time()
        subdir = os.path.basename(dataset_path)
        dataset_to_time_consumption[subdir] = dataset_end_time - dataset_start_time
        dataset_checkpoint_time[subdir] = time.time() - start_time
        scheduler.save_temporary_result(dataset_to_time_consumption, dataset_checkpoint_time)
        
    end_time = time.time()
    complete_execution_time = end_time - start_time
    logging.info(f"Complete execution time: {complete_execution_time}")
    results = evaluate(method_base_dir, lake_base_path, scheduler.get_results(dataset_to_time_consumption, dataset_checkpoint_time))
    return results
    
def evaluate_temp_result(method_base_dir: str, lake_base_path: str, temp_result_path: str):
    temp_result = pd.read_csv(temp_result_path)
    evaluate(method_base_dir, lake_base_path, temp_result)
