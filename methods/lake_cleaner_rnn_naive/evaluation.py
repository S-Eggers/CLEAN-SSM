import os
import numpy as np
import pandas as pd
from typing import List, Tuple
from eva import evaluate as garf_evaluation

def single_evaluation(lake_base_path: str, subdir: str):
    try:
        base_dir = os.path.join(lake_base_path, subdir)
        errors = np.genfromtxt(os.path.join(base_dir, "errors.csv"), delimiter=',', dtype=int)
        
        precision, recall, f1 = garf_evaluation(subdir, f"{subdir}_copy", base_dir, errors, "cleaned.db", store_report=False)
        
        print(precision, recall, f1)
        
        return pd.Series([precision, recall, f1])
    except:
        return pd.Series([-1, 0, 0])

def calculate_means(result: pd.DataFrame) -> Tuple[float, float, float]:
    result = result.copy()
    # remove datasets where we found nothing
    result = result[result["precision"] > -1]
    mean_precision = result['precision'].mean()
    mean_recall = result['recall'].mean()
    mean_f1 = result['f1'].mean()
       
    return mean_precision, mean_recall, mean_f1

def evaluate(output_path: str, lake_base_path: str, result: pd.DataFrame) -> pd.DataFrame:
    result = result.copy()
    result["precision"] = -1
    result["recall"] = 0
    result["f1"] = 0
    
    mask = result["executed"]
    
    result.loc[mask, ["precision", "recall", "f1"]] = result.loc[mask].apply(lambda row: single_evaluation(lake_base_path, row["dataset"]), axis=1).values    
    result.to_csv(os.path.join(output_path, "results.csv"), index=False)
    
    mean_precision, mean_recall, mean_f1 = calculate_means(result)
    print(f"Mean precision (ignored datasets included): {mean_precision}")
    print(f"Mean recall (ignored datasets included): {mean_recall}")
    print(f"Mean f1 (ignored datasets included): {mean_f1}")
    
    mean_precision, mean_recall, mean_f1 = calculate_means(result[mask])
    print(f"Mean precision (ignored datasets excluded): {mean_precision}")
    print(f"Mean recall (ignored datasets excluded): {mean_recall}")
    print(f"Mean f1 (ignored datasets excluded): {mean_f1}")
    
    return result