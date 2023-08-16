import os
import time
import sqlite3
import numpy as np
import pandas as pd
from eva import evaluate
from .experiment import Experiment
from multiprocessing import Process
from typing import Dict, Any, Tuple
from methods.lake_cleaner.utils import combine, count_rules
from methods.garf_original.main import main as garf_original


class GARFEnhancementExperiment(Experiment):
    def __init__(self, error_rate: float, method_name: str, dataset: str, error_generator: int = 1):
        super().__init__(error_rate, method_name, dataset, error_generator)
        self.base_dir = os.getcwd()
        self.method_dir = os.path.join(os.getcwd(), "methods", "lake_cleaner")
        self.garf_base_dir = os.path.join(os.getcwd(), "methods", "garf_original")
        self.rule_path = os.path.join(self.method_dir, "rules", self.dataset)

    def run(self, **kwargs):        
        rules = self.get_rules(self.dataset, kwargs["errors"], kwargs["dataset_clean"], kwargs["dataset_dirty_ori"])
        self.number_of_rules = sum(count_rules(rules).values())
        return self
    
    @staticmethod
    def worker(kwargs: Dict[str, Any] = dict()):
        if "flag" not in kwargs:
            raise ValueError("Flag is not set")
        if "order" not in kwargs:
            raise ValueError("Order is not set")
        if "dataset" not in kwargs:
            raise ValueError("Dataset is not set")

        remove_amount_of_error_tuples = 0
        flag = kwargs["flag"]
        order = kwargs["order"]
        dataset = kwargs["dataset"]
        g_h = 64
        g_e = 64
        
        os.chdir(os.path.join(os.getcwd(), "methods", "garf_original"))
        garf_original(flag, order, remove_amount_of_error_tuples, dataset, g_h, g_e)
        
    def get_rules(self, dataset: str, errors: np.ndarray, dataset_clean: pd.DataFrame, dataset_dirty_ori: pd.DataFrame):
        path = f"{dataset}_copy"
        path_ori = dataset
                
        if not os.path.exists(self.rule_path):
            os.makedirs(self.rule_path, exist_ok=True)        
        # first GARF execution
        process_kwargs = {
            "flag": 2, 
            "order": 1, 
            "dataset": dataset,
        }
        start_time = time.time()
        process = Process(target=self.worker, args=(process_kwargs,))
        process.start()
        process.join()
        
        with open(os.path.join(self.garf_base_dir, "data", "save", "rules_final.txt"), "r") as f:
            rules_forward = eval(f.read())
        
        with open(os.path.join(self.rule_path, f"rules_forward_{dataset}.txt"), "w") as f:
            f.write(str(rules_forward))
        
        # second GARF execution
        process_kwargs["order"] = 0        
        process = Process(target=self.worker, args=(process_kwargs,))
        process.start()
        process.join()
        end_time = time.time()
        
        with open(os.path.join(self.garf_base_dir, "data", "save", "rules_final.txt"), "r") as f:
            rules_backward = eval(f.read())
     
        with open(os.path.join(self.rule_path, f"rules_backward_{dataset}.txt"), "w") as f:
            f.write(str(rules_backward))
            
        rules = combine(rules_forward, rules_backward)
        
        """
        # remove duplicates
        id_keys = {
            "beers": ["indexvalue"],
            "rayyan": ["id"],
            "flights": ["tupleid"],
            "food": ["inspectionid", "license"],
            "hospital": ["providerid", "measureid"],
        }
        conn = sqlite3.connect(os.path.join(self.garf_base_dir, "database.db"))
        data = pd.read_sql_query(f"SELECT * FROM \"{path}\"", conn)
        data = data.drop_duplicates(subset=id_keys[dataset_name.lower()], keep="first")
        assert len(data) == n_tuples, f"Number of tuples is not correct: {len(data)} != {n_tuples}"
        data.to_sql(path, conn, if_exists="replace", index=False)
        """
        conn = sqlite3.connect(os.path.join(self.garf_base_dir, "database.db"))
        dataset_clean.to_sql(path_ori, conn, if_exists="replace", index=False)
        dataset_dirty_ori.to_sql(f"{path_ori}_err_ori", conn, if_exists="replace", index=False)
        conn.close()
        
        precision, recall, f1 = evaluate(path_ori, path, self.garf_base_dir, errors)
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.runtime = end_time - start_time
        
        return rules

    def result(self) -> Tuple[float, float, float, float, int]:
        return self.precision, self.recall, self.f1, self.runtime, self.number_of_rules
