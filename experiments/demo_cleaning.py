import os
import time
import sqlite3
import pandas as pd
from typing import Tuple, Dict, Any
from multiprocessing import Process
from methods.garf_custom.main import main as garf_original
from .experiment import Experiment


class DemoCleaning(Experiment):
    def __init__(self, error_rate: float, method_name: str, dataset: str, error_generator: int = 1):
        super().__init__(error_rate, method_name, dataset, error_generator)
        self.base_dir = os.path.join(os.getcwd(), "methods", "garf_custom")

    def run(self, **kwargs):
        path = f"{self.dataset}_copy"
        path_ori = path.strip('_copy')
        # insert errors
        g_h = kwargs["g_h"] if "g_h" in kwargs else 32
        g_e = kwargs["g_e"] if "g_e" in kwargs else 64
        
        remove_amount_of_error_tuples = kwargs["remove_amount_of_error_tuples"]
        start_time = time.time()
        process_kwargs = {
            "flag": 2, 
            "order": 1, 
            "remove_amount_of_error_tuples": remove_amount_of_error_tuples, 
            "dataset": self.dataset,
            "g_h": g_h,
            "g_e": g_e,
        }
        process = Process(target=self.worker, args=(process_kwargs,))
        process.start()
        process.join()
        
        process_kwargs["order"] = 0
        del process_kwargs["remove_amount_of_error_tuples"]
        
        process = Process(target=self.worker, args=(process_kwargs,))
        process.start()
        process.join()
        end_time = time.time()
        
        connection = sqlite3.connect(os.path.join(self.base_dir, f"database.db"))
        df = pd.read_sql(f"SELECT * FROM {path}", connection)
        df.to_csv(os.path.join(self.base_dir, f"{path}.csv"), index=False)
        
        self.result = pd.DataFrame()                
        self.runtime = end_time - start_time
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
        if "remove_amount_of_error_tuples" in kwargs:
            remove_amount_of_error_tuples = kwargs["remove_amount_of_error_tuples"]
        flag = kwargs["flag"]
        order = kwargs["order"]
        dataset = kwargs["dataset"]
        g_h = kwargs["g_h"] if "g_h" in kwargs else 64
        g_e = kwargs["g_e"] if "g_e" in kwargs else 64
        
        os.chdir(os.path.join(os.getcwd(), "methods", "garf_custom"))
        garf_original(flag, order, remove_amount_of_error_tuples, dataset, g_h, g_e)
        
    def result(self) -> Tuple[float, float, float, float]:
        return self.precision, self.recall, self.f1, self.runtime
