import os
import time
import shutil
from typing import Tuple, Dict, Any
from multiprocessing import Process
from methods.garf_original.main import main as garf_original
from .experiment import Experiment
from insert_error import insert_error_original, insert_error, insert_errors_bart, BART, ORIGINAL
from eva import evaluate_original, evaluate


class ColumnOrderingExperiment(Experiment):
    def __init__(self, error_rate: float, method_name: str, dataset: str, error_generator: int = 1, rules_path: str = None):
        super().__init__(error_rate, method_name, dataset, error_generator)
        self.rules_path = rules_path
        self.base_dir = os.path.join(os.getcwd(), "methods", "garf_original")

    def run(self, **kwargs):
        path = f"{self.dataset}_copy"
        path_ori = path.strip('_copy')
        errors = kwargs["errors"] if "errors" in kwargs else None
        
        if not os.path.exists(self.rules_path):
            os.makedirs(self.rules_path)
        
        g_h = kwargs["g_h"] if "g_h" in kwargs else 64
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
        shutil.copy2(os.path.join(self.base_dir, "data", "save", "rules_final.txt"), os.path.join(self.rules_path, "rules_forward.txt"))
        
        process_kwargs["order"] = 0
        del process_kwargs["remove_amount_of_error_tuples"]
        
        process = Process(target=self.worker, args=(process_kwargs,))
        process.start()
        process.join()
        shutil.copy2(os.path.join(self.base_dir, "data", "save", "rules_final.txt"), os.path.join(self.rules_path, "rules_backward.txt"))
        end_time = time.time()
        
        if self.error_generator == ORIGINAL:
            precision, recall, f1 = evaluate_original(path_ori, path, self.base_dir)
        else:
            precision, recall, f1 = evaluate(path_ori, path, self.base_dir, errors)
            
        print(f"The complete runtime of this fix is {end_time - start_time} Error rate is {self.error_rate}")
        self.precision = precision
        self.recall = recall
        self.f1 = f1
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
        
        os.chdir(os.path.join(os.getcwd(), "methods", "garf_original"))
        garf_original(flag, order, remove_amount_of_error_tuples, dataset, g_h, g_e)
        
    def result(self) -> Tuple[float, float, float, float]:
        return self.precision, self.recall, self.f1, self.runtime
