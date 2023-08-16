import os
import time
from eva import evaluate
from .experiment import Experiment
from multiprocessing import Process
from typing import Dict, Any, Tuple
from insert_error import insert_error
from methods.lake_cleaner.utils import combine, count_rules
from methods.garf_original.main import main as garf_original


class GARFApplicabilityExperiment(Experiment):
    def __init__(self, error_rate: float, method_name: str, dataset: str, error_generator: int = 1):
        super().__init__(error_rate, method_name, dataset, error_generator)
        self.base_dir = os.getcwd()
        self.method_dir = os.path.join(os.getcwd(), "methods", "lake_cleaner")
        self.garf_base_dir = os.path.join(os.getcwd(), "methods", "garf_original")
        self.rule_path = os.path.join(self.method_dir, "rules", self.dataset)

    def run(self, **kwargs):        
        rules = self.get_rules(self.dataset)
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
        
    def get_rules(self, dataset: str):
        path = f"{dataset}_copy"
        path_ori = dataset
                
        errors = insert_error(path_ori, path, 0.01, self.garf_base_dir)
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
        

        precision, recall, f1 = evaluate(path_ori, path, self.garf_base_dir, errors)
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.runtime = end_time - start_time
        
        return rules

    def result(self) -> Tuple[float, float, float, float, int]:
        return self.precision, self.recall, self.f1, self.runtime, self.number_of_rules
