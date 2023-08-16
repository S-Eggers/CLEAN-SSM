import os
import time
import sys
from pathlib import Path 
from shutil import copyfile, rmtree
from typing import Tuple, Dict, Any
from multiprocessing import Process
from methods.mpgarf.main import main as mpgarf
from .experiment import Experiment
from insert_error import insert_error_original, insert_error, insert_errors_bart, BART, ORIGINAL
from eva import evaluate_original, evaluate


class MPGARFExperiment(Experiment):
    def __init__(self, error_rate: float, method_name: str, dataset: str, error_generator: int = 1):
        super().__init__(error_rate, method_name, dataset, error_generator)
        self.base_dir = os.path.join(os.getcwd(), "methods", "mpgarf")

    def run(self, **kwargs):
        # insert errors
        path = f"{self.dataset}_copy"
        path_ori = path.strip('_copy')
        # insert errors
        if self.error_generator == ORIGINAL:
            insert_error_original(path_ori, path, self.error_rate, self.base_dir)
            errors = None
        elif self.error_generator == BART:
            errors = insert_errors_bart(path_ori, path, self.base_dir)
        else:
            errors = insert_error(path_ori, path, self.error_rate, self.base_dir)

        remove_amount_of_error_tuples = kwargs["remove_amount_of_error_tuples"]
        directory_id = kwargs["directory_id"]
        
        log = os.path.join(os.getcwd(), "log")
        # clear log
        with open(os.path.join(log, f"mpgarf_log_{directory_id}.txt"), "w"):
            pass
        
        base_dir = os.path.join(os.getcwd(), "methods", "mpgarf")
        new_base_dir = os.path.join(base_dir, "mp", directory_id)
        new_save_dir = os.path.join(new_base_dir, "data", "save")
        path = Path(new_save_dir)
        path.mkdir(exist_ok=True, parents=True)
        files = [
            "database.db",
            "config.ini", 
            os.path.join("data", "save", "att_name.txt"), 
            os.path.join("data", "save", "generated_sentences.txt"),
            os.path.join("data", "save", "rules_final.txt"),
            os.path.join("data", "save", "rules_read.txt"),
        ]
        for file in files:
            copyfile(os.path.join(base_dir, file), os.path.join(new_base_dir, file))

        start_time = time.time()
        process_kwargs = {
            "flag": 2, 
            "order": 1, 
            "remove_amount_of_error_tuples": remove_amount_of_error_tuples, 
            "dataset": self.dataset, 
            "directory_id": directory_id
        }
        process = Process(target=self.worker, args=(process_kwargs,))
        process.start()
        process.join()
        # change order to 0
        process_kwargs["order"] = 0
        process = Process(target=self.worker, args=(process_kwargs,))
        process.start()
        process.join()
        end_time = time.time()
    
        if self.error_generator == ORIGINAL:
            precision, recall, f1 = evaluate_original(path_ori, path, self.base_dir)
        else:
            precision, recall, f1 = evaluate(path_ori, path, self.base_dir, errors)
        
        rmtree(new_base_dir)
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
        if "directory_id" not in kwargs:
            raise ValueError("Directory ID is not set")
        remove_amount_of_error_tuples = 0
        if "remove_amount_of_error_tuples" in kwargs:
            remove_amount_of_error_tuples = kwargs["remove_amount_of_error_tuples"]
        flag = kwargs["flag"]
        order = kwargs["order"]
        dataset = kwargs["dataset"]
        directory_id = kwargs["directory_id"]
        log = os.path.join(os.getcwd(), "log")
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        with open(os.path.join(log, f"mpgarf_log_{directory_id}.txt"), "a") as f:
            sys.stdout = f
            sys.stderr = f
            base_dir = os.path.join(os.getcwd(), "methods", "mpgarf")
            new_base_dir = os.path.join(base_dir, "mp", directory_id)
            os.chdir(new_base_dir)
            mpgarf(flag, order, remove_amount_of_error_tuples, dataset)
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        
    def result(self) -> Tuple[float, float, float, float]:
        return self.precision, self.recall, self.f1, self.runtime
