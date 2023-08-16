import os
import time
from typing import Tuple, Dict, Any
from multiprocessing import Process
from insert_error import insert_error_original, insert_error, insert_errors_bart, BART, ORIGINAL
from eva import evaluate_original, evaluate
from methods.rnn_garf_old.main import main as rnn_garf
from .experiment import Experiment


class RNNGARFOldExperiment(Experiment):
    def __init__(self, error_rate: float, method_name: str, dataset: str, error_generator: int = 1):
        super().__init__(error_rate, method_name, dataset, error_generator)
        self.base_dir = os.path.join(os.getcwd(), "methods", "rnn_garf_old")

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
        start_time = time.time()
        process_kwargs = {"flag": 2, "order": 1, "remove_amount_of_error_tuples": remove_amount_of_error_tuples, "dataset": self.dataset}
        process = Process(target=self.worker, args=(process_kwargs,))
        process.start()
        process.join()
        process = Process(target=self.worker, args=({"flag": 2, "order": 0, "dataset": self.dataset},))
        process.start()
        process.join()
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
        
        os.chdir(os.path.join(os.getcwd(), "methods", "rnn_garf_old"))
        rnn_garf(flag, order, remove_amount_of_error_tuples, dataset)
        
    def result(self) -> Tuple[float, float, float, float]:
        return self.precision, self.recall, self.f1, self.runtime
