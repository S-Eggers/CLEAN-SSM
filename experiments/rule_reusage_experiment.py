import os
import pandas as pd
from typing import Tuple, Dict, Any
from .experiment import Experiment
from multiprocessing import Process
from create_db import dataset_names, datasets
from methods.garf_original.main import main as garf_original
from insert_error import insert_error
from methods.lake_cleaner.utils import combine, count_rules, experiment_row_removal, experiment_col_removal, experiment_random_nan


class RuleReusageExperiment(Experiment):
    def __init__(self, error_rate: float, method_name: str, dataset: str, error_generator: int = 1):
        super().__init__(error_rate, method_name, dataset, error_generator)
        self.base_dir = os.getcwd()
        self.method_dir = os.path.join(os.getcwd(), "methods", "lake_cleaner")
        self.garf_base_dir = os.path.join(os.getcwd(), "methods", "garf_original")
        self.rule_path = os.path.join(self.method_dir, "rules", self.dataset)

    def run(self, **kwargs):
        path = f"{self.dataset}_copy"
        path_ori = path.strip('_copy')
        insert_error(path_ori, path, 0, self.garf_base_dir)
        if not os.path.exists(self.rule_path):
            os.makedirs(self.rule_path, exist_ok=True)
        # store original data
        orginal_data = pd.read_csv(os.path.join(self.base_dir, datasets[self.dataset][0]), dtype=str)
        if "Label" in orginal_data.columns:
            orginal_data = orginal_data.drop(columns=["Label"])
        if "labelvalue" in orginal_data.columns:
            orginal_data = orginal_data.drop(columns=["labelvalue"])
        
        # first GARF execution
        process_kwargs = {
            "flag": 2, 
            "order": 1, 
            "dataset": self.dataset,
        }
        process = Process(target=self.worker, args=(process_kwargs,))
        process.start()
        process.join()
        
        with open(os.path.join(self.garf_base_dir, "data", "save", "rules_final.txt"), "r") as f:
            rules_forward = eval(f.read())
        
        with open(os.path.join(self.rule_path, f"rules_forward_{self.dataset}.txt"), "w") as f:
            f.write(str(rules_forward))
        
        # second GARF execution
        process_kwargs["order"] = 0        
        process = Process(target=self.worker, args=(process_kwargs,))
        process.start()
        process.join()
        
        with open(os.path.join(self.garf_base_dir, "data", "save", "rules_final.txt"), "r") as f:
            rules_backward = eval(f.read())
     
        with open(os.path.join(self.rule_path, f"rules_backward_{self.dataset}.txt"), "w") as f:
            f.write(str(rules_backward))
            
        rules = combine(rules_forward, rules_backward)
        count_rules(rules)
        
        row_results = experiment_row_removal(orginal_data, rules, 5)
        row_results_df = self._format_results(row_results)
        
        random_nan_results = experiment_random_nan(orginal_data, rules, 5)
        random_nan_results_df = self._format_results(random_nan_results)
        
        col_results = experiment_col_removal(orginal_data, rules)
        col_results_df = self._format_results(col_results)
        
        self.results = row_results_df, random_nan_results_df, col_results_df
        return self
    
    def _format_results(self, results: Dict) -> pd.DataFrame:
        result = {
            "remove": [],
            "coverage": [],
            "dataset": []
        }
        for key, value in results.items():
            keys = [key] * len(value)
            result["remove"] = [*result["remove"], *keys]
            result["coverage"] = [*result["coverage"], *value]
            datasets_ = [dataset_names[self.dataset]] * len(value)
            result["dataset"] = [*result["dataset"], *datasets_]
        
        return pd.DataFrame(result)
    
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

    def result(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return self.results[0], self.results[1], self.results[2]
