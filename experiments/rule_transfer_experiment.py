import os
import pandas as pd
from typing import Tuple, Dict, Any
from .experiment import Experiment
from multiprocessing import Process
from create_db import dataset_names, datasets
from methods.garf_original.main import main as garf_original
from insert_error import insert_error
from methods.lake_cleaner.utils import combine, count_rules, get_applicable_rules, get_assignments, get_dataframe_vocab, get_rule_vocab, neg_log_avg_score, vocab_overlap, jaccard_similarity


class RuleTransferExperiment(Experiment):
    def __init__(self, error_rate: float, method_name: str, dataset: str, error_generator: int = 1):
        super().__init__(error_rate, method_name, dataset, error_generator)
        self.base_dir = os.getcwd()
        self.method_dir = os.path.join(os.getcwd(), "methods", "lake_cleaner")
        self.garf_base_dir = os.path.join(os.getcwd(), "methods", "garf_original")
        self.rule_path = os.path.join(self.method_dir, "rules", self.dataset)

    def run(self, **kwargs):
        # insert errors
        data = pd.read_csv("./datasets/tax/clean.csv", dtype=str)
        df_vocab = get_dataframe_vocab(data)
        if "Label" in data.columns:
            data = data.drop(columns=["Label"])
        
        rules = self.get_rules(self.dataset)
        number_of_rules = sum(count_rules(rules).values())
        
        lhs_vocab, rhs_vocab = get_rule_vocab(rules)        
        lhs_assignment, _ = get_assignments(lhs_vocab, df_vocab, 0.5, scoring_fn=neg_log_avg_score, similarity_fn=vocab_overlap)
        rhs_assignment, _ = get_assignments(rhs_vocab, df_vocab, 0.01, scoring_fn=neg_log_avg_score, similarity_fn=jaccard_similarity)
        applicable_rules = get_applicable_rules(data, rules, lhs_assignment, rhs_assignment)
        number_of_applicable_rules = sum(count_rules(applicable_rules).values())
        
        self.results = pd.DataFrame({
            "dataset": [self.dataset],
            "found_rules": [number_of_rules],
            "transferable_rules": [number_of_applicable_rules],
            "transferable_rules_dict": [str(applicable_rules)]
        })
        return self
    
    @staticmethod
    def _format_results(results: Dict) -> pd.DataFrame:
        result = {
            "remove": [],
            "coverage": [],
            "dataset": []
        }
        for key, value in results.items():
            keys = [key] * len(value)
            result["remove"] = [*result["remove"], *keys]
            result["coverage"] = [*result["coverage"], *value]
        
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
        
    def get_rules(self, dataset: str):
        path = f"{dataset}_copy"
        path_ori = path.strip('_copy')
        insert_error(path_ori, path, 0, self.garf_base_dir)
        if not os.path.exists(self.rule_path):
            os.makedirs(self.rule_path, exist_ok=True)
        # store original data
        orginal_data = pd.read_csv(os.path.join(self.base_dir, datasets[dataset][0]), dtype=str)
        
        if "Label" in orginal_data.columns:
            orginal_data = orginal_data.drop(columns=["Label"])
        if "labelvalue" in orginal_data.columns:
            orginal_data = orginal_data.drop(columns=["labelvalue"])
        
        # first GARF execution
        process_kwargs = {
            "flag": 2, 
            "order": 1, 
            "dataset": dataset,
        }
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
        
        with open(os.path.join(self.garf_base_dir, "data", "save", "rules_final.txt"), "r") as f:
            rules_backward = eval(f.read())
     
        with open(os.path.join(self.rule_path, f"rules_backward_{dataset}.txt"), "w") as f:
            f.write(str(rules_backward))
            
        rules = combine(rules_forward, rules_backward)
        return rules

    def result(self) -> pd.DataFrame:
        return self.results
