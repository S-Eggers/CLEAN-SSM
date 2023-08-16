import os
import shutil
from typing import Dict, Any
from multiprocessing import Process
from methods.rnn_garf.main import main as garf


class GARFRunner():
    def __init__(self, lake_dataset_path: str):
        self.lake_dataset_path = lake_dataset_path
        self.base_dir = os.path.join(os.getcwd(), "methods", "rnn_garf")

    def run(self):        
        new_database_path = os.path.join(self.base_dir, "database.db")
        shutil.copy2(os.path.join(self.lake_dataset_path, "database.db"), new_database_path)
        new_column_dict_path = os.path.join(self.base_dir, "data", "save", "att_name.txt")
        shutil.copy2(os.path.join(self.lake_dataset_path, "att_name.txt"), new_column_dict_path)
        shutil.copy2(os.path.join(self.lake_dataset_path, "att_name.txt"), os.path.join(self.base_dir, "att_name.txt"))
        
        dataset = os.path.basename(self.lake_dataset_path)
        process_kwargs = {
            "flag": 2, 
            "order": 1, 
            "dataset": dataset,
        }
        process = Process(target=self.worker, args=(process_kwargs,))
        process.start()
        process.join()
        process_kwargs["order"] = 0
        
        process = Process(target=self.worker, args=(process_kwargs,))
        process.start()
        process.join()
        shutil.copy2(new_database_path, os.path.join(self.lake_dataset_path, "cleaned.db"))
    
    @staticmethod
    def worker(kwargs: Dict[str, Any] = dict()):
        if "flag" not in kwargs:
            raise ValueError("Flag is not set")
        if "order" not in kwargs:
            raise ValueError("Order is not set")
        if "dataset" not in kwargs:
            raise ValueError("Dataset is not set")

        flag = kwargs["flag"]
        order = kwargs["order"]
        dataset = kwargs["dataset"]
        
        os.chdir(os.path.join(os.getcwd(), "methods", "rnn_garf"))
        garf(flag, order, 0, dataset)
