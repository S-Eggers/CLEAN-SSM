import os
from .preparation import Preparation
from multiprocessing import Process
from create_db import create_database, create_database_original, datasets


class GARFPreparation(Preparation):
    def __init__(self, 
                 directory: str, 
                 dataset: str, 
                 error_generator: int, 
                 limit: int = -1):
        self.directory = directory
        self.dataset = dataset
        self.error_generator = error_generator
        self.limit = limit
    
    def run(self):
        process = Process(
            target=self._prepare, 
            args=(
                self.dataset, 
                self.limit, 
                self.error_generator,
                self.directory
            ))
        process.start()
        process.join()
    
    @staticmethod
    def _prepare(dataset: str = "Hospital",
                 limit: int = -1, 
                 error_generator: int = 1, 
                 method_dir: str = "garf_original"):
        base_dir = os.path.join(os.getcwd(), "methods", method_dir)
        if error_generator < 2:
            create_database_original(
                base_dir=base_dir, 
                datasets={dataset: datasets[dataset]}, 
                error_generator=error_generator, 
                limit=limit
            )
        else:
            create_database(
                base_dir=base_dir, 
                datasets={dataset: datasets[dataset]}, 
                limit=limit
            )
    