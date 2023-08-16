import os
import tqdm
import sqlite3
import pandas as pd
from .preparation import Preparation
from insert_error import insert_error_unidetect


class LakeCleanerPreparation(Preparation):
    def __init__(self,
                 method: str,
                 error_rate: float = 0.1,
                 limit: int = -1):
        self.method = method
        self.error_rate = error_rate
        self.method_dir = os.path.join(os.getcwd(), "methods", self.method)
        self.base_directory = os.path.join(os.getcwd(), "datasets", "data-gov-sandbox")
        self.limit = limit
        
    def run(self):
        datasets = list(os.listdir(self.base_directory))
        if self.limit > 0 and self.limit < len(datasets):
            datasets = datasets[:self.limit]
        
        progress_bar = tqdm.tqdm(datasets)
        for directory in progress_bar:
            csv_directory = os.path.join(self.base_directory, directory)
            if os.path.isdir(csv_directory):
                csv_path = os.path.join(csv_directory, "clean.csv")
                if os.path.exists(csv_path):
                    new_directory = os.path.join(self.method_dir, "datasets", "artificial_lake", directory)
                    # skip already prepared directories, we won't change error rate
                    #if os.path.exists(new_directory):
                    #    continue
                    os.makedirs(new_directory, exist_ok=True)
    
                    df_clean = pd.read_csv(csv_path, dtype=str)
                    for col in df_clean.columns:
                        # remove special characters, GARF can not handle them, should be improved in the future
                        df_clean[col] = df_clean[col].str.replace('[^a-zA-Z0-9 .]', ' ', regex=True)
                    
                    if "labelvalue" in df_clean.columns:
                        df_clean = df_clean.drop(columns=["labelvalue"])

                        
                    # df_clean = df_clean.loc[:500]
                    
                    progress_bar.write(f"Inserting errors into {directory}...")
                    df_dirty, df_clean, _ = insert_error_unidetect(df_clean, self.error_rate, new_directory, verbose=False)
                    
                    progress_bar.write(f"Writing att_name.txt to {directory}...")
                    _dict = {i: column for i, column in enumerate(df_clean.loc[:, ~df_clean.columns.isin(["Label"])].columns)} 
                    with open(os.path.join(new_directory, "att_name.txt"), "w") as f:
                        f.write(str(_dict))
                    
                    progress_bar.write(f"Creating database in {directory}...")
                    connection = sqlite3.connect(os.path.join(new_directory, "database.db"))
                    df_clean.to_sql(f"{directory}", connection, if_exists="replace", index=False)
                    df_dirty.to_sql(f"{directory}_copy", connection, if_exists="replace", index=False)
                    df_dirty.to_sql(f"{directory}_err_ori", connection, if_exists="replace", index=False)
                    
                    progress_bar.write(f"Finished preparing data lake tables for {directory}.")