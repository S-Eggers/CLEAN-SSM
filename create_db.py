import sqlite3
import pandas as pd
from typing import Dict, Optional, List
from os.path import join
import re
import os


datasets = {
    "Beers": [join("datasets", "beers", "clean.csv"), join("datasets", "beers", "clean_changes.csv"), join("datasets", "beers", "dirty_clean.csv")],
    "Flights": [join("datasets", "flights", "clean.csv"), join("datasets", "flights", "clean_changes.csv"), join("datasets", "flights", "dirty_clean.csv")],
    "Food": [join("datasets", "food", "clean.csv"), join("datasets", "food", "clean_changes.csv"), join("datasets", "food", "dirty_clean.csv")],
    "Hospital": [join("datasets", "hospital", "clean.csv"), join("datasets", "hospital", "clean_changes.csv"), join("datasets", "hospital", "dirty_clean.csv")],
    "Rayyan": [join("datasets", "rayyan", "clean.csv"), join("datasets", "rayyan", "clean.csv"), join("datasets", "rayyan", "dirty.csv")],
    "Tax": [join("datasets", "tax", "clean.csv"), join("datasets", "tax", "clean_changes.csv"), join("datasets", "tax", "dirty_clean.csv")],
}

dataset_names = {
    "Beers": "Beers",
    "Flights": "Flights",
    "Food": "Food",
    "Hospital": "Hospital 10000",
    "Rayyan": "Rayyan",
    "Tax": "Tax",
}


def replace_quotes(s: str) -> str:
    if isinstance(s, str):
        return s.replace('"', '').replace("'", '')
    return s

def create_tables_for_dataset(connection: sqlite3.Connection, base_dir: str, name: str, url: List[str], error_generator: int = 1, limit: int = -1):
    df = pd.read_csv(url[0], dtype=str)
    
    for col in df.columns:
        df[col] = df[col].str.replace(',', ' ')

    if len(url) > 1:
        df["Label"] = None
    
    if "labelvalue" in df.columns:
        df = df.drop(columns=["labelvalue"])
    
    if len(url) > 1 and error_generator > 1:
        df2 = pd.read_csv(url[-1], dtype=str)
        df["Label"] = (~df.eq(df2)).any(axis=1).astype(int)
        df["Label"] = df["Label"].replace({0: None})
        
        if not os.path.exists(os.path.join(base_dir, "data", "save")):
            os.makedirs(os.path.join(base_dir, "data", "save"))
        _dict = {i: column for i, column in enumerate(df.loc[:, ~df.columns.isin(["Label"])].columns)} 
        with open(join(base_dir, "data", "save", "att_name.txt"), "w") as f:
            f.write(str(_dict))
    
    if limit > 0:
        df = df.loc[:limit-1]
        
    df = df.applymap(replace_quotes) # this method can't handle quotes out of the box
    for column in df.columns:
        if column == "Label":
            continue
        df[column] = df[column].fillna("")
    
    df.to_sql(f"{name}", connection, if_exists="replace", index=False)
    if error_generator > 0:
        df.to_sql(f"{name}_copy", connection, if_exists="replace", index=False)
    else:
        empty_df = df.iloc[0:0, :]
        empty_df.to_sql(f"{name}_copy", connection, if_exists="replace", index=False)
        empty_df.to_sql(f"{name}_copy1", connection, if_exists="replace", index=False)
        empty_df.to_sql(f"{name}_copy2", connection, if_exists="replace", index=False)

def create_database_original(base_dir: str, datasets: Dict[str, List[str]], database_url: Optional[str] = None, error_generator: int = 1, limit: int = -1):
    if database_url is None:
        print(join(base_dir, "database.db"))
        connection = sqlite3.connect(join(base_dir, "database.db"))
    else:
        connection = sqlite3.connect(join(base_dir, database_url))

    for name, url in datasets.items():
        print(f"Creating tables for {name} from file at {url[0]}...")
        create_tables_for_dataset(connection, base_dir, name, url, error_generator, limit)

    connection.close()
    print("Database successfully created.")

def create_tables_for_dataset_bart(connection: sqlite3.Connection, base_dir: str, name: str, url: List[str], limit: int = -1):
    df_clean = pd.read_csv(url[0], dtype=str)
    df_changes = pd.read_csv(url[1], dtype=str, header=None, names=["index.column", "new_value", "old_value"])
    df_dirty = pd.read_csv(url[2], dtype=str)
    
    # remove BART artifacts in column names
    dirty_columns = list(df_dirty.columns)
    cleaned_column_names = []
    for column in dirty_columns:
        cleaned_name = re.sub(r'\(.*\)', '', column)
        cleaned_name = cleaned_name.strip()
        cleaned_column_names.append(cleaned_name)
    
    df_dirty.columns = cleaned_column_names
    
    # split index.column into index and column
    df_changes[["index", "column"]] = df_changes["index.column"].str.split(".", expand=True)
    df_changes = df_changes.drop(columns=["index.column"])
    
    
    if "labelvalue" in df_clean.columns:
        df_clean = df_clean.drop(columns=["labelvalue"])
    
    if "labelvalue" in df_dirty.columns:
        df_dirty = df_dirty.drop(columns=["labelvalue"])
        
    df_clean["Label"] = None
    df_dirty["Label"] = (~df_clean.eq(df_dirty)).any(axis=1).astype(int)
    
    if not os.path.exists(os.path.join(base_dir, "data", "save")):
        os.makedirs(os.path.join(base_dir, "data", "save"))
    _dict = {i: column for i, column in enumerate(df_clean.loc[:, ~df_clean.columns.isin(["Label"])].columns)} 
    with open(join(base_dir, "data", "save", "att_name.txt"), "w") as f:
        f.write(str(_dict))
    
    if limit > 0:
        df_clean = df_clean.loc[:limit-1]
        df_dirty = df_dirty.loc[:limit-1]
        df_changes = df_changes[df_changes["index"].astype(int) < limit]
        
    df_clean = df_clean.applymap(replace_quotes) # this method can't handle quotes out of the box
    df_dirty = df_dirty.applymap(replace_quotes) # this method can't handle quotes out of the box
    for column in df_clean.columns:
        if column == "Label":
            continue
        df_clean[column] = df_clean[column].fillna("")
        df_dirty[column] = df_dirty[column].fillna("")
    
    df_clean.to_sql(f"{name}", connection, if_exists="replace", index=False)
    # save 2 times in order to restore dirty version after cleaning
    df_dirty.to_sql(f"{name}_copy", connection, if_exists="replace", index=False)
    df_dirty.to_sql(f"{name}_dirty", connection, if_exists="replace", index=False)
    df_changes.to_sql(f"{name}_changes", connection, if_exists="replace", index=False)


def create_database(base_dir: str, datasets: Dict[str, List[str]], database_url: Optional[str] = None, limit: int = -1):
    if database_url is None:
        print(join(base_dir, "database.db"))
        connection = sqlite3.connect(join(base_dir, "database.db"))
    else:
        connection = sqlite3.connect(join(base_dir, database_url))

    for name, url in datasets.items():
        print(f"Creating tables for {name} from file at {url[0]}...")
        create_tables_for_dataset_bart(connection, base_dir, name, url, limit)

    connection.close()
    print("Database successfully created.")
    
    
if __name__ == '__main__':
    create_database(datasets)
