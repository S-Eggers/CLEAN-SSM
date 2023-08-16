import sqlite3
import random
import reset
import os
import pandas as pd
import numpy as np
import string
from itertools import combinations
from typing import Tuple


def insert_error_original(path_ori, path, error_rate, work_dir):
    print(path_ori, path)
    conn = sqlite3.connect(os.path.join(work_dir, "database.db"))
    cursor = conn.cursor()
    random.seed(1)
    path2 = path + "1"
    path3 = path + "2"
    reset.reset(path_ori, path, work_dir)
    sql = "DELETE FROM \"" + path2 + "\" "
    cursor.execute(sql)
    conn.commit() 
    sql = "DELETE FROM \"" + path3 + "\" "
    cursor.execute(sql)
    conn.commit()  
    sql1 = "select * from \"" + path + "\" "
    print(sql1)
    cursor.execute(sql1)
    data1 = cursor.fetchall()  # All data
    des = cursor.description
    t1 = len(data1)   # Total data -1 can be changed to index position
    t2 = len(data1[0]) - 1  # Length of data per row, -1 is to remove the label column
    att_name=[]
    for item in des:
        att_name.append(item[0])
    dict={}
    for i in range(t2):    #
        dict[i]=att_name[i]
    print(dict)
    if not os.path.exists(os.path.join(work_dir, "data", "save")):
        os.makedirs(os.path.join(work_dir, "data", "save"))
    
    f = open(os.path.join(work_dir, "data", "save", "att_name.txt"), 'w')
    f.write(str(dict))
    f.close()
    att = list(dict.values())
    count_error_all=int(error_rate*t1)
    error_index=random.sample(range(0,t1),count_error_all)    #Random 10 different numbers out of 10,000 numbers from 0-9999
    print(len(error_index))
    count=0
    for index in error_index:
        row=data1[index]
        for i in range(t2):  # t2
            if i == 0:
                sql_inf = f"\"{att[i]}\"='{row[i]}'"
            else:
                sql_inf += f" and \"{att[i]}\"='{row[i]}'"
        sql_info = sql_inf + " and \"Label\"='None'"
        if count<int(count_error_all/3):
            error_list = ["error1", "error2", "error3", "error4", "error5"]
            r = random.randint(1, t2 - 1)  # Assume a total of 9 columns, -1 divided by the last column label when calculating t2, now -1 again is to become the index position, so the number of random in 1-7
            r2 = random.randint(0, 4)
            error = error_list[r2]
        elif count<int(2*count_error_all/3):
            r = random.randint(1, t2 - 1)
            error="missing"                 # Note, here you can write error=" " as the missing value, but the execution of the model detection will replace all null values with "missing", so here it is written directly as missing
        else:
            r = random.randint(1, t2 - 1)
            sql = "select distinct (\"" + att[r] + "\") from \"" + path + "\""
            cursor.execute(sql)
            values = cursor.fetchall()
            values = [value for value in values if value[0] != row[r]]
            if values:  # Only if values is not empty
                error_value = random.choice(values)  # Other values of the same column
                error=error_value[0]
            else:
                error = None 

        if (error is None):
            error=""
        sql2 = f"update \"{path}\" set \"Label\"='1' , \"{att[r]}\"='{error}' where {sql_info}" #and \"" + att[r] + "\"='error'
        cursor.execute(sql2)
        conn.commit()
        
        # Generate Hosp_rules_copy2
        sql = "select * from \"" + path_ori + "\" where  " + sql_inf + ""
        cursor.execute(sql)
        data_clean = cursor.fetchall()
        # print(sql)
        # print(data_clean)
        t3 = len(data_clean[0])  # Length of data per row
        for num in range(t3):
            if num == 0:
                sql_before = "'%s'"
            else:
                sql_before = sql_before + ",'%s'"
        # print(sql_before)
        va = []
        for num in range(t3):
            va.append(data_clean[0][num])
        sql_after = tuple(va)
        sql_clean = "insert into \"" + path3 + "\" values(" + sql_before + ")"
        sql3=sql_clean% (sql_after)
        # print(sql4)
        cursor.execute(sql3)
        conn.commit()   # Reset
        sql4 = "insert into \"" + path2 + "\" values(" + sql_before + ")"
        sql5 = sql4 % (sql_after)
        # print(sql4)
        cursor.execute(sql5)
        conn.commit()
        sql_dirty = f"update \"{path2}\" set \"Label\"='1' , \"{att[r]}\"='{error}' where {sql_inf}"
        cursor.execute(sql_dirty)
        conn.commit()
        count = count + 1

    sql_check = "select * from \"" + path + "\" where \"Label\"='1'"
    cursor.execute(sql_check)
    data2 = cursor.fetchall()
    print("Number of Generated Errors:", len(data2))    # Generating an erroneous redundancy expectation may be due to the presence of duplicate items in the data set
    cursor.close()
    conn.close()

"""
New error generation
"""
MAX_FD = 4
def find_functional_dependencies(df):
    columns = list(df.columns)
    columns.remove('Label')
    dict = {}
    for i in range(1, MAX_FD):
        for subset in combinations(columns, i):
            remaining = [col for col in columns if col not in subset]
            
            for col in remaining:
                if df.groupby(list(subset))[col].nunique().max() == 1:
                    if col not in dict:
                        dict[col] = [subset]
                    else:
                        dict[col].append(subset)
    return dict

def get_random_cells(dataframe, error_rate):
    if error_rate <= 0:
        return []

    num_rows, num_cols = dataframe.shape
    cells = num_rows * (num_cols - 1) # exclude label column
    sample_size = int(error_rate * cells)
    selected_cells = []
    column_indicies = range(len(dataframe.columns) - 1) # exclude label column
    while len(selected_cells) < sample_size:
        random_index = random.choice(dataframe.index)
        random_column = random.choice(column_indicies)
        cell = (random_index, random_column)
        if cell not in selected_cells:
            selected_cells.append(cell)
    
    return selected_cells

def insert_error(path_ori, path, error_rate, work_dir):
    print(path_ori, path)
    random.seed(1)
    
    # reset tables
    reset.reset(path_ori, path, work_dir)
    
    # connect to database
    conn = sqlite3.connect(os.path.join(work_dir, "database.db"))
    sql = f"select * from \"{path}\""
    df = pd.read_sql(sql, conn)
    df["Label"] = 0
    df_original = df.copy()
    errors = np.zeros(shape=df.shape, dtype=int)
    fds = find_functional_dependencies(df)
    
    # save before
    df.to_csv(os.path.join(work_dir, "before.csv"), index=False)
    
    # generate errors
    cells = get_random_cells(df, error_rate)
    column_is_str = dict()
    for cell in cells:
        # store this action, would be expensive to calculate it every time
        column_name = df.columns[cell[1]]
        if column_name not in column_is_str:
            column_str = df[column_name].apply(lambda x: isinstance(x, str)).all()
            column_is_str[column_name] = column_str
        else:
            column_str = column_is_str[column_name]
        
        # if there exist a functional dependency and we have a string column, we can choose from all error types
        if column_name in fds and column_str:
            func = np.random.choice([_f_missing, _f_cell_swap, _f_functional_dependency, _f_spelling], p=[2/7, 1/7, 2/7, 2/7])
        # if we have a string column and no functional dependency, we can choose from all error types except functional dependency
        elif column_str:
            func = np.random.choice([_f_missing, _f_cell_swap, _f_spelling], p=[2/5, 1/5, 2/5])
        # we have no string column but a functional dependency, we can choose from all error types except spelling
        elif column_name in fds:
            func = np.random.choice([_f_missing, _f_cell_swap, _f_functional_dependency], p=[2/5, 1/5, 2/5])
        # no string column and no functional dependency, we can choose from missing cell and cell swap error type   
        else:
            func = np.random.choice([_f_missing, _f_cell_swap], p=[2/3, 1/3])

        df, errors = func(df, df_original, errors, cell)
    
    # save after
    df.to_csv(os.path.join(work_dir, "after.csv"), index=False)
    np.savetxt(os.path.join(work_dir, "errors.csv"), errors, delimiter=",", fmt="%d")
    
    # insert into database
    df.to_sql(path, conn, if_exists="replace", index=False)
    df.to_sql(f"{path_ori}_err_ori", conn, if_exists="replace", index=False)
    print(f"Number of Cells: {df.shape[0] * (df.shape[1] - 1)}, Number of Generated Errors: {len(cells)}")
    return errors


def insert_error_unidetect(df: pd.DataFrame, error_rate: float, work_dir: str, verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    random.seed(1)

    df["Label"] = 0
    df_original = df.copy()
    errors = np.zeros(shape=df.shape, dtype=int)
    fds = find_functional_dependencies(df)
    
    # save before
    df.to_csv(os.path.join(work_dir, "original.csv"), index=False)
    
    # generate errors
    cells = get_random_cells(df, error_rate)
    column_is_str = dict()
    for cell in cells:
        # store this action, would be expensive to calculate it every time
        column_name = df.columns[cell[1]]
        if column_name not in column_is_str:
            column_str = df[column_name].apply(lambda x: isinstance(x, str)).all()
            column_is_str[column_name] = column_str
        else:
            column_str = column_is_str[column_name]
        
        # if there exist a functional dependency and we have a string column, we can choose from all error types
        if column_name in fds and column_str:
            func = np.random.choice([_f_missing, _f_cell_swap, _f_functional_dependency, _f_spelling], p=[2/7, 1/7, 2/7, 2/7])
        # if we have a string column and no functional dependency, we can choose from all error types except functional dependency
        elif column_str:
            func = np.random.choice([_f_missing, _f_cell_swap, _f_spelling], p=[2/5, 1/5, 2/5])
        # we have no string column but a functional dependency, we can choose from all error types except spelling
        elif column_name in fds:
            func = np.random.choice([_f_missing, _f_cell_swap, _f_functional_dependency], p=[2/5, 1/5, 2/5])
        # no string column and no functional dependency, we can choose from missing cell and cell swap error type   
        else:
            func = np.random.choice([_f_missing, _f_cell_swap], p=[2/3, 1/3])

        df, errors = func(df, df_original, errors, cell)
    
    # save after
    df.to_csv(os.path.join(work_dir, "dirty.csv"), index=False)
    np.savetxt(os.path.join(work_dir, "errors.csv"), errors, delimiter=",", fmt="%d")
    
    # insert into database
    if verbose:
        print(f"Number of Cells: {df.shape[0] * (df.shape[1] - 1)}, Number of Generated Errors: {len(cells)}")
    return df, df_original, errors

def insert_specific_error(df: pd.DataFrame, error_rate: float, work_dir: str, error_type: str, verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    random.seed(1)
    df = df.copy()
    df["Label"] = 0
    df_original = df.copy()
    errors = np.zeros(shape=df.shape, dtype=int)
    fds = find_functional_dependencies(df)
    
    # save before
    df.to_csv(os.path.join(work_dir, "original.csv"), index=False)
        
    # generate errors
    cells = get_random_cells(df, error_rate if error_type != "cell_swap" else error_rate / 2)
    column_is_str = dict()
    for cell in cells:
        # store this action, would be expensive to calculate it every time
        column_name = df.columns[cell[1]]
        if column_name not in column_is_str:
            column_str = df[column_name].apply(lambda x: isinstance(x, str)).all()
            column_is_str[column_name] = column_str
        else:
            column_str = column_is_str[column_name]
        
        if error_type == "missing":
            func = _f_missing
        elif error_type == "cell_swap":
            func = _f_cell_swap
        elif error_type == "spelling" and column_str:
            func = _f_spelling
        elif error_type == "functional_dependency" and column_name in fds:
            func = _f_functional_dependency
        elif error_type == "outlier" and not column_str:
            func = _f_outlier
        elif error_type == "transformation" and column_str:
            func = _f_transformation
        else:
            # have to ignore cell
            continue
        
        df, errors = func(df, df_original, errors, cell)
    
    # save after
    df.to_csv(os.path.join(work_dir, "dirty.csv"), index=False)
    np.savetxt(os.path.join(work_dir, "errors.csv"), errors, delimiter=",", fmt="%d")
    
    # insert into database
    if verbose:
        print(f"Number of Cells: {df.shape[0] * (df.shape[1] - 1)}, Number of Generated Errors: {len(cells)}")
    return df, df_original, errors

MISSING = 1
SWAP = 2
SPELLING = 3
FUNCTIONAL_DEPENDENCY = 4
OUTLIER = 5
TRANSFORMATION = 6

def _f_transformation(df, df_original, errors, index):
    pass

def _f_outlier(df, df_original, errors, index):
    # sample defective rows     
    i, j = index
    if errors[i, j] != 0:
        return df, errors
    
    Q1 = df_original.iloc[:, j].quantile(0.25)
    Q3 = df_original.iloc[:, j].quantile(0.75)
    IQR = Q3 - Q1
    OUTLIER_VAL = 1.5 * IQR
    
    if np.random.choice([True, False]):
        new_value = Q3 + (OUTLIER_VAL * random.randint(1, 10))
    else:
        new_value = Q1 - (OUTLIER_VAL * random.randint(1, 10))
    
    if isinstance(df.iloc[i, j], int):
        new_value = int(new_value)
    
    df.iloc[i, j] = new_value
    df.loc[i, "Label"] = 1
    errors[i, j] = OUTLIER
    return df, errors


def _f_missing(df, df_original, errors, index):
    # sample defective rows     
    i, j = index
    if errors[i, j] != 0:
        return df, errors

    df.iloc[i, j] = ""
    df.loc[i, "Label"] = 1
    errors[i, j] = MISSING
    return df, errors

def _f_cell_swap(df, df_original, errors, index):
    direction = ["left", "right"]
    min_j = 0
    max_j = df.shape[1] - 2 # ignore label column
    i, j = index
    possible_directions = direction.copy()
    
    if j == min_j or errors[i, j - 1] != 0:
        possible_directions.remove("left")
    if j == max_j or errors[i, j + 1] != 0:
        possible_directions.remove("right")
    
    if len(possible_directions) == 0:
        return df, errors
    
    swap_direction = random.choice(possible_directions)
    old_value = df.iloc[i, j]
    
    if swap_direction == "left":
        df.iloc[i, j] = df.iloc[i, j - 1]
        df.iloc[i, j - 1] = old_value
        df.loc[i, "Label"] = 1
        errors[i, j] = SWAP
        errors[i, j - 1] = SWAP
    elif swap_direction == "right":
        df.iloc[i, j] = df.iloc[i, j + 1]
        df.iloc[i, j + 1] = old_value
        df.loc[i, "Label"] = 1
        errors[i, j] = SWAP
        errors[i, j + 1] = SWAP
        
    return df, errors

def _f_spelling(df, df_original, errors, index):
    i, j = index
    if errors[i, j] != 0:
        return df, errors
    
    spelling_errors = ["misspelling", "swapped", "missing"]
    choosen_error = np.random.choice(spelling_errors, p=[1/2, 1/6, 1/3])
    
    word = df.iloc[i, j]
    if len(word) < 2:
        return df, errors
    
    if choosen_error == "misspelling":
        char_index = random.randint(0, len(word) - 1)
        word = word[:char_index] + random.choice(string.ascii_letters) + word[char_index + 1:]
    elif choosen_error == "swapped":
        index = random.randint(0, len(word) - 2)
        word = word[:index] + word[index+1] + word[index] + word[index + 2:]
    elif choosen_error == "missing":
        index = random.randint(0, len(word) - 1)
        word = word[:index] + word[index + 1:]
    
    df.iloc[i, j] = word
    df.loc[i, "Label"] = 1
    errors[i, j] = SPELLING
    
    return df, errors

def _f_functional_dependency(df, df_original, errors, index):
    i, j = index
    if errors[i, j] != 0:
        return df, errors
    column_name = df_original.columns[j]
    unique_values = df_original[column_name].unique()
    if len(unique_values) > 1:
        new_value = np.random.choice(unique_values)
        while new_value == df.at[i, column_name]:
            new_value = np.random.choice(unique_values)
        
        # print(f"Changing {df.iloc[i, j]} to {new_value} in column {column_name} in row {i}")
        df.iloc[i, j] = new_value   
        df.loc[i, "Label"] = 1
        errors[i, j] = FUNCTIONAL_DEPENDENCY

    return df, errors

def _f_dummy(df, errors, index):
    return df, errors


ORIGINAL = 0
SIMPLE = 1
BART = 2
DETECTION = 3


def insert_errors_bart(path_ori, path, work_dir):
    print(path_ori, path)
    random.seed(1)
    
    # connect to database
    conn = sqlite3.connect(os.path.join(work_dir, "database.db"))
    
    sql = f"select * from \"{path_ori}_changes\""
    df_changes = pd.read_sql(sql, conn)
    
    sql = f"select * from \"{path_ori}_dirty\""
    df_dirty = pd.read_sql(sql, conn)
    
    errors = np.zeros(shape=df_dirty.shape, dtype=int)
    for _, row in df_changes.iterrows():
        i = int(row["index"]) - 1
        j = df_dirty.columns.get_loc(row["column"])
        errors[i, j] = 1
    
    df_dirty.to_sql(path, conn, if_exists="replace", index=False)
    df_dirty.to_sql(f"{path_ori}_err_ori", conn, if_exists="replace", index=False)

    print(f"Number of Cells: {df_dirty.shape[0] * (df_dirty.shape[1] - 1)}, Number of Generated Errors: {np.count_nonzero(errors)}")
    return errors

