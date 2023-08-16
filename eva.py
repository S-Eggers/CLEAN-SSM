
import sqlite3
import os
import pandas as pd
import numpy as np

def evaluate_original(path_ori, path, base_dir):
    conn = sqlite3.connect(os.path.join(base_dir, "database.db"))
    cursor = conn.cursor()

    sql1 = "select * from \"" + path + "\" where \"Label\"='2' or \"Label\"='3'"    #where rownum < 3  #order by "Provider ID" desc
    cursor.execute(sql1)
    data1 = cursor.fetchall()
    des = cursor.description
    att = []
    for item in des:
        att.append(item[0])
    print(att)
    t2 = len(data1[0]) - 1  # Length of data per row, -1 is to become index position

    correct=0
    error=0
    len_update=len(data1)
    for row in data1:
        # print(row)
        for i in range(t2):  # t2
            if i == 0:
                sql_info = "\"" + att[i] + "\"='" + row[i] + "'"
            else:
                sql_info = sql_info + " and \"" + att[i] + "\"='" + row[i] + "'"
        sql_info_label = sql_info + " and \"Label\"='2'"
        sql2 = "select * from \"" + path_ori + "\" where " + sql_info + ""  # where rownum < 3  #order by "Provider ID" desc
        cursor.execute(sql2)
        data_ori = cursor.fetchall()
        if data_ori==[]:
            sql_update = "update \"" + path + "\" set \"Label\"='3'  where  " + sql_info + ""
            cursor.execute(sql_update)
            conn.commit()
            error += 1
            continue
        x1=data_ori[0]
        x1=list(x1)[:-1]

        sql3 = "select * from \"" + path + "\" where " + sql_info_label + ""  # where rownum < 3  #order by "Provider ID" desc
        cursor.execute(sql3)
        data_new = cursor.fetchall()
        x2 = data_new[0]
        x2 = list(x2)[:-1]
        if x1==x2:
            correct+=1
        else:
            print("Fix the bug, fix for", x1,"         The actual correct data is",x2)
            error+=1
    try:
        precision=1-error/(error+correct)
    except ZeroDivisionError:
        precision=0

    sql4 = "select * from \"" + path + "\" where \"Label\"='2' or \"Label\"='1' or  \"Label\"='3'"
    # print(sql4)
    cursor.execute(sql4)
    data2 = cursor.fetchall()
    
    try:
        recall = correct / len(data2)
    except ZeroDivisionError:
        recall = 0

    print(len(data2),"Fixed",len_update,"of which the correct number is",correct,", The number of errors is",error)
    print("precision:", precision)
    print("recall:", recall)
    cursor.close()
    conn.close()

    try:
        f_measure = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        f_measure = 0
    
    print("f1:", f_measure)

    with open(os.path.join(base_dir, "data", "save", "log_evaluation.txt"), 'a') as f:
        f.write("")
        f.write(str(len(data2)))
        f.write("Fixed")
        f.write(str(len_update))
        f.write("of which the correct number is")
        f.write(str(correct))
        f.write(", The number of errors is")
        f.write(str(error))
        f.write("precision:")
        f.write(str(precision))
        f.write("           recall:")
        f.write(str(recall))
        f.write("           f_measure:")
        f.write(str(f_measure))
        f.write("\n")
        f.close()
    
    return precision, recall, f_measure

def evaluate(path_ori, path, base_dir, errors, database_name: str = "database.db", store_report: bool = True):
    print(os.path.join(base_dir, database_name))
    print(path_ori, path)
    conn = sqlite3.connect(os.path.join(base_dir, database_name))
    df = pd.read_sql_query(f"SELECT * FROM \"{path}\"", conn)
    df_err_ori = pd.read_sql_query(f"SELECT * FROM \"{path_ori}_err_ori\"", conn)
    df_ori = pd.read_sql_query(f"SELECT * FROM \"{path_ori}\"", conn)
    conn.close()
    
    print(df.Label.value_counts())
    print(df_ori.Label.value_counts())
    
    # remove label column from evaluation
    df = df.drop(columns=["Label"])
    df_ori = df_ori.drop(columns=["Label"])
    df_err_ori = df_err_ori.drop(columns=["Label"])
    errors = errors[:, :-1]
    
    """
    For debugging purposes, save the false positives to a file               
    df_1 = df
    df_ori_1 = df_ori
    fp_mask = (df_ori != df) & (errors == 0) & (df_err_ori != df)
    df_1 = df_1.where(fp_mask, "")
    df_1.to_csv("fp_df.csv", index=False)
    df_ori_1 = df_ori_1.where(fp_mask, "")
    df_ori_1.to_csv("fp_df_ori.csv", index=False)
    """
    
    df = df.values
    df_ori = df_ori.values
    df_err_ori = df_err_ori.values
    
    TP = np.sum((df_ori == df) & (errors >  0) & (df_err_ori != df))
    FP = np.sum((df_ori != df) & (errors == 0) & (df_err_ori != df))
    TN = np.sum((df_ori == df) & (errors == 0) & (df_err_ori == df))
    FN = np.sum((df_ori != df) & (errors >  0) & (df_err_ori == df))

    precision = TP / (TP + FP) if TP + FP != 0 else -1
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1 = 2 * ((precision * recall) / (precision + recall)) if precision + recall != 0 else 0
    
    print(f"Cells {df.shape[0] * df.shape[1]}, fixed {TP + FP} of which the correct number is {TP}, the number of errors is {FP}")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    
    errors_flat = np.ravel(errors)
    error_stats = {i: np.sum((df_ori == df) & (errors == i)) for i in range(1, errors_flat.max() + 1)}
    count_errors = np.bincount(errors_flat)
    for error, count in error_stats.items():
        print(f"Found error {error}: {count}/{count_errors[error]}")
    
    if store_report:
        with open(os.path.join(base_dir, "data", "save", "log_evaluation.txt"), 'a') as f:
            f.write("")
            f.write(str(df.shape[0]))
            f.write("Fixed")
            f.write(str(TP + FP))
            f.write("of which the correct number is")
            f.write(str(TP))
            f.write(", The number of errors is")
            f.write(str(FP))
            f.write("precision:")
            f.write(str(precision))
            f.write("           recall:")
            f.write(str(recall))
            f.write("           f_measure:")
            f.write(str(f1))
            f.write("\n")
            f.close()

    return precision, recall, f1


def evaluate_detection(path_ori, path, base_dir, errors, database_name: str = "database.db", store_report: bool = True):
    print(os.path.join(base_dir, database_name))
    print(path_ori, path)
    conn = sqlite3.connect(os.path.join(base_dir, database_name))
    df = pd.read_sql_query(f"SELECT * FROM \"{path}\"", conn)
    df_err_ori = pd.read_sql_query(f"SELECT * FROM \"{path_ori}_err_ori\"", conn)
    df_ori = pd.read_sql_query(f"SELECT * FROM \"{path_ori}\"", conn)
    conn.close()
    
    print(df.Label.value_counts())
    print(df_ori.Label.value_counts())
    
    # remove label column from evaluation
    df = df.drop(columns=["Label"])
    df_ori = df_ori.drop(columns=["Label"])
    df_err_ori = df_err_ori.drop(columns=["Label"])
    errors = errors[:, :-1]
    
    """
    For debugging purposes, save the false positives to a file               
    df_1 = df
    df_ori_1 = df_ori
    fp_mask = (df_ori != df) & (errors == 0) & (df_err_ori != df)
    df_1 = df_1.where(fp_mask, "")
    df_1.to_csv("fp_df.csv", index=False)
    df_ori_1 = df_ori_1.where(fp_mask, "")
    df_ori_1.to_csv("fp_df_ori.csv", index=False)
    """
    
    df = df.values
    df_ori = df_ori.values
    df_err_ori = df_err_ori.values
    
    TP = np.sum((df_ori == df) & (errors >  0))
    FP = np.sum((df_ori != df) & (errors == 0))
    FN = np.sum((df_ori != df) & (errors >  0) & (df_err_ori == df))

    precision = TP / (TP + FP) if TP + FP != 0 else -1
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1 = 2 * ((precision * recall) / (precision + recall)) if precision + recall != 0 else 0
    
    print(f"Cells {df.shape[0] * df.shape[1]}, fixed {TP + FP} of which the correct number is {TP}, the number of errors is {FP}")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    
    errors_flat = np.ravel(errors)
    error_stats = {i: np.sum((df_ori == df) & (errors == i)) for i in range(1, errors_flat.max() + 1)}
    count_errors = np.bincount(errors_flat)
    for error, count in error_stats.items():
        print(f"Found error {error}: {count}/{count_errors[error]}")
    
    if store_report:
        with open(os.path.join(base_dir, "data", "save", "log_evaluation.txt"), 'a') as f:
            f.write("")
            f.write(str(df.shape[0]))
            f.write("Fixed")
            f.write(str(TP + FP))
            f.write("of which the correct number is")
            f.write(str(TP))
            f.write(", The number of errors is")
            f.write(str(FP))
            f.write("precision:")
            f.write(str(precision))
            f.write("           recall:")
            f.write(str(recall))
            f.write("           f_measure:")
            f.write(str(f1))
            f.write("\n")
            f.close()

    return precision, recall, f1