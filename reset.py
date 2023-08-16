import sqlite3
import os

def reset(path_ori,path, work_dir):
    print("Resetting the data set", path)
    conn = sqlite3.connect(os.path.join(work_dir, "database.db")) 
    cursor = conn.cursor()
    sql = "DELETE FROM \"" + path + "\" "
    cursor.execute(sql)
    conn.commit()
    sql1 = "select * from \"" + path_ori + "\" " 
    cursor.execute(sql1)
    data1 = cursor.fetchall()
    t2 = len(data1[0])
    for row in data1:
        for num in range(t2):
            if num == 0:
                sql_before = "'%s'"
            else:
                sql_before = sql_before + ",'%s'"
        va = []
        for num in range(t2):
            va.append(row[num])
        sql_after = tuple(va)
        sql3 = "insert into \"" + path + "\" values(" + sql_before + ")"
        sql4=sql3% (sql_after)
        cursor.execute(sql4)
        conn.commit()   #Reset
    cursor.close()
    conn.close()
    print("Reset complete")
