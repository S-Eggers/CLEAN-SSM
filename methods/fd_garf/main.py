import sqlite3
import pandas as pd
import numpy as np
from typing import List, Dict
from itertools import combinations
from tqdm import tqdm
import sys
import math 


MAX_LHS = 3
SUBSETS = 50

def split_dataframe(df: pd.DataFrame, n: int):
    subsets = []
    chunk_size = len(df) // n  
    remainder = len(df) % n 

    for i in range(n):
        if i < remainder:
            subset = df[i * (chunk_size + 1) : (i + 1) * (chunk_size + 1)]
        else:
            subset = df[i * chunk_size + remainder : (i + 1) * chunk_size + remainder]
        subsets.append(subset)

    return subsets

def find_functional_dependencies(df: pd.DataFrame) -> Dict[str, List[str]]:
    columns = list(df.columns)

    _dict = {}
    for i in range(1, MAX_LHS + 1):
        for subset in combinations(columns, i):
            remaining = [col for col in columns if col not in subset]
            
            for col in remaining:
                if df.groupby(list(subset))[col].nunique().max() == 1:
                    if col not in _dict:
                        _dict[col] = [subset]
                    else:
                        _dict[col].append(subset)
    return _dict

def addtwodimdict(thedict, key_a, key_b, val):
  if key_a in thedict:
    thedict[key_a].update({key_b: val})
  else:
    thedict.update({key_a:{key_b: val}})

def main(flag=2, order=1, remove_amount_of_error_tuples=0.0, dataset="Hospital"):   
    path = f"{dataset}_copy"
    path_ori = path.strip('_copy')
    print(path_ori)
    connection = sqlite3.connect(f"database.db")
    cursor = connection.cursor()
    sql = f"SELECT * FROM {path}"
    df = pd.read_sql(sql, con=connection)
    df = df.drop(columns=["Label"])
    
    print(f"Splitted dataframe into {SUBSETS} subsets")
    df_subsets = split_dataframe(df, SUBSETS)
    
    found_fds = {}
    tqdm.write(f"Finding functional dependencies for {SUBSETS} subsets")
    progress_bar = tqdm(df_subsets)
    for subset in progress_bar:
        fds = find_functional_dependencies(subset)
        for RHS in fds.keys():
            if RHS not in found_fds:
                found_fds[RHS] = set(fds[RHS])
            else:
                found_fds[RHS] = set([*found_fds[RHS], *fds[RHS]])
    
    rules_dict = {}
    
    tqdm.write(f"Finding rules for {len(found_fds)} columns")
    progress_bar = tqdm(found_fds.items(), desc="FDs         ", position=0)
    for rhs, lhs_set in progress_bar:
        progress_bar_inner = tqdm(lhs_set, desc="Combinations", position=1, leave=False)
        for lhs in progress_bar_inner:
            lhs_values = df[list(lhs)].drop_duplicates().to_records(index=False).tolist()
            rhs_values = df[rhs].unique()
            
            lhs_values = [tuple(val) for val in lhs_values]
            for lhs_value in lhs_values:
                for rhs_value in rhs_values:
                    key = str(list(lhs_value))
                    if key not in rules_dict:
                        # Initialize the dict with 'reason' as a dict and 'result' as rhs_value
                        rules_dict[key] = {'reason': dict(zip(list(lhs), lhs_value)), 'result': dict(zip([rhs], [rhs_value]))}
                    else:
                        # if key already exists, update the 'reason' and 'result'
                        rules_dict[key]['reason'].update(dict(zip(list(lhs), lhs_value)))
                        rules_dict[key]['result'] = dict(zip([rhs], [rhs_value]))
    
    prev_len = len(list(rules_dict.keys()))
    print(prev_len)
    
    f = open('data/save/rules_read.txt', 'w')
    # print(str(rules_final))
    for item in rules_dict.items():
        f.write(str(item))
        f.write('\r\n')
    f.write(str(rules_dict))
    f.close()
    
    num = 0
    for rulename, ruleinfo in list(rules_dict.items()):
        num += 1
        # print("Filter page", num, "Rules and corresponding data")
        # print("ruleinfo:", ruleinfo)
    
        left = list(ruleinfo['reason'].keys())
        # print(left)
        word = list(ruleinfo['reason'].values())
        # print(word)
        k = list(ruleinfo['result'].keys())
        right = k[0]
        v = list(ruleinfo['result'].values())
        result = v[0]

        sqlex = left[0] + "\"='" + word[0] + "'"
        i = 1
        while (i < len(left)):
            sqlex = sqlex + " and \"" + left[i] + "\"='" + word[i] + "'"
            i += 1

        sql1 = "select \"" + right + "\" from \"" + path + "\" where \"" + sqlex
        # print(sql1)         #select "MINIT" from "UIS_copy" where "CUID"='9078' and "RUID"='15896' and "SSN"='463210223' and "FNAME"='Monken'
        cursor.execute(sql1)  # "City","State" ,where rownum<=10
        rows = cursor.fetchall()
        num1=len(rows)
        if num1<3:
            # print("The data that satisfy the rule are",num1,"Article, the source is presumed to be wrong data, no sense of repair, delete the rule "",rules_final[str(rulename)])
            del rules_dict[str(rulename)]
            continue
        else:
            t_rule=1
            for row in rows:
                if (str(row[-1]) == str(result)):  # In this case, the rule matches the data, and the confidence of the rule increases
                    t_rule = t_rule + 1
                    print("-->", t_rule, end='')
                else:  # In this case, the rule is contrary to the data, and the confidence of the rule is reduced
                    t_rule = t_rule - 2
                    print("-->", t_rule, end='')
                    flag = 0  # Mark the rule as conflicting with the data
            rules_dict[str(rulename)].update({'confidence': t_rule})  # Rule confidence initialization
    
    
    f = open('data/save/rules_final.txt', 'w')
    # print(str(rules_final))
    f.write(str(rules_dict))
    f.close()
    
    l2 = len(rules_dict)
    print("\nRule filtering is complete and the remaining number of", )
    print(str(l2))
    print(f"Initial length was {prev_len}")
    
    with open('data/save/log_filter.txt', 'w') as f:
        f.write("The number of original rules is")
        f.write(str(prev_len))
        f.write("After rule filtering, the remaining number of")
        f.write(str(l2))
        f.write("__________")
    f.close()
    
    att_name = []
    for item in df.columns:
        # print(item)
        att_name.append(item)
    # print(att_name)
    _dict = {}
    for i in range(len(df.columns)):
        _dict[i] = att_name[i]
    # print(dict)
    # f = open('att_name.txt', 'w')
    print(_dict)
    f = open('data/save/att_name.txt', 'w')
    f.write(str(_dict))
    f.close()

    repair(path)

def repair(path):

    print("Performed a REPAIR")
    f = open('data/save/att_name.txt', 'r')
    label2att = eval(f.read())
    f.close()
    # print(label2att)
    att2label = {v: k for k, v in label2att.items()}  # Dictionary Reverse Transfer
    
    print(label2att, att2label)
    
    f = open('data/save/rules_final.txt', 'r')
    rule = eval(f.read())
    f.close()
    # print(self.rule)
    num = 0
    error_rule=0
    error_data=0
    # conn = cx_Oracle.connect('system', 'Pjfpjf11', '127.0.0.1:1521/orcl')  # Connecting to the database
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    for rulename, ruleinfo in list(rule.items()):
        num += 1
        print("Fix the first", num, "Rules and corresponding data")
        # print("rulename:" + rulename)
        print("ruleinfo:", ruleinfo)

        left = list(ruleinfo['reason'].keys())
        # print(left)
        word = list(ruleinfo['reason'].values())
        # print(word)
        k = list(ruleinfo['result'].keys())
        right = k[0]
        v = list(ruleinfo['result'].values())
        result = v[0]

        LHS = []
        LHS.append(att2label[left[0]])
        RHS=att2label[right]
        sqlex = left[0] + "\"='" + word[0] + "'"
        i = 1
        # AV_p = "\""+left[0]+"\""+","     # Convert the data in left to string form
        while (i < len(left)):
            sqlex = sqlex + " and \"" + left[i] + "\"='" + word[i] + "'"
            # AV_p = AV_p +"\""+ left[i]+"\""+","
            LHS.append(att2label[left[i]])
            i += 1
            # print(sqlex)
        # print("AV_p Index：",LHS,"AV_c Index：",RHS)
        # AV_c = "\"" + right + "\""
        # print("AV_p:",AV_p,"AV_c:",AV_c)

        sql1 = "select * from \"" + path + "\" where \"" + sqlex
        #sql1 = "select " +AV_p+ AV_c + " from \"" + path + "\" where \"" + sqlex
        # sql1 = "select \"" + right + "\" from \"" + path + "\" where \"" + sqlex
        print(sql1)
        cursor.execute(sql1)  # "City","State" ,where rownum<=10
        rows = cursor.fetchall()
        # print("rows:")
        flag_trust = detect(rows,result,rulename,LHS,RHS,att2label,label2att)

        if (flag_trust == 3):       # 3 means the rule is correct, and there is no conflict, proceed directly to the next rule
            continue

        if (flag_trust == 0):
            error_rule += 1

        s2 = 0
        while (flag_trust == 0 and s2 < 3):
            result = v[0]
            print("Repair rules left")
            s2 += 1
            min=10
            flag = int(att2label[left[0]])
            # print(flag)
            if (min > flag):
                min = flag  # The index corresponding to the leftmost part of the current REASON section
            # print(min)
            if(min==0):
                print("No fixes can be added to the left side of the rule, delete the rule")
                del rule[str(rulename)]
                break
            left_new=label2att[min-1]
            print("Add",left_new,"Information")
            sqladd= "select \"" + left_new + "\" from \"" + path + "\" where \"" + sqlex+"and \"" + right + "\"='" + result + "'"
            print("sqladd:",sqladd)
            cursor.execute(sqladd)
            rows_left = cursor.fetchall()

            if(rows_left ==[]):
                # print("There is no condition modified on the left side of the rule, delete the rule")
                del rule[str(rulename)]
                break
            # print(rows_left)
            addtwodimdict(rule[str(rulename)], 'reason', str(left_new), str(rows_left[0][0]))
            for n in range(len(word)):
                del rule[str(rulename)]['reason'][left[n]]
                addtwodimdict(rule[str(rulename)], 'reason', str(left[n]), str(word[n]))

            left = list(ruleinfo['reason'].keys())
            word = list(ruleinfo['reason'].values())
            # print(word)
            # print(self.rule[str(rulename)])
            sqlex = left[0] + "\"='" + word[0] + "'"
            i = 1
            while (i < len(left)):
                sqlex = sqlex + " and \"" + left[i] + "\"='" + word[i] + "'"
                i += 1
            sql1 = "select * from \"" + path + "\" where \"" + sqlex
            # print(sql1)
            cursor.execute(sql1)  # "City","State" ,where rownum<=10
            rows = cursor.fetchall()

            if (len(rows)<3):
                continue

        if (flag_trust == 0):
            print("Rule is not available to fix, delete the rule")

        if (flag_trust == 1):
            t0=1
            for row in rows:
                if (str(row[RHS]) == str(result)):
                    continue
                else:
                    AV_p = []
                    t_tp = 999  # The confidence of the current tuple is calculated as the minimum value of the confidence of different AV_i in a tuple, and a large value is set first to avoid the interference of the initial value.
                    t_tc = t0
                    flag_p=0     # Used to record the position of the attribute corresponding to the lowest confidence level in AV_p
                    rule_p_name=[] # Record the rule with the highest confidence that can repair the attribute with the lowest confidence in the above AV_p
                    print("The tuples that match the current rule are: ", row)
                    for i in LHS:  # Calculate the minimum value of confidence in different AV_i in a tuple
                        AV_p.append(row[i])
                        t_AV_i = t0
                        attribute_p = label2att[i]
                        rulename_p_max = []
                        t_rmax = -999       # The maximum confidence level of the rules that correct AV_i in the following iterative dictionary, initially set to a minimal value
                        for rulename_p, ruleinfo_p in list(rule.items()):  # Traversing the dictionary
                            if rulename == rulename_p:
                                continue
                            if t_AV_i > 100 or t_AV_i < -100:
                                break
                            v = list(ruleinfo_p['result'].values())
                            left = list(ruleinfo_p['reason'].keys())
                            word = list(ruleinfo_p['reason'].values())
                            k = list(ruleinfo_p['result'].keys())
                            t_r = ruleinfo_p['confidence']
                            if t_r < 0:
                                continue
                            right = k[0]
                            if attribute_p == right:
                                flag_equal = 0  # Can the rule determine the token of row[i]
                                for k in range(len(left)):
                                    if row[att2label[left[k]]] == word[k]:  # If the tuple where row[i] is located satisfies all AV_p of a rule, mark it as 1
                                        flag_equal = 1
                                    else:
                                        flag_equal = 0
                                        break
                                if flag_equal == 1:  # If the row[i] in the tuple can be determined by other rules, check whether it satisfies the rule
                                    # print(row, "Medium")
                                    # print(right, "Can be determined by other rules：", ruleinfo)
                                    result2 = v[0]
                                    if t_rmax < t_r:  # 记Record the maximum rule confidence in these rules
                                        t_rmax = t_rmax
                                        rulename_p_max = rulename_p  # Record the identification of the most trusted rule in the dictionary
                                    if str(row[i]) == str(result2):
                                        t_AV_i = t_AV_i + t_r
                                    else:
                                        t_AV_i = t_AV_i - t_r
                                        print("In AV_p", str(row[i]), "with", str(result2), "does not match, the corresponding rule is", ruleinfo_p,
                                                "Its confidence level is", t_r)

                        if t_tp > t_AV_i:
                            t_tp = t_AV_i
                            flag_p=i                    # Record the index of AV_i with the lowest confidence
                            rule_p_name=rulename_p_max  # Record the name of the rule that corrects this AV_i with the highest confidence

                    for rulename_c, ruleinfo_c in list(rule.items()):  # Iterate through the dictionary, calculate t_c
                        if rulename == rulename_c:
                            continue
                        v = list(ruleinfo_c['result'].values())
                        left = list(ruleinfo_c['reason'].keys())
                        word = list(ruleinfo_c['reason'].values())
                        k = list(ruleinfo_c['result'].keys())
                        t_r = ruleinfo_c['confidence']
                        if t_r < 0:
                            continue
                        right = k[0]
                        attribute_c = label2att[RHS]
                        if attribute_c == right:
                            flag_equal = 0  # Can the rule determine the token of row[i]
                            for k in range(len(left)):
                                if row[att2label[left[k]]] == word[k]:  # If the tuple in which AV_c is located satisfies all AV_p of a rule, mark it as 1
                                    flag_equal = 1
                                else:
                                    flag_equal = 0
                                    break
                            if flag_equal == 1:  # If the AV_c in the tuple can be determined by other rules, check whether it satisfies the rules
                                result2 = v[0]
                                if str(row[RHS]) == str(result2):
                                    t_tc = t_tc + t_r
                                else:
                                    t_tc = t_tc - t_r
                                    print("In AV_c", str(row[RHS]), "with", str(result2), "does not match, the corresponding rule is", ruleinfo_c, "Its confidence level is",
                                            t_r)

                    if t_tp == 999:  # means that all cells in it cannot be determined by other rules, reset its value to t0
                        t_tp = t0
                    if t_tc < t_tp or t_tc == t_tp:
                        print("In this case, the data result is considered partially wrong, and the data is repaired according to the rule, the current rule is",rulename,"-->",result,"t_p is",t_tp,"t_c is",t_tc)
                        for x in range(len(row)-1):  # t2
                            if x == 0:
                                sql_info = f"\"{label2att[x]}\"='{row[x]}'"
                            else:
                                sql_info += f" and \"{label2att[x]}\"='{row[x]}'"
                        sql_update = "update \"" + path + "\" set \"Label\"='2' , \"" + label2att[RHS] + "\"='" + result + "' where  " + sql_info + ""
                        print("Original: ", sql_info)
                        print("Update Information: ", sql_update)
                        cursor.execute(sql_update)
                        conn.commit()
                    else:
                        print(rule_p_name)
                        if rule_p_name==[]:
                            print("There may be errors")
                            continue
                        rname=rule[str(rule_p_name)]
                        v2 = list(rname['result'].values())
                        result2 = v2[0]
                        print("At this point, the data inference is considered partially wrong, and the data is "
                                "repaired according to the rule, the current rule is", rule_p_name, "-->", result2,
                                "t_p is", t_tp, "t_c is", t_tc)
                        for x in range(len(row)-1):  # t2
                            if x == 0:
                                sql_info = "\"" + label2att[x] + "\"='" + row[x] + "'"
                            else:
                                sql_info = sql_info + " and \"" + label2att[x] + "\"='" + row[x] + "'"
                        sql_update = "update \"" + path + "\" set \"Label\"='2' , \"" + label2att[flag_p] + "\"='" + result2 + "' where  " + sql_info + ""
                        print("Original: ", sql_info)
                        print("Update Information: ", sql_update)
                        cursor.execute(sql_update)
                        conn.commit()
                        continue

    cursor.close()
    conn.close()

    print("Repair completed")
    print("Save repair rules")
    print("Rule dictionary size", len(rule))
    # print(str(self.rule))
    f = open('data/save/rules_final.txt', 'w')
    f.write(str(rule))
    f.close()
    with open('data/save/log.txt', 'a') as f:
        f.write("Total number of rules used this time")
        f.write(str(num))
        f.write("Number of rule errors")
        f.write(str(error_rule))
        f.write("Number of data errors")
        f.write(str(error_data))
        f.write("__________")
        f.close()


rule = dict()

def detect(rows,result,rulename,LHS,RHS,att2label,label2att):
    dert = 0
    t0=1
    t_rule=t0
    t_tuple=t0
    t_max=t_tuple   # The maximum value of confidence in different tuples satisfying the RULE condition
    flag=1         # Flag whether the rule conflicts with the data
    flag_trust = 0  # 0 for believe data, 1 for believe rule
    for row in rows:
        if (str(row[RHS]) == str(result)):
            continue
        else:
            dert += 1
            flag = 0   # Mark the rule as conflicting with the data
    if (flag==1):           # If the rule does not conflict with the data, a great confidence level is given directly
        t_rule=t_rule+100
        flag_trust = 3  # 3 means the rule is correct and there is no conflict
        return flag_trust
    else:                   # The rule conflicts with the data, then the confidence of each tuple is calculated to adjust t_rule
        print("The rule conflicts with the data")
        print("Estimated changes for this restoration", dert)
        error_row=[]
        rule_other=[]
        t_rule=t0
        for row in rows:    # Each tuple that satisfies the rule condition
            AV_p=[]
            t_tp = 999
            t_tc = t0
            for i in LHS:
                AV_p.append(row[i])
                t_AV_i = t0
                # rulename_p_max = []
                # t_rmax = 0
                attribute_p=label2att[i]
                for rulename_p, ruleinfo_p in list(rule.items()):
                    if rulename == rulename_p:
                        continue
                    if t_AV_i>100 or t_AV_i<-100:
                        break
                    v = list(ruleinfo_p['result'].values())
                    left = list(ruleinfo_p['reason'].keys())
                    word = list(ruleinfo_p['reason'].values())
                    k = list(ruleinfo_p['result'].keys())
                    t_r = ruleinfo_p['confidence']
                    if t_r<0:
                        continue
                    right = k[0]
                    if attribute_p == right:
                        flag_equal = 0 
                        for k in range(len(left)):
                            if row[att2label[left[k]]] == word[k]:
                                flag_equal = 1
                            else:
                                flag_equal = 0
                                break
                        if flag_equal == 1:
                            result2 = v[0]
                            if str(row[i]) == str(result2):
                                t_AV_i = t_AV_i + t_r
                            else:
                                t_AV_i = t_AV_i - t_r
                                print("The tuples that match the current rule are: ", row)
                                print("In AV_p",str(row[i]), "with", str(result2), "does not match, the corresponding rule is", ruleinfo_p, "Its confidence level is", t_r)

                if t_tp > t_AV_i:
                    t_tp = t_AV_i

            for rulename_c, ruleinfo_c in list(rule.items()):  # Iterate through the dictionary, calculate t_c
                if rulename==rulename_c:
                    continue
                v = list(ruleinfo_c['result'].values())
                left = list(ruleinfo_c['reason'].keys())
                word = list(ruleinfo_c['reason'].values())
                k = list(ruleinfo_c['result'].keys())
                t_r = ruleinfo_c['confidence']
                if t_r < 0:
                    continue
                right = k[0]
                attribute_c = label2att[RHS]
                if attribute_c == right:
                    flag_equal = 0  
                    for k in range(len(left)):
                        if row[att2label[left[k]]] == word[k]: 
                            flag_equal = 1
                        else:
                            flag_equal = 0
                            break
                    if flag_equal == 1:  
                        result2 = v[0]
                        if str(row[RHS]) == str(result2):
                            t_tc = t_tc + t_r
                        else:
                            t_tc = t_tc - t_r
                            print("The tuples that match the current rule are: ", row)
                            print("In AV_c",str(row[RHS]), "with", str(result2), "does not match, the corresponding rule is", ruleinfo_c, "Its confidence level is", t_r)

            if t_tp==999:        # means that all cells in it cannot be determined by other rules, reset its value to t0
                t_tp=t0
            if t_tc < t_tp:
                t_tuple = t_tc
            else:
                t_tuple = t_tp

            if (str(row[RHS]) == str(result)):  # The tuple data is consistent with the rule, and the confidence level increases
                if t_tuple>0:
                    t_rule = t_rule + math.ceil(math.log(1+t_tuple))
                else:
                    t_rule = t_rule + t_tuple
                t_max = t_max
                print("-->", t_rule, end='')
            else:  # If the tuple data violates the rule, calculate the confidence of the corresponding tuple
                if t_tuple>0:
                    t_rule = t_rule - 2*t_tuple
                else:
                    t_rule = t_rule + math.ceil(math.log(1+abs(t_tuple)))
                    print("-->", t_rule, end='')

                if (t_rule < -100):
                    flag_trust = 0
                    return flag_trust  # In this case, the confidence level of the rule is too small, so the loop is directly jumped and marked as error.

            if t_max < t_tuple:
                t_max = t_tuple

        print("The final rule confidence level is",t_rule,"The tuple with which it conflicts has the highest confidence level of",t_max)
    if (t_rule > t_max ):
            flag_trust = 1  # At this point the rule is considered correct and the data is modified
    elif (t_rule < t_max ):
            flag_trust = 0
            # print("The final rule confidence level is", t_rule, "The tuple with which it conflicts has the highest confidence level of", t_max)
            return flag_trust  # At this point the data is considered correct and the rule is modified
    rule[str(rulename)].update({'confidence': t_rule}) # Rule confidence initialization can be considered to be taken out separately
    print()
    return flag_trust