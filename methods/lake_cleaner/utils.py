import sys
import math
import time
import tqdm
import random
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict
from typing import Dict, Set, Tuple, List, Callable


def calculate_rule_ratio(dict1, dict2):
    rules1 = extract_rules(dict1)
    rules2 = extract_rules(dict2)
    same_rules_count = len(rules1.intersection(rules2))    
    total_rules_count = len(rules1.union(rules2))

    ratio = round(same_rules_count / total_rules_count, 3) if total_rules_count != 0 else 0
    return ratio

def sort_dict_by_key(d: Dict) -> Dict:
    return {k: d[k] for k in sorted(d)}

def convert_to_tuple(d: Dict) -> Tuple:
    return tuple((k, v) for k, v in d.items())

def extract_rules(dictionary):
    rules = set()
    for value in dictionary.values():
        reason = sort_dict_by_key(value['reason'])
        reason = convert_to_tuple(reason)
        result = sort_dict_by_key(value['result'])
        result = convert_to_tuple(result)
        rules.add((reason, result))
    return rules

def combine(rules_forward: Dict, rules_backward: Dict) -> Dict[str, Dict[str, Dict[str, str]]]:
    results = {**rules_forward}
    for key, value in rules_backward.items():
        if key not in results or results[key]["confidence"] < value["confidence"]:
            results[key] = value
    
    return results


def get_rule_vocab(rules: Dict[str, Dict[str, Dict[str, str]]]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    lhs_vocab = dict()
    rhs_vocab = dict()
    for value in rules.values():
        reason = value["reason"]
        result = value["result"]
        
        for key, value in reason.items():
            value = str(value).lower()
            if key not in lhs_vocab:
                lhs_vocab[key] = set()
            lhs_vocab[key].add(value)
        
        for key, value in result.items():
            value = str(value).lower()
            if key not in rhs_vocab:
                rhs_vocab[key] = set()
            rhs_vocab[key].add(value)

    return lhs_vocab, rhs_vocab

def get_dataframe_vocab(df: pd.DataFrame) -> Dict[str, Set[str]]:
    vocab = dict()
    
    for column in df.columns:
        vocab[column] = set(df[column].str.lower().unique())
    
    return vocab

def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

def vocab_overlap(set1: Set[str], set2: Set[str]) -> float:
    intersection = set1.intersection(set2)
    return len(intersection) / len(set1)

def assign_columns(pairs: List[Tuple[Tuple[str, str], float]]) -> Dict[str, str]:
    matches = defaultdict(list)
    
    for pair, score in pairs:
        column1, column2 = pair
        matches[column1].append((column2, score))

    for column in matches:
        matches[column].sort(key=lambda x: x[1], reverse=True)

    assigned1, assigned2 = set(), set()
    final_matches = {}

    for column in sorted(matches, key=lambda x: matches[x][0][1], reverse=True):
        if column in assigned1:
            continue

        for match, _ in matches[column]:
            if match not in assigned2:
                final_matches[column] = match
                assigned1.add(column)
                assigned2.add(match)
                break

    return final_matches

def avg_score(similarities: List[Tuple[Tuple[str, str], float]], column_assignment: Dict[str, str]) -> float:
    return sum(
        [
            score for columns, score in similarities 
            if columns[0] in column_assignment and column_assignment[columns[0]] == columns[1]
        ]
    ) / len(column_assignment.keys())
    
def neg_log_avg_score(similarities: List[Tuple[Tuple[str, str], float]], column_assignment: Dict[str, str]) -> float:
    avg =  sum(
        [
            score for columns, score in similarities 
            if columns[0] in column_assignment and column_assignment[columns[0]] == columns[1]
        ]
    ) / len(column_assignment.keys())
    return -1 * math.log(avg)

def vocabulary_similarity(
    vocab1: Dict[str, Set[str]], 
    vocab2: Dict[str, Set[str]], 
    min_threshold: float = 0.01, 
    similarity_fn: Callable[[Set[str], Set[str]], float] = vocab_overlap
) -> Dict[Tuple[str, str], float]:
    similarities = {}

    for col1, tokens1 in vocab1.items():
        for col2, tokens2 in vocab2.items():
            similarity = similarity_fn(tokens1, tokens2)
            if similarity > min_threshold:
                similarities[(col1, col2)] = similarity

    return similarities

def get_most_similar(similarities: Dict[Tuple[str, str], float]) -> List[Tuple[Tuple[str, str], float]]: 
    similarities = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
    return similarities

def get_assignments(
    vocab: Dict[str, Set[str]], 
    df_vocab: Dict[str, Set[str]],
    min_threshold: float = 0.01, 
    scoring_fn: Callable[[List[Tuple[Tuple[str, str], float]], Dict[str, str]], float] = avg_score, 
    similarity_fn: Callable[[Set[str], Set[str]], float] = vocab_overlap
):
    """
    
    ToDo: Label based schema matching hinzufügen?
            Aber irgendwie gewichtet?
        Was mit Duplicate-Driven Schema Matching?

    """
    #for key, value in vocab.items():
    #    print(f"{key}: {len(value)}")
        
    similarities = vocabulary_similarity(vocab, df_vocab, min_threshold, similarity_fn)    
    similarities = get_most_similar(similarities)
    #print(similarities, sep="\n")
    
    column_assignment = assign_columns(similarities)
    #print(column_assignment)
    
    if len(column_assignment) > 0:
        ruleset_score = scoring_fn(similarities, column_assignment)
    else:
        ruleset_score = -1
    #print(f"Score: {ruleset_score}")
    
    return column_assignment, ruleset_score

def count_rules(rules: Dict[str, Dict[str, Dict[str, str]]]):
    len_to_count = dict()
    
    for value in rules.values():
        reason = value["reason"]
        len_to_count[len(reason)] = len_to_count.get(len(reason), 0) + 1   
    
    print(len_to_count)
    return len_to_count
    
def filter_rules(rules: Dict[str, Dict[str, Dict[str, str]]], lhs_assignment, rhs_assignment) -> Dict[str, Dict[str, Dict[str, str]]]:
    rules_dict = dict()
        
    for key, value in rules.items():
        reason = value["reason"]
        result = value["result"]
        rule_applicable = True    
        
        for lhs_key in reason.keys():
            if lhs_key not in lhs_assignment:
                rule_applicable = False
                break
                
        if not rule_applicable:
            continue
        
        for rhs_key in result.keys():
            if rhs_key not in rhs_assignment:
                rule_applicable = False
                break
        
        if rule_applicable:
            rules_dict[key] = value
    
    return rules_dict

def get_applicable_rules(data: pd.DataFrame, rules: Dict[str, Dict[str, Dict[str, str]]], lhs_assignment, rhs_assignment) -> Dict[str, Dict[str, Dict[str, str]]]:
    rules_dict = dict()
    data = data.copy()
    data = data.reset_index(drop=True)
        
    for key, value in rules.items():
        reason = value["reason"]
        result = value["result"]
        rule_applicable = True
        mask = pd.Series([True] * len(data))
        
        for lhs_key, lhs_value in reason.items():
            
            if lhs_key not in lhs_assignment:
                rule_applicable = False
                break
            
            mask &= data[lhs_assignment[lhs_key]] == lhs_value
                
        if not rule_applicable or data[mask].empty:
            continue
        
        for rhs_key in result.keys():
            if rhs_key not in rhs_assignment:
                rule_applicable = False
                break
        
        if rule_applicable:
            rules_dict[key] = value
    
    return rules_dict

def calc_rule_coverage(data: pd.DataFrame, rules: Dict[str, Dict[str, Dict[str, str]]]) -> float:
    #start = time.time()
    lhs_vocab, rhs_vocab = get_rule_vocab(rules)        
    df_vocab = get_dataframe_vocab(data)
    
    lhs_assignment, _ = get_assignments(lhs_vocab, df_vocab, 0.5, scoring_fn=neg_log_avg_score, similarity_fn=vocab_overlap)
    rhs_assignment, _ = get_assignments(rhs_vocab, df_vocab, 0.01, scoring_fn=neg_log_avg_score, similarity_fn=jaccard_similarity)
    
    applicable_rules = get_applicable_rules(data, rules, lhs_assignment, rhs_assignment)
    # applicable_rules = filter_rules(rules, lhs_assignment, rhs_assignment)
    coverage = len(applicable_rules) / len(rules)
    
    #print(f"{coverage * 100:.2f}% ({len(applicable_rules)}/{len(rules)}) rules applicable")
    #end = time.time()
    #print(f"Took {end - start} seconds for rule to column matching")
    return coverage

def experiment_col_removal(data: pd.DataFrame, rules: Dict[str, Dict[str, Dict[str, str]]]) -> Dict[int, List[float]]:
    results = dict()
    for i in range(len(data.columns)):
        results[i] = list()
        print(f"Calculating for {i} columns removed")
        for combination in tqdm.tqdm(list(combinations(data.columns, len(data.columns) - i))):
            results[i].append(calc_rule_coverage(data.drop(list(combination), axis=1), rules))
    return results

def experiment_row_removal(data: pd.DataFrame, rules: Dict[str, Dict[str, Dict[str, str]]], n_calculations: int = 5) -> Dict[float, List[float]]:
    results = dict()
    for remove_fraction in np.arange(0, 1, 0.1):
        remove_fraction = round(remove_fraction, 2)
        results[remove_fraction] = [
            calc_rule_coverage(data.sample(frac=1 - remove_fraction), rules) for _ in range(n_calculations)
        ]
    return results

def experiment_random_nan(data: pd.DataFrame, rules: Dict[str, Dict[str, Dict[str, str]]], n_calculations: int = 5) -> Dict[float, List[float]]:
    org_data = data.copy()
    results = dict()
    for nan_fraction in np.arange(0, 1, 0.1):
        nan_fraction = round(nan_fraction, 2)
        results[nan_fraction] = []
        for _ in range(n_calculations):
            cells = list(np.ndindex(data.shape))
            n = int(len(cells) * nan_fraction)
            random.shuffle(cells)
            indicies_to_nan = cells[:n]
            
            for index in indicies_to_nan:
                data.iloc[index] = np.nan
            
            results[nan_fraction].append(calc_rule_coverage(data, rules))
            data = org_data.copy()
        
    return results

def test_unique_combinations(df: pd.DataFrame) -> List[Tuple[str]]:
    column_names = df.columns.tolist()
    unique_combinations = []
    
    for r in range(1, len(column_names) + 1):
        for columns in combinations(column_names, r):
            if df[list(columns)].duplicated().sum() == 0:
                unique_combinations.append(columns)
                
    return unique_combinations

def uniqueness_score(data: pd.DataFrame) -> float:
    uniqueness_df = pd.DataFrame(index=data.columns, columns=["uniqueness"])
    for column in data.columns:
        uniqueness_df.loc[column] = data[column].nunique() / data.shape[0]
    return uniqueness_df["uniqueness"].mean()

def type_token_ratio(data: pd.DataFrame) -> float:
    vocab = set()
    for column in data.columns:
        vocab.update(data[column].unique())
        
    return len(vocab) / data.shape[0] * data.shape[1]