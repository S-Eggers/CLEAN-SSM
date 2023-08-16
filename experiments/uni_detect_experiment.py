import os
import time
import pickle
import logging
import traceback
import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations
from typing import Dict, Any, Tuple
from .experiment import Experiment
from insert_error import insert_error_unidetect
import methods.uni_detect_fatemeh.ud_utils as udt
import methods.uni_detect_fatemeh.uv.uniqueness as uv
import methods.uni_detect_fatemeh.fd_violations.fd as fd
import methods.uni_detect_fatemeh.Spelling.spelling_errors as se


class UniDetectExperiment(Experiment):
    def __init__(self, error_rate: float, method_name: str, dataset: str, error_generator: int = 1):
        super().__init__(error_rate, method_name, dataset, error_generator)
        self.base_dir = os.path.join(os.getcwd(), "methods", "uni_detect_fatemeh")
        self.precision = 0
        self.recall = 0
        self.f1 = 0
        self.runtime = np.inf
        
    def run(self, **kwargs):
        logging.getLogger().setLevel(logging.INFO)
        dataset_path = os.path.join(self.base_dir, "datasets", "artificial_lake", self.dataset, "clean.csv")
        self.clean_path = dataset_path
        clean_dataset = pd.read_csv(dataset_path)
        dirty_dataset, clean_dataset, errors = insert_error_unidetect(clean_dataset, self.error_rate, self.base_dir)
        dirty_dataset = dirty_dataset.drop(columns=["Label"])
        dirty_path = os.path.join(os.path.dirname(dataset_path), "unidetect_dirty.csv")
        dirty_dataset.to_csv(dirty_path, index=False)
        self._get_tokens_dict(dirty_path, os.path.join(self.base_dir, "datasets", "artificial_lake", self.dataset))
        path_config = {
            "tokens_path": os.path.join(self.base_dir, "datasets", "artificial_lake", self.dataset, "tokens_dict.pkl"),
            "fd_path": "/home/WDC_corpus/uni-detect-results/pretrained-1m/fd_tables_results_1m.pickle",
            "se_path": "/home/WDC_corpus/uni-detect-results/pretrained-1m/se_tables_results_1m.pickle",
            "uv_path": "/home/WDC_corpus/uni-detect-results/pretrained-1m/uv_tables_results_1m.pickle",
            "output_path": os.path.join(self.base_dir, "datasets", "artificial_lake", self.dataset)
        }
        start_time = time.time()
        
        num_errors = np.count_nonzero(errors)
        fd_results = self._fd_violations(dirty_dataset, path_config)
        tp = fd_results[fd_results["error"] == True].shape[0]
        fp = fd_results[fd_results["error"] == False].shape[0]     
        
        uv_results = self._uv(dirty_dataset, path_config)
        tp += uv_results[uv_results["error"] == True].shape[0]
        fp += uv_results[uv_results["error"] == False].shape[0]
        
        se_results = self._se(dirty_dataset, path_config)
        tp += se_results[se_results["error"] == True].shape[0]
        fp += se_results[se_results["error"] == False].shape[0]
        
        end_time = time.time()
        fn = num_errors - tp
        self.precision = tp / (tp + fp) if tp + fp != 0 else 0
        self.recall = tp / (tp + fn) if tp + fn != 0 else 0
        self.f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall) if self.precision + self.recall != 0 else 0
        self.runtime = end_time - start_time  
        print(f"Finished Uni Detect pipeline, took {self.runtime}s")
        
    def _get_table_tokens_dict(self, table_path: str, file_type: str, n_cells_limit: int, output_path: str) -> set:
        """
        This function calculates the tokens dictionary for a single table.
        parameters
        ----------
        :param table_pat: str
            The path to the table.
        :param file_type: str
            The file type of the table.
        :return: set
            The tokens dictionary for the table.
        """
        try:
            logging.info(f"Start getting tokens dict for table {table_path}")
            tokens_dict = {}
            if file_type == "parquet":
                train_df = pd.read_parquet(table_path)
            else:
                train_df = pd.read_csv(table_path).astype(str)
            logging.info(f"Read table {table_path}")
            if train_df.shape[0] * train_df.shape[1] < n_cells_limit:
                # concatenate all the columns into a single Series
                text_series = train_df.apply(lambda x: ' '.join(x.astype(str).values), axis=1)
                text_series = text_series.astype(str)
                # tokenize the text in each row of the Series and concatenate the resulting Series
                tokens = text_series.str.split(expand=True).stack()
                # count the frequency of each token
                token_counts = tokens.value_counts()
                # create a dictionary with tokens as keys and their frequency as values
                tokens_dict = token_counts.to_dict()
                logging.info(f"Finish getting tokens dict for table {table_path}")
                with open(os.path.join(output_path, f'tokens_dict_{os.path.basename(table_path).removesuffix(".csv")}.pkl'), 'wb') as f:
                    pickle.dump(tokens_dict, f)
                return tokens_dict
            else:
                logging.info(f"Skipping getting tokens dict for table {table_path}")
                return {}
        except Exception as e:
            logging.info(f"Exception in getting tokens dict for table {table_path}, {e}")
        
    def _get_tokens_dict(self, table_path: str, output_path: str) -> Dict:
        """
        This function calculates the tokens dictionary. Tokens_dict is a dictionary that maps each token to its frequency / number of tables.
        parameters
        ----------
        :param train_path: str
            The path to the train data.
        :param output_path: str
            The path to save the tokens dictionary.
        :param file_type: str
            The file type of the train data.
        :param executor: ThreadPoolExecutor
            The executor to use for parallelization.
        :return: dict
            The tokens dictionary.
        """
        logging.info(f"Start getting tokens dict")
        # table_path: str, file_type: str, n_cells_limit: int, output_path: str
        executor_features = [self._get_table_tokens_dict(table_path, "csv", 10000000000, output_path)]
        tokens_dict = None
        td = [feature for feature in executor_features]
        aggregated_tokens_counter = sum((Counter(token_dict) for token_dict in td), Counter())
        tokens_dict = {k: v for k, v in aggregated_tokens_counter.items()}
        logging.info(f"Finish getting tokens dict")
        with open(os.path.join(output_path, 'tokens_dict.pkl'), 'wb') as f:
                pickle.dump(tokens_dict, f)
        return tokens_dict
        
    def _se(self, data: pd.DataFrame, config: Dict[str, str]) -> pd.DataFrame:
        path = self.clean_path
        ground_truth = True 
        config['ground_truth_path'] = path
        
        with open(config['se_path'], 'rb') as f:
            se_dict = pickle.load(f)
            print("fd_dict loaded")
            
        output_path = config['output_path']
        tables_output_path = os.path.join(output_path, "se_tables_test_results")
        if not os.path.exists(tables_output_path):
            os.makedirs(tables_output_path)

        spelling_results = pd.DataFrame(columns=['error_type', 'path', 'col_name', 'row_idx', 'col_idx', 'LR', 'value',
                                                'ground_truth', 'error'])
        t_init = time.time()
        
        spelling_results_table = pd.DataFrame(columns=['error_type', 'path', 'col_name', 'row_idx', 'col_idx', 'LR', 'value',
                                            'ground_truth', 'error'])
        try:
            test_df = data.copy()
            
            for test_column_idx, test_column_name in enumerate(test_df.columns):
                    test_column = test_df[test_column_name]
                    mpd_d, mpd_do, avg_len_diff_tokens, idx_p = se.perturbation(test_column)
                    p_dt, p_dot = 0, 0
                    number_of_rows_range = udt.get_range_count(test_column.count())
                    range_mpd = udt.get_range_mpd(avg_len_diff_tokens)

                    if not np.isnan(mpd_d) and not np.isnan(mpd_do) and mpd_do != mpd_d:
                        for col_id in se_dict.keys():
                            str_col = test_column.astype(str)
                            test_col_dtype = "alnumeric" if str_col.str.isalnum().all() else test_column.dtype,
                            if se_dict[col_id]["d_type"] != test_col_dtype:
                                continue
                            if se_dict[col_id]["number_of_rows_range"] != number_of_rows_range:
                                continue
                            if se_dict[col_id]["range_mpd"] != range_mpd:
                                continue

                            train_mpd_d, train_mpd_do = se_dict[col_id]["mpd"], se_dict[col_id]["mpd_p"]
                            if train_mpd_d <= mpd_do:
                                p_dt = p_dt + 1
                            if train_mpd_d <= mpd_d and train_mpd_do >= mpd_do:
                                p_dot = p_dot + 1
                        lr = p_dot / p_dt if p_dt else -np.inf
                        if ground_truth:
                            ground_truth_path = config['ground_truth_path']
                            clean_df = pd.read_csv(ground_truth_path)
                            test_column_name_clean = clean_df.columns[test_column_idx]
                            correct_value = clean_df[test_column_name_clean].values.astype(str)[idx_p]
                            dirty_value = test_df[test_df.columns[test_column_idx]].values.astype(str)[idx_p]
                        else:
                            correct_value = "----Ground Truth is Not Available----"
                        if idx_p and lr != -np.inf:
                            error = str(correct_value) != str(dirty_value)
                            row = ["spelling", path, test_column_name, idx_p, list(test_df.columns).index(test_column_name), lr,
                                test_column.loc[idx_p], correct_value, error]
                                        
                            spelling_results.loc[len(spelling_results)] = row
                            spelling_results_table.loc[len(spelling_results_table)] = row
            
            with open(os.path.join(tables_output_path, (os.path.basename(path).removesuffix(f'.csv') + ".pickle")), 'wb') as f:
                pickle.dump(spelling_results_table, f)
        except Exception as e:
            print(f"Error in {path}: {e}")
            traceback.print_exc()
            

        spelling_results_table.to_csv(os.path.join(output_path, "se_test_results.csv"))
        t_f = time.time() - t_init
        print(f"SE Test finished in {t_f} seconds")
        print(spelling_results_table)
        return spelling_results_table

    def _uv(self, data: pd.DataFrame, config: Dict[str, str]) -> pd.DataFrame:
        """
        Uniqueness Violation Test
        """
        path = self.clean_path
        ground_truth = True
        config['ground_truth_path'] = path
        
        with open(config['uv_path'], 'rb') as f:
            uv_dict = pickle.load(f)
            print("fd_dict loaded")
        
        with open(config['tokens_path'], 'rb') as f:
            tokens_dict = pickle.load(f)
            print("tokens_dict loaded")

        output_path = config['output_path']
        tables_output_path = os.path.join(output_path, "uv_tables_test_results")
        if not os.path.exists(tables_output_path):
            os.makedirs(tables_output_path)

        uniqueness_results = pd.DataFrame(
            columns=['error_type', 'path', 'col_name', 'row_idx', 'col_idx', 'LR', 'value', 'correct_value', 'error'])

        t_init = time.time()
   
        uniqueness_results_table = pd.DataFrame(
        columns=['error_type', 'path', 'col_name', 'row_idx', 'col_idx', 'LR', 'value', 'correct_value', 'error'])

        try:
            test_df = data.copy()

            for test_column_idx, test_column_name in enumerate(test_df.columns):
                try:
                    test_column = test_df[test_column_name]
                    left_ness = list(test_df.columns).index(test_column_name)
                    uniqueness_d, uniqueness_do, duplicate_idx = uv.perturbation(test_column)
                    number_of_rows_range = udt.get_range_count(test_column.count())
                    avg_col_pre = udt.get_prev_range(tokens_dict, test_column)
                except Exception as e:
                    logging.info(f"Error in {path}: {e}")
                    continue

                try:
                    p_dt, p_dot = 0, 0
                    if duplicate_idx != -1 and uniqueness_d != uniqueness_do:
                        for col_id in uv_dict.keys():
                            if uv_dict[col_id]["d_type"] != test_column.dtype:
                                continue
                            if uv_dict[col_id]["number_of_rows_range"] != number_of_rows_range:
                                continue
                            if uv_dict[col_id]["left_ness"] != left_ness:
                                continue
                            if uv_dict[col_id]["avg_col_pre"] != avg_col_pre:
                                continue

                            train_uniqueness_d, train_uniqueness_do = uv_dict[col_id]["ur"], \
                                                                    uv_dict[col_id]["ur_p"]

                            if train_uniqueness_d <= uniqueness_do:
                                p_dt = p_dt + 1
                            if train_uniqueness_d <= uniqueness_d and train_uniqueness_do >= uniqueness_do:
                                p_dot = p_dot + 1

                        lr = p_dot / p_dt if p_dt else -np.inf
                        if ground_truth:
                            ground_truth_path = config['ground_truth_path']
                            clean_df = pd.read_csv(ground_truth_path)
                            test_column_name_clean = clean_df.columns[test_column_idx]
                            correct_value = clean_df[test_column_name_clean].values.astype(str)[duplicate_idx]
                            dirty_value = test_df[test_df.columns[test_column_idx]].values.astype(str)[duplicate_idx]
                        else:
                            correct_value = "----Ground Truth is Not Available----"
                        if duplicate_idx and lr != -np.inf:
                            row = ["uniqueness", path, test_column_name, duplicate_idx,
                                list(test_df.columns).index(test_column_name), lr,
                                test_column.loc[duplicate_idx], correct_value, str(correct_value) != str(dirty_value)]

                            uniqueness_results.loc[len(uniqueness_results)] = row
                            uniqueness_results_table.loc[len(uniqueness_results_table)] = row
                except Exception as e:
                    logging.info(f"Error in {path}: {e}")
                    traceback.print_exc()
                    continue
            with open(os.path.join(tables_output_path, (os.path.basename(path).removesuffix(f'.csv') + ".pickle")), 'wb') as f:
                pickle.dump(uniqueness_results_table, f)
            logging.info(f"UV Test finished for {path}")
        except Exception as e:
            logging.info(f"Error in {path}: {e}")
            traceback.print_exc()

        uniqueness_results_table.to_csv(os.path.join(output_path, "uv_test_results.csv"))
        t_f = time.time() - t_init
        print(f"UV Test finished in {t_f} seconds")
        print(uniqueness_results_table)
        return uniqueness_results_table
    
    def _fd_violations(self, data: pd.DataFrame, config: Dict[str, str]) -> pd.DataFrame:
        """
        Functional Dependency Violation Test
        """
        path = self.clean_path
        ground_truth = True 
        config['ground_truth_path'] = path
        
        with open(config['fd_path'], 'rb') as f:
            fd_dict = pickle.load(f)
            print("fd_dict loaded")
        
        with open(config['tokens_path'], 'rb') as f:
            tokens_dict = pickle.load(f)
            print("tokens_dict loaded")
        
        output_path = config['output_path']
        tables_output_path = os.path.join(output_path, "fd_tables_test_results")
        if not os.path.exists(tables_output_path):
            os.makedirs(tables_output_path)

        fd_results = pd.DataFrame(
            columns=['error_type', 'path', 'col_1_name', 'col_2_name', 'row_idx', 'col_1_idx', 'col_2_idx', 'LR',
                    'value_1', 'value_2', 'correct_value_1', 'correct_value_2', 'error'])
        test_column_1, test_column_2, left_ness_1, left_ness_2, fd_d, fd_do, idx_d = np.nan, np.nan, np.nan, np.nan,\
                                                                                    np.nan, np.nan, np.nan
        t_init = time.time()
   
        fd_results_table = pd.DataFrame(
        columns=['error_type', 'path', 'col_1_name', 'col_2_name', 'row_idx', 'col_1_idx', 'col_2_idx', 'LR',
                'value_1', 'value_2', 'correct_value_1', 'correct_value_2', 'error'])
        try:
            test_df = data.copy()

            for pair in combinations(test_df.columns, 2):
                test_column_1, test_column_2 = test_df[pair[0]], test_df[pair[1]]
                fd_d, fd_do, idx_d = fd.perturbation(test_column_1, test_column_2)
                p_dt, p_dot = 0, 0
                if idx_d != -1 and fd_do != fd_d:
                    left_ness_1 = list(test_df.columns).index(pair[0])
                    left_ness_2 = list(test_df.columns).index(pair[1])
                    number_of_rows_range = udt.get_range_count(test_column_1.count())
                    avg_col_pre_1 = udt.get_prev_range(tokens_dict, test_column_1)
                    avg_col_pre_2 = udt.get_prev_range(tokens_dict, test_column_2)

                    for cols_id in fd_dict.keys():
                        if fd_dict[cols_id]["d_type_1"] != test_column_1.dtype or\
                                fd_dict[cols_id]["d_type_2"] != test_column_2.dtype:
                            continue
                        if fd_dict[cols_id]["number_of_rows_range"] != number_of_rows_range:
                            continue
                        if fd_dict[cols_id]["left_ness_1"] != left_ness_1 or fd_dict[cols_id]["left_ness_2"] != left_ness_2:
                            continue
                        if fd_dict[cols_id]["avg_col_pre_1"] != avg_col_pre_1 or\
                                fd_dict[cols_id]["avg_col_pre_2"] != avg_col_pre_2 :
                            continue

                        train_fd_d, train_fd_do = fd_dict[cols_id]["fd"], fd_dict[cols_id]["fd_p"]
                        if train_fd_d != train_fd_do:
                            if train_fd_d <= fd_do:
                                p_dt = p_dt + 1
                            if train_fd_d <= fd_d and train_fd_do >= fd_do:
                                p_dot = p_dot + 1

                    lr = p_dot / p_dt if p_dt and p_dot else -np.inf

                    if lr != -np.inf:
                        if idx_d != -1:
                            if ground_truth:
                                ground_truth_path = config['ground_truth_path']
                                clean_df = pd.read_csv(ground_truth_path)
                                clean_col_idx_0 = list(test_df.columns).index(pair[0])
                                clean_col_idx_1 = list(test_df.columns).index(pair[1])
                                correct_value_1 = clean_df[clean_df.columns[clean_col_idx_0]].values.astype(str)[idx_d]
                                correct_value_2 = clean_df[clean_df.columns[clean_col_idx_1]].values.astype(str)[idx_d]
                                dirty_value_1 = test_column_1.values.astype(str)[idx_d]
                                dirty_value_2 = test_column_2.values.astype(str)[idx_d]
                                
                            else:
                                correct_value_1 = "----Ground Truth is Not Available----"
                                correct_value_2 = "----Ground Truth is Not Available----"

                            row = ["fd", path, pair[0], pair[1], idx_d, list(test_df.columns).index(pair[0]),
                                list(test_df.columns).index(pair[1]),
                                lr, test_column_1.loc[idx_d], test_column_2.loc[idx_d],
                                correct_value_1, correct_value_2,
                                str(correct_value_1) != str(dirty_value_1) or str(correct_value_2) != str(dirty_value_2)]
                            fd_results.loc[len(fd_results)] = row
                            fd_results_table.loc[len(fd_results_table)] = row

                with open(os.path.join(tables_output_path, (os.path.basename(path).removesuffix(f'.csv') + ".pickle")), 'wb') as f:
                    pickle.dump(fd_results_table, f)
                print(f"fd_test_results_table saved in {tables_output_path}")
        
        except Exception as e:
            print(f"Error in {path}: {e}")
            traceback.print_exc()

        fd_results_table.to_csv(os.path.join(output_path, "fd_test_results.csv"))
        t_f = time.time() - t_init
        print(f"FD Test finished in {t_f} seconds")
        print(fd_results_table)
        return fd_results_table
    
    @staticmethod
    def worker(kwargs: Dict[str, Any] = dict()):
        return None
     
    def result(self) -> Tuple[float, float, float, float]:
        return self.precision, self.recall, self.f1, self.runtime